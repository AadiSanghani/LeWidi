import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from train_varier_nli import VariErrNLIDemogModel


def get_age_bin(age):
    """Convert age to age bin."""
    if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
        return "<UNK>"
    try:
        age = float(age)
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        else:
            return "55+"
    except (ValueError, TypeError):
        return "<UNK>"


def build_input(example: dict, tokenizer: AutoTokenizer) -> str:
    """Recreate the input string exactly like during training."""
    context = example["text"].get("context", "")
    statement = example["text"].get("statement", "")
    return f"{context} {tokenizer.sep_token} {statement}"


def process_demographic_data(example: dict, annot_meta: dict, vocab: dict):
    """Process demographic data for a single example - matches training script logic."""
    PAD_IDX = 0
    UNK_IDX = 1
    
    # Use the same reduced field set as training script
    FIELD_KEYS = {
        "gender": "Gender", 
        "nationality": "Nationality",
        "education": "Education",
    }
    
    ann_str = example.get("annotators", "")
    ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
    if not ann_list:
        ann_list = []

    # For submission, we need to handle multiple annotators
    # We'll use the first annotator's demographics or UNK if no annotators
    if ann_list:
        # Use first annotator's demographics
        ann_tag = ann_list[0]
        meta = annot_meta.get(ann_tag, {})
        
        demographic_ids = {}
        for field, json_key in FIELD_KEYS.items():
            if field == "age":
                # Special handling for age - convert to age bin
                age_bin = get_age_bin(meta.get(json_key))
                idx = vocab[field].get(age_bin, UNK_IDX)
            else:
                val = str(meta.get(json_key, "")).strip()
                if val == "":
                    val = "<UNK>"
                idx = vocab[field].get(val, UNK_IDX)
            demographic_ids[field] = idx
    else:
        # No annotators - use UNK for all fields
        demographic_ids = {field: UNK_IDX for field in FIELD_KEYS}
            
    return demographic_ids


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Load annotator metadata and build vocabulary
    with open(args.annot_meta, "r", encoding="utf-8") as f:
        annot_meta = json.load(f)
    
    # Build vocabulary from annotator metadata - matches training script
    FIELD_KEYS = {
        "gender": "Gender", 
        "nationality": "Nationality",
        "education": "Education",
    }
    
    vocab = {
        field: {"<PAD>": 0, "<UNK>": 1}
        for field in FIELD_KEYS
    }

    # Build vocabulary from all annotators
    for ann_data in annot_meta.values():
        for field, json_key in FIELD_KEYS.items():
            if field == "age":
                # Special handling for age - convert to age bin
                age_bin = get_age_bin(ann_data.get(json_key))
                if age_bin not in vocab[field]:
                    vocab[field][age_bin] = len(vocab[field])
            else:
                val = str(ann_data.get(json_key, "")).strip()
                if val == "":
                    val = "<UNK>"
                if val not in vocab[field]:
                    vocab[field][val] = len(vocab[field])

    vocab_sizes = {field: len(v) for field, v in vocab.items()}
    
    # Load the custom model with the base model name
    model = VariErrNLIDemogModel(
        base_name=args.base_model_name,
        vocab_sizes=vocab_sizes,
        dem_dim=args.dem_dim,
        sbert_dim=args.sbert_dim
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(f"{args.model_dir}/pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()

    with open(args.test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build output filename if not provided
    if args.output_tsv is None:
        dataset_name = Path(args.test_file).stem.replace("_test_clear", "").replace("_clear", "")
        # Ensure we get just "VariErrNLI" from "VariErrNLI_test_clear"
        if "VariErr" in dataset_name:
            dataset_name = "VariErrNLI"
        suffix = "_test_pe.tsv" if args.task == "B" else "_test_soft.tsv"
        args.output_tsv = f"{dataset_name}{suffix}"

    with open(args.output_tsv, "w", encoding="utf-8", newline='\n') as out_f:
        for ex_id, ex in tqdm(data.items(), desc="Predicting"):
            text = build_input(ex, tokenizer)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            
            # Process demographic data
            dem_data = process_demographic_data(ex, annot_meta, vocab)
            
            # Convert to tensors - now single values, not lists
            demographic_inputs = {}
            for field in FIELD_KEYS:
                field_key = f"{field}_ids"
                demographic_inputs[field_key] = torch.tensor([dem_data[field]], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    texts=[text],  # SBERT needs list of texts
                    **demographic_inputs
                )
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

            if args.task == "A":
                # Task A: VariErrNLI expects format: [[c_0, c_1], [e_0, e_1], [n_0, n_1]]
                # Where each pair is [P(not_class), P(class)] for contradiction, entailment, neutral
                
                out_probs = probs.tolist()
                # Ensure we output exactly 3 probabilities for NLI
                if len(out_probs) < 3:
                    pad = [0.0] * (3 - len(out_probs))
                    out_probs = pad + out_probs
                elif len(out_probs) > 3:
                    out_probs = out_probs[:3]
                
                # Ensure probabilities sum to 1.0 exactly
                total = sum(out_probs)
                if total > 0:
                    out_probs = [p / total for p in out_probs]
                else:
                    out_probs = [1/3, 1/3, 1/3]
                
                # Our model outputs: [contradiction, entailment, neutral] 
                contradiction_prob = out_probs[0]
                entailment_prob = out_probs[1] 
                neutral_prob = out_probs[2]
                
                # Create the format: [[c_0, c_1], [e_0, e_1], [n_0, n_1]]
                prediction_format = [
                    [1.0 - contradiction_prob, contradiction_prob],  # contradiction [P(0), P(1)]
                    [1.0 - entailment_prob, entailment_prob],        # entailment [P(0), P(1)]
                    [1.0 - neutral_prob, neutral_prob]               # neutral [P(0), P(1)]
                ]
                
                # Convert to the exact string format
                out_f.write(f"{ex_id}\t{prediction_format}\n")
            else:
                # Task B: Perspectivist - format: [[ann1_c, ann2_c, ...], [ann1_e, ann2_e, ...], [ann1_n, ann2_n, ...]]
                # Where each inner list has predictions for each annotator for that class
                
                ann_str = ex.get("annotators", "")
                ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
                
                # Convert prob distribution to single label (argmax for NLI)
                label_idx = torch.argmax(probs).item()
                # label_idx: 0=contradiction, 1=entailment, 2=neutral
                
                # Create predictions for each annotator for each class
                num_annotators = len(ann_list) if ann_list else 4  # Default to 4 if no annotators
                
                # Initialize all predictions to 0
                predictions = [
                    [0] * num_annotators,  # contradiction predictions for each annotator
                    [0] * num_annotators,  # entailment predictions for each annotator  
                    [0] * num_annotators   # neutral predictions for each annotator
                ]
                
                # Set the predicted class to 1 for all annotators
                for i in range(num_annotators):
                    predictions[label_idx][i] = 1
                
                # Convert to the exact string format
                out_f.write(f"{ex_id}\t{predictions}\n")

    print(f"Saved submission file to {args.output_tsv}")
    print("To submit: zip -j res.zip", args.output_tsv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission TSV for VariErrNLI dataset (Task A or B).")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--base_model_name", type=str, default="roberta-large", help="Base model name (e.g., roberta-large)")
    parser.add_argument("--test_file", type=str, default="dataset/VariErrNLI/VariErrNLI_test_clear.json", help="Path to *_test_clear.json file")
    parser.add_argument("--annot_meta", type=str, default="dataset/VariErrNLI/VariErrNLI_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--task", choices=["A", "B"], default="A", help="Which task: A (soft) or B (perspectivist)")
    parser.add_argument("--output_tsv", type=str, default=None, help="Optional output filename")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    main(args)