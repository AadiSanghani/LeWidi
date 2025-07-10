import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from train_par_improved import ParDemogModel


def get_age_bin(age):
    """Convert age to age bin."""
    if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
        return "<UNK>"
    try:
        age_int = int(age)
        if age_int < 25:
            return "18-24"
        elif age_int < 35:
            return "25-34"
        elif age_int < 45:
            return "35-44"
        elif age_int < 55:
            return "45-54"
        else:
            return "55+"
    except (ValueError, TypeError):
        return "<UNK>"


def build_input(example: dict, tokenizer: AutoTokenizer) -> str:
    """Recreate the input string exactly like during training."""
    question1 = example["text"].get("Question1", "")
    question2 = example["text"].get("Question2", "")
    return f"{question1} {tokenizer.sep_token} {question2}"


def process_demographic_data(example: dict, annot_meta: dict, vocab: dict):
    """Process demographic data for a single example - matches training script logic."""
    PAD_IDX = 0
    UNK_IDX = 1
    
    # Use the same reduced field set as training script
    FIELD_KEYS = {
        "age": "Age",  # Will be binned
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
        "age": "Age",  # Will be binned
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
    model = ParDemogModel(
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
        dataset_name = Path(args.test_file).stem.replace("_clear", "")
        suffix = "_pe.tsv" if args.task == "B" else "_soft.tsv"
        args.output_tsv = f"{dataset_name}{suffix}"

    with open(args.output_tsv, "w", encoding="utf-8") as out_f:
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
                # Task A: Output probability distribution for soft labels
                # Paraphrase has 11 classes: probabilities for Likert scale -5 to +5
                out_probs = probs.tolist()
                # Ensure we output exactly 11 probabilities for Likert scale
                if len(out_probs) < 11:
                    pad = [0.0] * (11 - len(out_probs))
                    out_probs = pad + out_probs
                elif len(out_probs) > 11:
                    out_probs = out_probs[:11]
                
                # round to 10 decimals and fix any rounding drift
                out_probs = [round(p, 10) for p in out_probs]
                drift = 1.0 - sum(out_probs)
                if abs(drift) > 1e-10:
                    # add drift to the max prob to keep list summing to 1
                    idx_max = max(range(len(out_probs)), key=out_probs.__getitem__)
                    out_probs[idx_max] = round(out_probs[idx_max] + drift, 10)
                prob_str = ",".join(f"{p:.10f}" for p in out_probs)
                out_f.write(f"{ex_id}\t[{prob_str}]\n")
            else:
                # Task B: repeat predicted rating for each annotator (simple baseline)
                ann_list = ex.get("annotators", "").split(",") if ex.get("annotators") else []
                # Convert prob distribution to single rating (argmax + convert back to -5:5 scale)
                rating_idx = torch.argmax(probs).item()
                predicted_rating = rating_idx - 5  # Convert from 0:10 back to -5:5
                preds = ", ".join(str(predicted_rating) for _ in ann_list)
                out_f.write(f"{ex_id}\t[{preds}]\n")

    print(f"Saved submission file to {args.output_tsv}")
    print("To submit: zip -j res.zip", args.output_tsv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission TSV for Paraphrase dataset (Task A or B).")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--base_model_name", type=str, default="roberta-large", help="Base model name (e.g., roberta-large)")
    parser.add_argument("--test_file", type=str, default="dataset/Paraphrase/Paraphrase_test_clear.json", help="Path to *_test_clear.json file")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--task", choices=["A", "B"], default="A", help="Which task: A (soft) or B (perspectivist)")
    parser.add_argument("--output_tsv", type=str, default=None, help="Optional output filename")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    main(args)