import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import our custom model
from train_roberta_csc import CSCDemogModel, CSCDataset


def build_input(example: dict, tokenizer: AutoTokenizer) -> str:
    """Recreate the input string exactly like during training."""
    context = example["text"].get("context", "")
    response = example["text"].get("response", "")
    if context:
        return f"{context} {tokenizer.sep_token} {response}"
    return response


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Create a dummy dataset instance to get vocabulary info
    # We need this to know the vocab sizes for our custom model
    # For inference, we'll use a training file to build the vocabulary
    # since test files don't have valid soft labels
    train_file = args.test_file.replace("_test_clear.json", "_train.json")
    dummy_dataset = CSCDataset(
        train_file,  # Use train file to build vocab
        tokenizer,
        args.annot_meta,
        max_length=args.max_length
    )
    
    # Load our custom model
    model = CSCDemogModel(
        base_name=args.model_name,
        vocab_sizes=dummy_dataset.vocab_sizes,
        dem_dim=args.dem_dim,
        dropout_rate=args.dropout_rate
    )
    
    # Load the trained weights
    model_state = torch.load(
        f"{args.model_dir}/pytorch_model.bin", 
        map_location=device
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # We'll use UNK demographic values for inference since we don't have annotator-specific info
    unk_demographics = {
        f"{field}_ids": torch.tensor([CSCDataset.UNK_IDX], device=device)
        for field in dummy_dataset.active_field_keys
    }

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
            
            # Prepare demographic inputs for batch size 1
            batch_demographics = {}
            for field in dummy_dataset.active_field_keys:
                batch_demographics[f"{field}_ids"] = torch.tensor([CSCDataset.UNK_IDX], device=device)
            
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    **batch_demographics
                )
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

            if args.task == "A":
                # Ensure we output exactly num_bins probabilities (pad if needed)
                out_probs = probs.tolist()
                if len(out_probs) < args.num_bins:
                    pad = [0.0] * (args.num_bins - len(out_probs))
                    out_probs = pad + out_probs
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
                # Convert prob distribution to single rating (argmax + offset if bins=7)
                rating_idx = torch.argmax(probs).item()
                preds = ", ".join(str(rating_idx) for _ in ann_list)
                out_f.write(f"{ex_id}\t[{preds}]\n")

    print(f"Saved submission file to {args.output_tsv}")
    print("To submit: zip -j res.zip", args.output_tsv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission TSV for CSC dataset (Task A or B).")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to *_test_clear.json file")
    parser.add_argument("--task", choices=["A", "B"], default="A", help="Which task: A (soft) or B (perspectivist)")
    parser.add_argument("--output_tsv", type=str, default=None, help="Optional output filename")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_bins", type=int, default=7, help="Number of bins to output for task A.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    # Additional arguments needed for our custom model
    parser.add_argument("--model_name", type=str, default="roberta-large", help="Base model name used during training")
    parser.add_argument("--annot_meta", type=str, default="dataset/CSC/CSC_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of demographic embeddings (must match training)")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate (must match training)")
    
    args = parser.parse_args()
    main(args) 