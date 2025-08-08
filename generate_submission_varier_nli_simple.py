import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import the model class from the simplified training script
from train_varier_nli_simple import VariErrNLISimpleModel


def load_model_and_metadata(model_dir):
    """Load trained model and its metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata if available
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata.get('training_config', {})}")
    else:
        print(f"No metadata found at {metadata_path}, using defaults")
        metadata = {
            'training_config': {
                'model_name': 'roberta-large',
                'sbert_dim': 384,
                'dropout_rate': 0.3
            }
        }
    
    return metadata, device


def build_input(example: dict, tokenizer: AutoTokenizer) -> str:
    """Recreate the input string exactly like during training."""
    context = example["text"]["context"]
    statement = example["text"]["statement"]
    return f"{context} {tokenizer.sep_token} {statement}".strip()


def predict_single_example(model, tokenizer, text, device):
    """Make prediction for a single text example (no demographics needed)."""
    # Tokenize text
    enc = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    texts = [text]
    
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts
        )
        probabilities = torch.softmax(logits, dim=-1)
    
    return probabilities.cpu().numpy()[0]


def generate_soft_submission(model, tokenizer, test_data, device, output_file):
    """Generate soft-label submission for VariErrNLI."""
    print("Generating soft-label submission...")
    
    # Label mapping for NLI
    label_to_id = {"contradiction": 0, "entailment": 1, "neutral": 2}
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex_id, ex in tqdm(test_data.items(), desc="Predicting"):
            # Build input text
            text = build_input(ex, tokenizer)
            
            # Get prediction (no demographics)
            probs = predict_single_example(model, tokenizer, text, device)
            
            # Convert to list and ensure it sums to 1
            prob_list = probs.tolist()
            
            # Ensure we have exactly 3 probabilities (contradiction, entailment, neutral)
            if len(prob_list) != 3:
                print(f"Warning: Expected 3 classes, got {len(prob_list)} for example {ex_id}")
                if len(prob_list) < 3:
                    prob_list.extend([0.0] * (3 - len(prob_list)))
                else:
                    prob_list = prob_list[:3]
            
            # Normalize to ensure sum = 1
            prob_sum = sum(prob_list)
            if prob_sum > 0:
                prob_list = [p / prob_sum for p in prob_list]
            else:
                # If all probabilities are 0, use uniform distribution
                prob_list = [1.0 / 3] * 3
            
            # Round to avoid floating point precision issues
            prob_list = [round(p, 10) for p in prob_list]
            
            # Fix any remaining rounding drift
            drift = 1.0 - sum(prob_list)
            if abs(drift) > 1e-10:
                idx_max = max(range(len(prob_list)), key=prob_list.__getitem__)
                prob_list[idx_max] = round(prob_list[idx_max] + drift, 10)
            
            # Format as required: [p_contradiction, p_entailment, p_neutral]
            prob_str = ",".join(f"{p:.10f}" for p in prob_list)
            f.write(f"{ex_id}\t[{prob_str}]\n")
    
    print(f"Soft-label submission saved to {output_file}")


def generate_pe_submission(model, tokenizer, test_data, device, output_file):
    """Generate perspectivist submission for VariErrNLI."""
    print("Generating perspectivist submission...")
    
    # Label mapping for NLI
    label_to_id = {"contradiction": 0, "entailment": 1, "neutral": 2}
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex_id, ex in tqdm(test_data.items(), desc="Predicting"):
            # Build input text
            text = build_input(ex, tokenizer)
            
            # Get prediction (no demographics)
            probs = predict_single_example(model, tokenizer, text, device)
            
            # Convert to single label (argmax)
            predicted_class = np.argmax(probs)
            predicted_label = id_to_label[predicted_class]
            
            # For each annotator, predict the same label (simple baseline)
            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            
            if not ann_list:
                # If no annotators listed, create a default prediction
                predictions = [predicted_label]
            else:
                predictions = [predicted_label for _ in ann_list]
            
            # Format as required: [pred1, pred2, pred3, ...]
            pred_str = ", ".join(predictions)
            f.write(f"{ex_id}\t[{pred_str}]\n")
    
    print(f"Perspectivist submission saved to {output_file}")


def main(args):
    # Load model metadata
    metadata, device = load_model_and_metadata(args.model_dir)
    
    # Load tokenizer
    model_name = metadata.get('training_config', {}).get('model_name', 'roberta-large')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create model with the same configuration as training (no demographics)
    sbert_dim = metadata.get('training_config', {}).get('sbert_dim', 384)
    dropout_rate = metadata.get('training_config', {}).get('dropout_rate', 0.3)
    
    model = VariErrNLISimpleModel(
        base_name=model_name,
        sbert_dim=sbert_dim,
        dropout_rate=dropout_rate,
    )
    
    # Load the trained weights
    model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print("Model uses RoBERTa-Large + SBERT (no demographic embeddings)")
    
    # Load test data
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Generate submissions
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task == "soft" or args.task == "both":
        output_file_soft = os.path.join(args.output_dir, "VariErrNLI_test_soft.tsv")
        generate_soft_submission(model, tokenizer, test_data, device, output_file_soft)
    
    if args.task == "pe" or args.task == "both":
        output_file_pe = os.path.join(args.output_dir, "VariErrNLI_test_pe.tsv")
        generate_pe_submission(model, tokenizer, test_data, device, output_file_pe)
    
    print("\nSubmission generation completed!")
    print("Files ready for submission:")
    if args.task == "soft" or args.task == "both":
        print(f"  Soft-label: {output_file_soft}")
    if args.task == "pe" or args.task == "both":
        print(f"  Perspectivist: {output_file_pe}")
    print("\nTo create submission zip:")
    print("  zip -j varier_nli_simple_submission.zip *.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission files for VariErrNLI task (NO demographics)")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained model (best_model/)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--task", choices=["soft", "pe", "both"], default="both", help="Which task to generate submissions for")
    parser.add_argument("--output_dir", type=str, default="submissions", help="Output directory for submission files")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    args = parser.parse_args()
    main(args) 