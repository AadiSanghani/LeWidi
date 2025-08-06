import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import the model class from the no-demographics training script
from train_par_no_demog import ParModel


def load_model_and_metadata(model_dir):
    """Load trained model and its metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Training metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata: {metadata['training_config']}")
    print(f"Model type: {metadata.get('model_type', 'unknown')}")
    
    return metadata, device


def build_input(example: dict, tokenizer: AutoTokenizer) -> str:
    """Recreate the input string exactly like during training."""
    question1 = example["text"].get("Question1", "")
    question2 = example["text"].get("Question2", "")
    return f"{question1} [SEP] {question2}".strip()


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


def generate_task_a_submission(model, tokenizer, test_data, device, output_file):
    """Generate Task A (soft-label) submission."""
    print("Generating Task A submission...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex_id, ex in tqdm(test_data.items(), desc="Predicting"):
            # Build input text
            text = build_input(ex, tokenizer)
            
            # Get prediction (no demographics)
            probs = predict_single_example(model, tokenizer, text, device)
            
            # Convert to list and ensure it sums to 1
            prob_list = probs.tolist()
            
            # Ensure we have exactly 11 probabilities (for Likert scale -5 to 5)
            if len(prob_list) != 11:
                print(f"Warning: Expected 11 classes, got {len(prob_list)} for example {ex_id}")
                if len(prob_list) < 11:
                    prob_list.extend([0.0] * (11 - len(prob_list)))
                else:
                    prob_list = prob_list[:11]
            
            # Normalize to ensure sum = 1
            prob_sum = sum(prob_list)
            if prob_sum > 0:
                prob_list = [p / prob_sum for p in prob_list]
            else:
                # If all probabilities are 0, use uniform distribution
                prob_list = [1.0 / 11] * 11
            
            # Round to avoid floating point precision issues
            prob_list = [round(p, 10) for p in prob_list]
            
            # Fix any remaining rounding drift
            drift = 1.0 - sum(prob_list)
            if abs(drift) > 1e-10:
                idx_max = max(range(len(prob_list)), key=prob_list.__getitem__)
                prob_list[idx_max] = round(prob_list[idx_max] + drift, 10)
            
            # Format as required: [p1, p2, ..., p11]
            prob_str = ",".join(f"{p:.10f}" for p in prob_list)
            f.write(f"{ex_id}\t[{prob_str}]\n")
    
    print(f"Task A submission saved to {output_file}")


def generate_task_b_submission(model, tokenizer, test_data, device, output_file):
    """Generate Task B (perspectivist) submission."""
    print("Generating Task B submission...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex_id, ex in tqdm(test_data.items(), desc="Predicting"):
            # Build input text
            text = build_input(ex, tokenizer)
            
            # Get prediction (no demographics)
            probs = predict_single_example(model, tokenizer, text, device)
            
            # Convert to single rating (argmax and convert to -5:5 scale)
            predicted_class = np.argmax(probs)
            predicted_rating = predicted_class - 5  # Convert from 0:10 to -5:5
            
            # For each annotator, predict the same rating (simple baseline)
            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            
            if not ann_list:
                # If no annotators listed, create a default prediction
                predictions = [str(predicted_rating)]
            else:
                predictions = [str(predicted_rating) for _ in ann_list]
            
            # Format as required: [pred1, pred2, pred3, ...]
            pred_str = ", ".join(predictions)
            f.write(f"{ex_id}\t[{pred_str}]\n")
    
    print(f"Task B submission saved to {output_file}")


def main(args):
    # Load model metadata
    metadata, device = load_model_and_metadata(args.model_dir)
    
    # Verify this is a no-demographics model
    if metadata.get('model_type') != 'no_demographics':
        print(f"Warning: This script is for no-demographics models, but model type is: {metadata.get('model_type')}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(metadata['training_config']['model_name'])
    
    # Create model with the same configuration as training (no demographics)
    model = ParModel(
        base_name=metadata['training_config']['model_name'],
        sbert_dim=metadata['training_config']['sbert_dim'],
        dropout_rate=metadata['training_config']['dropout_rate'],
        num_classes=metadata['training_config']['num_classes']
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
    
    if args.task == "A" or args.task == "both":
        output_file_a = os.path.join(args.output_dir, "Paraphrase_test_soft.tsv")
        generate_task_a_submission(model, tokenizer, test_data, device, output_file_a)
    
    if args.task == "B" or args.task == "both":
        output_file_b = os.path.join(args.output_dir, "Paraphrase_test_pe.tsv")
        generate_task_b_submission(model, tokenizer, test_data, device, output_file_b)
    
    print("\nSubmission generation completed!")
    print("Files ready for submission:")
    if args.task == "A" or args.task == "both":
        print(f"  Task A (soft): {output_file_a}")
    if args.task == "B" or args.task == "both":
        print(f"  Task B (perspectivist): {output_file_b}")
    print("\nTo create submission zip:")
    print("  zip -j paraphrase_no_demog_submission.zip *.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission files for Paraphrase Detection task (NO demographics)")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained model (best_model/)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both", help="Which task to generate submissions for")
    parser.add_argument("--output_dir", type=str, default="submissions", help="Output directory for submission files")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    args = parser.parse_args()
    main(args)