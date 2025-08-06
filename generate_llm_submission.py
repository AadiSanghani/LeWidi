#!/usr/bin/env python3
"""
Generate submission files from Mistral LLM evaluation results.
Converts discrete predictions to soft labels and per-annotator predictions.
"""

import json
import numpy as np
import argparse
from collections import Counter

def nli_prediction_to_soft_label(prediction):
    """Convert discrete NLI prediction to soft label distribution."""
    # Create a soft distribution centered on the predicted class
    labels = ["contradiction", "entailment", "neutral"]
    soft_label = {label: {"0": 0.9, "1": 0.1} for label in labels}  # Default low confidence
    
    if prediction in labels:
        # High confidence for predicted label, low for others
        soft_label[prediction] = {"0": 0.1, "1": 0.9}
        for other_label in labels:
            if other_label != prediction:
                soft_label[other_label] = {"0": 0.95, "1": 0.05}
    
    return soft_label

def paraphrase_rating_to_soft_label(rating):
    """Convert discrete paraphrase rating to soft label distribution."""
    # Convert rating from -5 to +5 scale to 0-10 indices
    rating = max(-5, min(5, rating))  # Clamp to valid range
    rating_index = rating + 5  # Convert to 0-10 range
    
    # Create soft distribution with high probability on predicted rating
    soft_label = {}
    for i in range(11):  # -5 to +5 ratings (11 total)
        actual_rating = i - 5
        if i == rating_index:
            soft_label[str(actual_rating)] = 0.8  # High confidence for predicted rating
        else:
            # Decay probability with distance
            distance = abs(i - rating_index)
            prob = max(0.01, 0.2 * np.exp(-distance))  # Exponential decay
            soft_label[str(actual_rating)] = prob
    
    # Normalize to sum to 1
    total_prob = sum(soft_label.values())
    soft_label = {k: v / total_prob for k, v in soft_label.items()}
    
    return soft_label

def generate_nli_submission(results_file, output_file, task="A"):
    """Generate NLI submission file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex_id, result in results.items():
            if 'error' in result:
                # Use neutral prediction for errors
                prediction = "neutral"
            else:
                prediction = result.get('prediction', 'neutral')
            
            if task == "A":
                # Task A: Generate soft labels
                soft_label = nli_prediction_to_soft_label(prediction)
                
                # Format as required: contradiction_prob,entailment_prob,neutral_prob
                probs = [
                    soft_label["contradiction"]["1"],
                    soft_label["entailment"]["1"], 
                    soft_label["neutral"]["1"]
                ]
                prob_str = ",".join(f"{p:.10f}" for p in probs)
                f.write(f"{ex_id}\t[{prob_str}]\n")
                
            elif task == "B":
                # Task B: Per-annotator predictions (repeat same prediction)
                original_data = result.get('annotations', {})
                if isinstance(original_data, dict):
                    num_annotators = len(original_data)
                else:
                    num_annotators = 3  # Default
                
                # Map prediction to label index
                label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
                pred_idx = label_map.get(prediction, 2)  # Default to neutral
                
                preds = ", ".join([str(pred_idx)] * num_annotators)
                f.write(f"{ex_id}\t[{preds}]\n")
    
    print(f"Generated NLI submission: {output_file}")

def generate_paraphrase_submission(results_file, output_file, task="A"):
    """Generate paraphrase detection submission file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex_id, result in results.items():
            if 'error' in result:
                # Use neutral rating (0) for errors
                prediction = 0
            else:
                prediction = result.get('prediction', 0)
            
            if task == "A":
                # Task A: Generate soft labels (probability distribution over -5 to +5)
                soft_label = paraphrase_rating_to_soft_label(prediction)
                
                # Format as 11 probabilities for ratings -5 to +5
                probs = [soft_label[str(i)] for i in range(-5, 6)]
                prob_str = ",".join(f"{p:.10f}" for p in probs)
                f.write(f"{ex_id}\t[{prob_str}]\n")
                
            elif task == "B":
                # Task B: Per-annotator predictions (repeat same rating)
                original_data = result.get('annotations', {})
                if isinstance(original_data, dict):
                    num_annotators = len(original_data)
                else:
                    num_annotators = 4  # Default for paraphrase task
                
                preds = ", ".join([str(prediction)] * num_annotators)
                f.write(f"{ex_id}\t[{preds}]\n")
    
    print(f"Generated paraphrase submission: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate submission files from LLM evaluation results")
    parser.add_argument("--results_file", type=str, required=True, help="LLM evaluation results JSON file")
    parser.add_argument("--task_type", choices=["nli", "paraphrase"], required=True, help="Task type")
    parser.add_argument("--submission_task", choices=["A", "B"], default="A", help="Submission task: A (soft) or B (perspectivist)")
    parser.add_argument("--output_file", type=str, help="Output submission file (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if not args.output_file:
        base_name = args.results_file.replace('.json', '')
        suffix = '_soft.tsv' if args.submission_task == 'A' else '_pe.tsv'
        args.output_file = f"{base_name}{suffix}"
    
    # Generate appropriate submission file
    if args.task_type == "nli":
        generate_nli_submission(args.results_file, args.output_file, args.submission_task)
    elif args.task_type == "paraphrase":
        generate_paraphrase_submission(args.results_file, args.output_file, args.submission_task)
    
    print(f"Submission file created: {args.output_file}")
    print("To create a submission archive:")
    print(f"zip -j submission.zip {args.output_file}")

if __name__ == "__main__":
    main()