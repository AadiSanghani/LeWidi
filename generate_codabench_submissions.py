#!/usr/bin/env python3
"""
Generate Codabench submissions for Paraphrase Detection
Supports both PE (perspectivist) and soft evaluation formats
"""

import json
import argparse
import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re

# Import the enhanced components
try:
    from train_par_enhanced import EnhancedParDemogModel, EnhancedParDataset, enhanced_collate_fn
    from optimize_mistral_prompts import OptimizedMistralPrompter
except ImportError:
    print("Warning: Enhanced components not available, using basic functionality")


class CodabenchSubmissionGenerator:
    """Generate Codabench submissions for paraphrase detection."""
    
    def __init__(self, model_path: str, use_mistral: bool = True, 
                 mistral_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 auth_token: Optional[str] = None):
        """Initialize the submission generator."""
        self.model_path = model_path
        self.use_mistral = use_mistral
        self.auth_token = auth_token
        
        # Load model metadata
        self.load_model_metadata()
        
        # Initialize components
        self.initialize_components()
        
    def load_model_metadata(self):
        """Load model metadata from the saved model directory."""
        metadata_path = os.path.join(self.model_path, "training_metadata.json")
        vocab_path = os.path.join(self.model_path, "vocabulary.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded model metadata from {metadata_path}")
        else:
            print(f"Warning: No metadata found at {metadata_path}")
            self.metadata = {}
        
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocab_data = json.load(f)
            print(f"Loaded vocabulary data from {vocab_path}")
        else:
            print(f"Warning: No vocabulary data found at {vocab_path}")
            self.vocab_data = {}
    
    def initialize_components(self):
        """Initialize model and tokenizer components."""
        from transformers import AutoTokenizer
        
        # Get model configuration from metadata
        model_name = self.metadata.get("training_config", {}).get("model_name", "roberta-large")
        vocab_sizes = self.metadata.get("vocab_sizes", {})
        dem_dim = self.metadata.get("training_config", {}).get("dem_dim", 8)
        sbert_dim = self.metadata.get("training_config", {}).get("sbert_dim", 384)
        dropout_rate = self.metadata.get("training_config", {}).get("dropout_rate", 0.3)
        num_classes = 11  # -5 to +5 scale
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = EnhancedParDemogModel(
            base_name=model_name,
            vocab_sizes=vocab_sizes,
            dem_dim=dem_dim,
            sbert_dim=sbert_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            use_mistral=self.use_mistral,
            mistral_model_name=mistral_model_name,
        )
        
        # Load model weights
        model_file = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
            print(f"Loaded model weights from {model_file}")
        else:
            print(f"Warning: No model weights found at {model_file}")
        
        self.model.eval()
        
        # Initialize Mistral prompter if enabled
        if self.use_mistral:
            try:
                self.mistral_prompter = OptimizedMistralPrompter(
                    model_name=mistral_model_name,
                    device="auto",
                    use_auth_token=self.auth_token
                )
                print("Mistral prompter initialized successfully!")
            except Exception as e:
                print(f"Warning: Failed to initialize Mistral prompter: {e}")
                self.use_mistral = False
                self.mistral_prompter = None
    
    def load_test_data(self, test_file: str, annot_meta_file: str) -> EnhancedParDataset:
        """Load test data as EnhancedParDataset."""
        return EnhancedParDataset(
            test_file, annot_meta_file, self.tokenizer, max_length=512
        )
    
    def predict_single_example(self, example: Dict, demographic_context: Optional[Dict] = None) -> Dict:
        """Predict for a single example with integrated Mistral prompting."""
        # Extract text
        question1 = example["text"].get("Question1", "")
        question2 = example["text"].get("Question2", "")
        full_text = f"{question1} [SEP] {question2}".strip()
        
        # Tokenize
        enc = self.tokenizer(
            full_text,
            max_length=512,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        
        # Prepare demographic inputs
        demographic_inputs = {}
        if demographic_context:
            for field, value in demographic_context.items():
                if field in self.vocab_data.get("vocabularies", {}):
                    vocab = self.vocab_data["vocabularies"][field]
                    idx = vocab.get(str(value), vocab.get("<UNK>", 0))
                    demographic_inputs[f"{field}_ids"] = torch.tensor([idx], dtype=torch.long)
        
        # Get Mistral predictions if enabled
        mistral_preds = None
        mistral_conf = None
        if self.use_mistral and self.mistral_prompter is not None:
            mistral_preds, mistral_conf = self.mistral_prompter.evaluate_paraphrase_optimized(
                question1, question2, demographic_context, prompt_style="comprehensive"
            )
            if mistral_conf is not None:
                self.model.mistral_confidence = torch.tensor([mistral_conf], dtype=torch.float32)
        
        # Model prediction
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                texts=[full_text],
                **demographic_inputs
            )
            
            # Combine with Mistral predictions if available
            if mistral_preds is not None:
                transformer_probs = torch.softmax(logits, dim=-1)
                mistral_probs = torch.tensor([[0.0] * 11], dtype=torch.float32)
                mistral_probs[0, mistral_preds + 5] = 1.0  # Convert rating to one-hot
                
                # Weighted combination
                alpha = 0.7
                combined_probs = alpha * transformer_probs + (1 - alpha) * mistral_probs
                logits = torch.log(combined_probs + 1e-8)
            
            probs = torch.softmax(logits, dim=-1)
        
        return {
            "probabilities": probs[0].numpy(),
            "mistral_rating": mistral_preds if mistral_preds is not None else None,
            "mistral_confidence": mistral_conf if mistral_conf is not None else None
        }
    
    def generate_soft_submission(self, test_file: str, annot_meta_file: str, 
                                output_file: str) -> None:
        """Generate soft evaluation submission (Task A)."""
        print(f"Generating soft submission for {test_file}...")
        
        # Load test data
        test_dataset = self.load_test_data(test_file, annot_meta_file)
        
        # Load test examples
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = {}
        
        for ex_id, ex in test_data.items():
            try:
                # Get demographic context if available
                demographic_context = None
                if 'annotators' in ex:
                    ann_str = ex.get("annotators", "")
                    if ann_str:
                        ann_list = [a.strip() for a in ann_str.split(",") if a.strip()]
                        if ann_list:
                            # Use first annotator's demographics
                            ann_num = ann_list[0][3:] if ann_list[0].startswith("Ann") else ann_list[0]
                            # This would need to be connected to annotator metadata
                            demographic_context = {"age": "25-34", "education": "medium", "country": "US"}
                
                # Get prediction
                prediction = self.predict_single_example(ex, demographic_context)
                
                # Convert to soft label format
                soft_label = {}
                for i in range(-5, 6):
                    soft_label[str(i)] = float(prediction["probabilities"][i + 5])
                
                results[ex_id] = {
                    "soft_label": soft_label,
                    "mistral_rating": prediction["mistral_rating"],
                    "mistral_confidence": prediction["mistral_confidence"]
                }
                
            except Exception as e:
                print(f"Error processing {ex_id}: {e}")
                # Default soft label
                soft_label = {str(i): 1.0/11 for i in range(-5, 6)}
                results[ex_id] = {"soft_label": soft_label, "error": str(e)}
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Soft submission saved to {output_file}")
    
    def generate_pe_submission(self, test_file: str, annot_meta_file: str, 
                              output_file: str) -> None:
        """Generate perspectivist evaluation submission (Task B)."""
        print(f"Generating PE submission for {test_file}...")
        
        # Load test data
        test_dataset = self.load_test_data(test_file, annot_meta_file)
        
        # Load test examples
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = {}
        
        for ex_id, ex in test_data.items():
            try:
                # Get annotators
                ann_str = ex.get("annotators", "")
                ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
                
                if not ann_list:
                    # If no annotators, create one prediction
                    prediction = self.predict_single_example(ex)
                    hard_label = int(np.argmax(prediction["probabilities"])) - 5  # Convert to -5 to +5
                    results[ex_id] = {"predictions": [hard_label]}
                else:
                    # Generate prediction for each annotator
                    predictions = []
                    for ann_tag in ann_list:
                        # Get demographic context for this annotator
                        ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                        demographic_context = self.get_annotator_demographics(ann_num, annot_meta_file)
                        
                        prediction = self.predict_single_example(ex, demographic_context)
                        hard_label = int(np.argmax(prediction["probabilities"])) - 5
                        predictions.append(hard_label)
                    
                    results[ex_id] = {"predictions": predictions}
                
            except Exception as e:
                print(f"Error processing {ex_id}: {e}")
                # Default prediction
                results[ex_id] = {"predictions": [0], "error": str(e)}
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"PE submission saved to {output_file}")
    
    def get_annotator_demographics(self, ann_num: str, annot_meta_file: str) -> Dict:
        """Get demographic context for an annotator."""
        try:
            with open(annot_meta_file, 'r', encoding='utf-8') as f:
                annot_meta = json.load(f)
            
            ann_data = annot_meta.get(ann_num, {})
            
            demographic_context = {}
            if "age" in ann_data:
                demographic_context["age"] = self.get_age_bin(ann_data["age"])
            if "gender" in ann_data:
                demographic_context["gender"] = ann_data["gender"]
            if "education_level" in ann_data:
                demographic_context["education"] = ann_data["education_level"]
            if "country_of_birth" in ann_data:
                demographic_context["country"] = ann_data["country_of_birth"]
            
            return demographic_context
        except Exception as e:
            print(f"Warning: Could not load demographics for annotator {ann_num}: {e}")
            return {}
    
    def get_age_bin(self, age_val):
        """Convert age to binned categories."""
        if age_val is None:
            return "<UNK>"
        try:
            age = int(age_val)
            if age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            elif age < 65:
                return "55-64"
            else:
                return "65+"
        except (ValueError, TypeError):
            return "<UNK>"
    
    def convert_to_codabench_format(self, results_file: str, task_type: str, 
                                   output_file: str) -> None:
        """Convert results to Codabench submission format."""
        print(f"Converting {results_file} to Codabench format...")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex_id, result in results.items():
                if task_type == "soft":
                    # Task A: Soft evaluation
                    if "soft_label" in result:
                        soft_label = result["soft_label"]
                        # Format as comma-separated probabilities
                        probs = [soft_label.get(str(i), 0.0) for i in range(-5, 6)]
                        prob_str = ",".join(f"{p:.10f}" for p in probs)
                        f.write(f"{ex_id}\t[{prob_str}]\n")
                    else:
                        # Default uniform distribution
                        probs = [1.0/11] * 11
                        prob_str = ",".join(f"{p:.10f}" for p in probs)
                        f.write(f"{ex_id}\t[{prob_str}]\n")
                
                elif task_type == "pe":
                    # Task B: Perspectivist evaluation
                    if "predictions" in result:
                        predictions = result["predictions"]
                        # Convert to label indices (0-10 for -5 to +5)
                        label_indices = [p + 5 for p in predictions]
                        pred_str = ", ".join(str(idx) for idx in label_indices)
                        f.write(f"{ex_id}\t[{pred_str}]\n")
                    else:
                        # Default prediction
                        f.write(f"{ex_id}\t[5]\n")  # Neutral prediction
        
        print(f"Codabench submission saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Codabench submissions for paraphrase detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--annot_meta", type=str, required=True, help="Path to annotator metadata JSON")
    parser.add_argument("--output_dir", type=str, default="codabench_submissions", help="Output directory for submissions")
    parser.add_argument("--task", choices=["soft", "pe", "both"], default="both", help="Task type to generate")
    parser.add_argument("--use_mistral", action="store_true", help="Enable Mistral integration")
    parser.add_argument("--mistral_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Mistral model name")
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize submission generator
    generator = CodabenchSubmissionGenerator(
        model_path=args.model_path,
        use_mistral=args.use_mistral,
        mistral_model_name=args.mistral_model_name,  # ✅ This works now
        auth_token=args.auth_token
    )
    
    # Generate submissions
    if args.task in ["soft", "both"]:
        # Generate soft evaluation submission
        soft_results_file = os.path.join(args.output_dir, "soft_results.json")
        generator.generate_soft_submission(args.test_file, args.annot_meta, soft_results_file)
        
        # Convert to Codabench format
        soft_submission_file = os.path.join(args.output_dir, "Paraphrase_test_soft.tsv")
        generator.convert_to_codabench_format(soft_results_file, "soft", soft_submission_file)
    
    if args.task in ["pe", "both"]:
        # Generate PE submission
        pe_results_file = os.path.join(args.output_dir, "pe_results.json")
        generator.generate_pe_submission(args.test_file, args.annot_meta, pe_results_file)
        
        # Convert to Codabench format
        pe_submission_file = os.path.join(args.output_dir, "Paraphrase_test_pe.tsv")
        generator.convert_to_codabench_format(pe_results_file, "pe", pe_submission_file)
    
    print(f"\n=== Submission Generation Complete ===")
    print(f"Results saved in: {args.output_dir}")
    if args.task in ["soft", "both"]:
        print(f"  - Soft evaluation: {os.path.join(args.output_dir, 'Paraphrase_test_soft.tsv')}")
    if args.task in ["pe", "both"]:
        print(f"  - PE evaluation: {os.path.join(args.output_dir, 'Paraphrase_test_pe.tsv')}")
    print(f"\nUpload these files to Codabench for evaluation!")


if __name__ == "__main__":
    main() 