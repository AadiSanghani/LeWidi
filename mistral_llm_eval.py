#!/usr/bin/env python3
"""
LLM-based evaluation using Mistral-7B-Instruct-v0.2 for VariErrNLI and Paraphrase detection tasks.
This script implements zero-shot and few-shot prompting approaches.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter
import os

class MistralEvaluator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="auto"):
        """Initialize Mistral-7B-Instruct-v0.2 model."""
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate device mapping
        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
        
        self.model.eval()
        print(f"Model loaded successfully!")

    def generate_response(self, prompt, max_new_tokens=50, temperature=0.1):
        """Generate response from Mistral model."""
        # Format prompt with Mistral chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return response

    def evaluate_nli(self, context, statement, few_shot_examples=None):
        """Evaluate Natural Language Inference with VariErrNLI format."""
        
        # Build few-shot examples if provided
        few_shot_text = ""
        if few_shot_examples:
            for ex in few_shot_examples:
                few_shot_text += f"""
Context: {ex['context']}
Statement: {ex['statement']}
Answer: {ex['label']}

"""
        
        prompt = f"""You are an expert at natural language inference. Given a context and a statement, determine the relationship between them.

{few_shot_text}Context: {context}
Statement: {statement}

The relationship can be:
- "entailment": The statement is definitely true given the context
- "contradiction": The statement is definitely false given the context  
- "neutral": The statement might be true or false, there's not enough information

Answer with only one word: entailment, contradiction, or neutral."""

        response = self.generate_response(prompt, max_new_tokens=10)
        
        # Extract the label from response
        response_lower = response.lower().strip()
        if "entailment" in response_lower:
            return "entailment"
        elif "contradiction" in response_lower:
            return "contradiction"
        elif "neutral" in response_lower:
            return "neutral"
        else:
            # Default to neutral if unclear
            return "neutral"

    def evaluate_paraphrase(self, question1, question2, few_shot_examples=None):
        """Evaluate paraphrase detection with rating scale -5 to +5."""
        
        # Build few-shot examples if provided
        few_shot_text = ""
        if few_shot_examples:
            for ex in few_shot_examples:
                few_shot_text += f"""
Question 1: {ex['question1']}
Question 2: {ex['question2']}
Rating: {ex['rating']}

"""
        
        prompt = f"""You are an expert at determining semantic similarity between questions. Rate how similar two questions are on a scale from -5 to +5:

-5: Completely different topics, no similarity
-4: Very different topics with minimal overlap
-3: Different topics with some shared concepts
-2: Related topics but different focus/intent
-1: Similar topics with different specific details
0: Somewhat similar but distinct questions
+1: Similar questions with minor differences
+2: Very similar questions, same intent, slight wording differences
+3: Nearly identical questions with minimal variation
+4: Almost exactly the same question
+5: Identical questions (perfect paraphrases)

{few_shot_text}Question 1: {question1}
Question 2: {question2}

Rate the similarity as a single integer from -5 to +5."""

        response = self.generate_response(prompt, max_new_tokens=10)
        
        # Extract rating from response
        try:
            # Look for number in response
            import re
            numbers = re.findall(r'-?\d+', response)
            if numbers:
                rating = int(numbers[0])
                # Clamp to valid range
                rating = max(-5, min(5, rating))
                return rating
        except:
            pass
        
        # Default to 0 if no valid rating found
        return 0

def load_few_shot_examples(data_file, task_type, num_examples=5):
    """Load few-shot examples from training data."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    items = list(data.items())[:num_examples * 10]  # Get more items to filter from
    
    if task_type == "nli":
        for ex_id, ex in items:
            if len(examples) >= num_examples:
                break
            
            # Get majority vote label
            annotations = ex.get("annotations", {})
            if isinstance(annotations, dict):
                label_counts = Counter(annotations.values())
                majority_label = label_counts.most_common(1)[0][0] if label_counts else "neutral"
                
                examples.append({
                    'context': ex['text'].get('context', ''),
                    'statement': ex['text'].get('statement', ''),
                    'label': majority_label
                })
    
    elif task_type == "paraphrase":
        for ex_id, ex in items:
            if len(examples) >= num_examples:
                break
            
            # Get average rating
            annotations = ex.get("annotations", {})
            if isinstance(annotations, dict):
                ratings = [int(r) for r in annotations.values() if r.lstrip('-').isdigit()]
                if ratings:
                    avg_rating = int(round(np.mean(ratings)))
                    
                    examples.append({
                        'question1': ex['text'].get('Question1', ''),
                        'question2': ex['text'].get('Question2', ''),
                        'rating': avg_rating
                    })
    
    return examples

def evaluate_dataset(evaluator, data_file, task_type, output_file, few_shot_examples=None):
    """Evaluate entire dataset."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    print(f"Evaluating {len(data)} examples...")
    for ex_id, ex in tqdm(data.items()):
        try:
            if task_type == "nli":
                context = ex['text'].get('context', '')
                statement = ex['text'].get('statement', '')
                prediction = evaluator.evaluate_nli(context, statement, few_shot_examples)
                
            elif task_type == "paraphrase":
                question1 = ex['text'].get('Question1', '')
                question2 = ex['text'].get('Question2', '')
                prediction = evaluator.evaluate_paraphrase(question1, question2, few_shot_examples)
            
            results[ex_id] = {
                'prediction': prediction,
                'text': ex['text'],
                'annotations': ex.get('annotations', {}),
                'soft_label': ex.get('soft_label', {})
            }
            
        except Exception as e:
            print(f"Error processing {ex_id}: {e}")
            results[ex_id] = {'prediction': 'neutral' if task_type == 'nli' else 0, 'error': str(e)}
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

def calculate_metrics(results, task_type):
    """Calculate evaluation metrics."""
    if task_type == "nli":
        # Calculate accuracy against majority vote
        correct = 0
        total = 0
        
        for ex_id, result in results.items():
            if 'error' in result:
                continue
                
            prediction = result['prediction']
            annotations = result.get('annotations', {})
            
            if isinstance(annotations, dict) and annotations:
                # Get majority label
                label_counts = Counter(annotations.values())
                majority_label = label_counts.most_common(1)[0][0]
                
                if prediction == majority_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"NLI Accuracy: {accuracy:.3f} ({correct}/{total})")
        
    elif task_type == "paraphrase":
        # Calculate MAE and correlation with average ratings
        predictions = []
        true_ratings = []
        
        for ex_id, result in results.items():
            if 'error' in result:
                continue
                
            prediction = result['prediction']
            annotations = result.get('annotations', {})
            
            if isinstance(annotations, dict) and annotations:
                ratings = [int(r) for r in annotations.values() if str(r).lstrip('-').isdigit()]
                if ratings:
                    avg_rating = np.mean(ratings)
                    predictions.append(prediction)
                    true_ratings.append(avg_rating)
        
        if predictions and true_ratings:
            mae = np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
            correlation = np.corrcoef(predictions, true_ratings)[0, 1]
            print(f"Paraphrase MAE: {mae:.3f}")
            print(f"Paraphrase Correlation: {correlation:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate VariErrNLI and Paraphrase tasks with Mistral-7B-Instruct-v0.2")
    parser.add_argument("--data_file", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--task_type", choices=["nli", "paraphrase"], required=True, help="Task type")
    parser.add_argument("--output_file", type=str, required=True, help="Output results file")
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot prompting")
    parser.add_argument("--few_shot_file", type=str, help="Training file for few-shot examples")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MistralEvaluator(device=args.device)
    
    # Load few-shot examples if requested
    few_shot_examples = None
    if args.few_shot and args.few_shot_file:
        print(f"Loading {args.num_shots} few-shot examples...")
        few_shot_examples = load_few_shot_examples(
            args.few_shot_file, args.task_type, args.num_shots
        )
        print(f"Loaded {len(few_shot_examples)} examples")
    
    # Evaluate dataset
    results = evaluate_dataset(
        evaluator, args.data_file, args.task_type, 
        args.output_file, few_shot_examples
    )
    
    # Calculate metrics
    calculate_metrics(results, args.task_type)

if __name__ == "__main__":
    main()