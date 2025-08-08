#!/usr/bin/env python3
"""
Optimized Mistral-7B-Instruct-v0.2 Prompting for Paraphrase Detection
This script implements advanced prompt engineering techniques to maximize efficiency.
"""

import json
import torch
import numpy as np
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os


class OptimizedMistralPrompter:
    """Optimized Mistral-7B-Instruct-v0.2 prompting with advanced techniques."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="auto", use_auth_token=None):
        """Initialize optimized Mistral model."""
        print(f"Loading optimized Mistral model: {model_name}")
        
        # Alternative models for fallback
        alternative_models = [
            "microsoft/DialoGPT-large",
            "facebook/opt-6.7b",
            "EleutherAI/gpt-j-6b",
            "bigscience/bloomz-7b1"
        ]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_auth_token=use_auth_token,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    use_auth_token=use_auth_token
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_auth_token=use_auth_token
                ).to(device)
            
            self.model_name = model_name
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print("Trying alternative models...")
            
            model_loaded = False
            for alt_model in alternative_models:
                try:
                    print(f"Trying {alt_model}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(alt_model)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    if device == "auto":
                        self.model = AutoModelForCausalLM.from_pretrained(
                            alt_model,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            alt_model,
                            torch_dtype=torch.float16
                        ).to(device)
                    
                    self.model_name = alt_model
                    model_loaded = True
                    print(f"Successfully loaded {alt_model}")
                    break
                    
                except Exception as alt_e:
                    print(f"Failed to load {alt_model}: {alt_e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("Failed to load any suitable model")
        
        self.model.eval()
        print(f"Optimized Mistral model {self.model_name} loaded successfully!")
        
        # Initialize prompt templates
        self.initialize_prompt_templates()

    def initialize_prompt_templates(self):
        """Initialize optimized prompt templates."""
        
        # Base system prompt
        self.system_prompt = """You are an expert semantic similarity evaluator with deep understanding of language nuances, cultural contexts, and demographic variations. Your task is to assess the semantic similarity between two questions on a scale from -5 to +5.

CRITICAL EVALUATION CRITERIA:
-5: Completely different topics, no semantic overlap
-4: Very different topics with minimal conceptual overlap
-3: Different topics with some shared concepts or domain
-2: Related topics but different focus, intent, or specificity
-1: Similar topics with different specific details or approaches
0: Somewhat similar but distinct questions with different emphasis
+1: Similar questions with minor differences in wording or scope
+2: Very similar questions, same intent, slight wording variations
+3: Nearly identical questions with minimal variation
+4: Almost exactly the same question, minor differences
+5: Identical questions (perfect paraphrases)

EVALUATION GUIDELINES:
- Consider semantic meaning, not just surface similarity
- Account for cultural and demographic context
- Focus on intent and purpose of questions
- Ignore minor grammatical differences
- Consider domain-specific terminology
- Evaluate logical equivalence, not just lexical similarity"""

        # Demographic-aware prompts
        self.demographic_prompts = {
            "age": {
                "young": "Consider that younger annotators may focus more on modern terminology and contemporary context.",
                "middle": "Consider that middle-aged annotators may balance traditional and modern perspectives.",
                "senior": "Consider that older annotators may emphasize traditional terminology and established concepts."
            },
            "education": {
                "high": "Consider that highly educated annotators may focus on precise terminology and academic rigor.",
                "medium": "Consider that moderately educated annotators may balance accessibility and accuracy.",
                "low": "Consider that less educated annotators may prioritize clarity and everyday language."
            },
            "country": {
                "US": "Consider American English conventions and cultural context.",
                "UK": "Consider British English conventions and cultural context.",
                "other": "Consider international English variations and cultural diversity."
            }
        }

    def generate_optimized_prompt(self, question1: str, question2: str, 
                                 demographic_context: Optional[Dict] = None,
                                 few_shot_examples: Optional[List] = None,
                                 prompt_style: str = "comprehensive") -> str:
        """Generate optimized prompt based on context and style."""
        
        if prompt_style == "comprehensive":
            return self._comprehensive_prompt(question1, question2, demographic_context, few_shot_examples)
        elif prompt_style == "concise":
            return self._concise_prompt(question1, question2, demographic_context, few_shot_examples)
        elif prompt_style == "chain_of_thought":
            return self._chain_of_thought_prompt(question1, question2, demographic_context, few_shot_examples)
        elif prompt_style == "demographic_aware":
            return self._demographic_aware_prompt(question1, question2, demographic_context, few_shot_examples)
        else:
            return self._comprehensive_prompt(question1, question2, demographic_context, few_shot_examples)

    def _comprehensive_prompt(self, question1: str, question2: str, 
                            demographic_context: Optional[Dict] = None,
                            few_shot_examples: Optional[List] = None) -> str:
        """Generate comprehensive prompt with detailed instructions."""
        
        # Build few-shot examples
        few_shot_text = ""
        if few_shot_examples:
            few_shot_text = "\nEXAMPLES:\n"
            for i, ex in enumerate(few_shot_examples, 1):
                few_shot_text += f"""
Example {i}:
Q1: {ex['question1']}
Q2: {ex['question2']}
Analysis: {ex.get('analysis', '')}
Rating: {ex['rating']} ({self._rating_explanation(ex['rating'])})
"""
        
        # Build demographic context
        demographic_text = ""
        if demographic_context:
            demo_parts = []
            for field, value in demographic_context.items():
                if field in self.demographic_prompts:
                    if field == "age":
                        if "18-24" in str(value) or "25-34" in str(value):
                            demo_parts.append(self.demographic_prompts[field]["young"])
                        elif "35-44" in str(value) or "45-54" in str(value):
                            demo_parts.append(self.demographic_prompts[field]["middle"])
                        elif "55-64" in str(value) or "65+" in str(value):
                            demo_parts.append(self.demographic_prompts[field]["senior"])
                    elif field == "education":
                        if "high" in str(value).lower() or "university" in str(value).lower():
                            demo_parts.append(self.demographic_prompts[field]["high"])
                        elif "medium" in str(value).lower() or "college" in str(value).lower():
                            demo_parts.append(self.demographic_prompts[field]["medium"])
                        else:
                            demo_parts.append(self.demographic_prompts[field]["low"])
                    elif field == "country":
                        if "united states" in str(value).lower() or "us" in str(value).lower():
                            demo_parts.append(self.demographic_prompts[field]["US"])
                        elif "united kingdom" in str(value).lower() or "uk" in str(value).lower():
                            demo_parts.append(self.demographic_prompts[field]["UK"])
                        else:
                            demo_parts.append(self.demographic_prompts[field]["other"])
            
            if demo_parts:
                demographic_text = f"\nCONTEXTUAL CONSIDERATIONS:\n" + "\n".join(demo_parts)
        
        prompt = f"""{self.system_prompt}{demographic_text}{few_shot_text}

CURRENT EVALUATION:
Question 1: {question1}
Question 2: {question2}

Please analyze the semantic similarity between these questions, considering:
1. Core meaning and intent
2. Topic overlap and domain relevance
3. Specificity and detail level
4. Cultural and contextual factors

Provide your rating as a single integer from -5 to +5."""

        return prompt

    def _concise_prompt(self, question1: str, question2: str, 
                       demographic_context: Optional[Dict] = None,
                       few_shot_examples: Optional[List] = None) -> str:
        """Generate concise prompt for faster inference."""
        
        few_shot_text = ""
        if few_shot_examples:
            for ex in few_shot_examples[:2]:  # Limit to 2 examples for conciseness
                few_shot_text += f"\nQ1: {ex['question1']}\nQ2: {ex['question2']}\nRating: {ex['rating']}\n"
        
        prompt = f"""Rate semantic similarity between questions (-5 to +5):

{few_shot_text}Q1: {question1}
Q2: {question2}

Rating:"""

        return prompt

    def _chain_of_thought_prompt(self, question1: str, question2: str, 
                                demographic_context: Optional[Dict] = None,
                                few_shot_examples: Optional[List] = None) -> str:
        """Generate chain-of-thought prompt for detailed reasoning."""
        
        few_shot_text = ""
        if few_shot_examples:
            few_shot_text = "\nEXAMPLES:\n"
            for ex in few_shot_examples:
                few_shot_text += f"""
Q1: {ex['question1']}
Q2: {ex['question2']}
Reasoning: {ex.get('reasoning', 'Analyzing semantic overlap and intent...')}
Rating: {ex['rating']}
"""
        
        prompt = f"""Analyze the semantic similarity between two questions step by step:

{few_shot_text}Q1: {question1}
Q2: {question2}

Let me think through this step by step:
1. What is the core topic of each question?
2. How similar are the topics?
3. What is the intent of each question?
4. How similar are the intents?
5. Are there any cultural or contextual factors to consider?
6. What is my final rating (-5 to +5)?

Rating:"""

        return prompt

    def _demographic_aware_prompt(self, question1: str, question2: str, 
                                 demographic_context: Optional[Dict] = None,
                                 few_shot_examples: Optional[List] = None) -> str:
        """Generate demographic-aware prompt."""
        
        # Build demographic-specific instructions
        demographic_instructions = ""
        if demographic_context:
            instructions = []
            for field, value in demographic_context.items():
                if field == "age":
                    if "18-24" in str(value) or "25-34" in str(value):
                        instructions.append("Consider modern terminology and contemporary context")
                    elif "55-64" in str(value) or "65+" in str(value):
                        instructions.append("Consider traditional terminology and established concepts")
                elif field == "education":
                    if "high" in str(value).lower() or "university" in str(value).lower():
                        instructions.append("Consider academic precision and formal language")
                    else:
                        instructions.append("Consider everyday language and accessibility")
                elif field == "country":
                    if "united states" in str(value).lower():
                        instructions.append("Consider American English conventions")
                    elif "united kingdom" in str(value).lower():
                        instructions.append("Consider British English conventions")
            
            if instructions:
                demographic_instructions = f"\nEVALUATION CONTEXT: {'; '.join(instructions)}"
        
        few_shot_text = ""
        if few_shot_examples:
            for ex in few_shot_examples:
                few_shot_text += f"\nQ1: {ex['question1']}\nQ2: {ex['question2']}\nRating: {ex['rating']}\n"
        
        prompt = f"""{self.system_prompt}{demographic_instructions}

{few_shot_text}Q1: {question1}
Q2: {question2}

Rating:"""

        return prompt

    def _rating_explanation(self, rating: int) -> str:
        """Provide explanation for rating."""
        explanations = {
            -5: "completely different topics",
            -4: "very different topics, minimal overlap",
            -3: "different topics, some shared concepts",
            -2: "related topics, different focus",
            -1: "similar topics, different details",
            0: "somewhat similar, different emphasis",
            1: "similar questions, minor differences",
            2: "very similar, same intent",
            3: "nearly identical, minimal variation",
            4: "almost exactly the same",
            5: "identical questions"
        }
        return explanations.get(rating, "unknown rating")

    def generate_response(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.1) -> str:
        """Generate response from the language model."""
        # Check if this is a Mistral model that supports chat template
        if "mistral" in self.model_name.lower() and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                formatted_prompt = prompt
        else:
            formatted_prompt = f"Human: {prompt}\n\nAssistant:"
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            add_special_tokens=False,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return response

    def evaluate_paraphrase_optimized(self, question1: str, question2: str, 
                                    demographic_context: Optional[Dict] = None,
                                    few_shot_examples: Optional[List] = None,
                                    prompt_style: str = "comprehensive") -> Tuple[int, float]:
        """Evaluate paraphrase detection with optimized prompting."""
        
        prompt = self.generate_optimized_prompt(
            question1, question2, demographic_context, few_shot_examples, prompt_style
        )
        
        response = self.generate_response(prompt, max_new_tokens=20)
        
        # Extract rating from response
        try:
            numbers = re.findall(r'-?\d+', response)
            if numbers:
                rating = int(numbers[0])
                rating = max(-5, min(5, rating))  # Clamp to valid range
                
                # Calculate confidence based on response clarity
                confidence = self._calculate_confidence(response, rating)
                return rating, confidence
        except:
            pass
        
        return 0, 0.5  # Default to neutral with medium confidence

    def _calculate_confidence(self, response: str, rating: int) -> float:
        """Calculate confidence score based on response characteristics."""
        confidence = 0.5  # Base confidence
        
        # Factors that increase confidence
        if f"{rating}" in response:
            confidence += 0.2
        if "rating" in response.lower() or "score" in response.lower():
            confidence += 0.1
        if len(response.strip()) > 10:  # Detailed response
            confidence += 0.1
        if abs(rating) >= 3:  # Extreme ratings tend to be more confident
            confidence += 0.1
        
        # Factors that decrease confidence
        if "?" in response:  # Uncertainty indicators
            confidence -= 0.1
        if "maybe" in response.lower() or "possibly" in response.lower():
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))


def load_optimized_few_shot_examples(data_file: str, num_examples: int = 5) -> List[Dict]:
    """Load optimized few-shot examples with reasoning."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    items = list(data.items())[:num_examples * 20]  # Get more items to filter from
    
    for ex_id, ex in items:
        if len(examples) >= num_examples:
            break
        
        # Get average rating
        annotations = ex.get("annotations", {})
        if isinstance(annotations, dict):
            ratings = [int(r) for r in annotations.values() if str(r).lstrip('-').isdigit()]
            if ratings:
                avg_rating = int(round(np.mean(ratings)))
                
                # Generate reasoning based on questions
                question1 = ex['text'].get('Question1', '')
                question2 = ex['text'].get('Question2', '')
                
                reasoning = generate_reasoning(question1, question2, avg_rating)
                
                examples.append({
                    'question1': question1,
                    'question2': question2,
                    'rating': avg_rating,
                    'reasoning': reasoning,
                    'analysis': f"Semantic similarity analysis: {reasoning}"
                })
    
    return examples


def generate_reasoning(question1: str, question2: str, rating: int) -> str:
    """Generate reasoning for the rating."""
    if rating >= 4:
        return "Questions are nearly identical with minimal variation in wording"
    elif rating >= 2:
        return "Questions are very similar with same intent but slight wording differences"
    elif rating >= 0:
        return "Questions are somewhat similar but have different emphasis or scope"
    elif rating >= -2:
        return "Questions are related but focus on different aspects or details"
    elif rating >= -4:
        return "Questions have minimal overlap but are in different domains"
    else:
        return "Questions are completely different with no semantic overlap"


def evaluate_dataset_optimized(evaluator: OptimizedMistralPrompter, data_file: str, 
                             output_file: str, few_shot_examples: Optional[List] = None,
                             prompt_style: str = "comprehensive") -> Dict:
    """Evaluate entire dataset with optimized prompting."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    print(f"Evaluating {len(data)} examples with {prompt_style} prompting...")
    for ex_id, ex in tqdm(data.items()):
        try:
            question1 = ex['text'].get('Question1', '')
            question2 = ex['text'].get('Question2', '')
            
            # Get demographic context if available
            demographic_context = None
            if 'annotators' in ex:
                # Extract demographic info from annotators
                ann_str = ex.get("annotators", "")
                if ann_str:
                    ann_list = [a.strip() for a in ann_str.split(",") if a.strip()]
                    if ann_list:
                        # Use first annotator's demographics as context
                        ann_num = ann_list[0][3:] if ann_list[0].startswith("Ann") else ann_list[0]
                        # This would need to be connected to annotator metadata
                        demographic_context = {"age": "25-34", "education": "medium", "country": "US"}
            
            rating, confidence = evaluator.evaluate_paraphrase_optimized(
                question1, question2, demographic_context, few_shot_examples, prompt_style
            )
            
            results[ex_id] = {
                'prediction': rating,
                'confidence': confidence,
                'text': ex['text'],
                'annotations': ex.get('annotations', {}),
                'soft_label': ex.get('soft_label', {}),
                'prompt_style': prompt_style
            }
            
        except Exception as e:
            print(f"Error processing {ex_id}: {e}")
            results[ex_id] = {'prediction': 0, 'confidence': 0.5, 'error': str(e)}
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results


def calculate_optimized_metrics(results: Dict) -> Dict:
    """Calculate comprehensive evaluation metrics."""
    predictions = []
    true_ratings = []
    confidences = []
    
    for ex_id, result in results.items():
        if 'error' in result:
            continue
            
        prediction = result['prediction']
        confidence = result.get('confidence', 0.5)
        annotations = result.get('annotations', {})
        
        if isinstance(annotations, dict) and annotations:
            ratings = [int(r) for r in annotations.values() if str(r).lstrip('-').isdigit()]
            if ratings:
                avg_rating = np.mean(ratings)
                predictions.append(prediction)
                true_ratings.append(avg_rating)
                confidences.append(confidence)
    
    if predictions and true_ratings:
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
        correlation = np.corrcoef(predictions, true_ratings)[0, 1]
        
        # Confidence-weighted metrics
        weighted_errors = np.array(predictions) - np.array(true_ratings)
        confidence_weights = np.array(confidences)
        weighted_mae = np.average(np.abs(weighted_errors), weights=confidence_weights)
        
        # High-confidence predictions
        high_conf_mask = np.array(confidences) > 0.7
        if np.any(high_conf_mask):
            high_conf_mae = np.mean(np.abs(weighted_errors[high_conf_mask]))
        else:
            high_conf_mae = None
        
        return {
            'mae': mae,
            'correlation': correlation,
            'weighted_mae': weighted_mae,
            'high_confidence_mae': high_conf_mae,
            'avg_confidence': np.mean(confidences),
            'num_predictions': len(predictions)
        }
    
    return {'error': 'No valid predictions'}


def main():
    parser = argparse.ArgumentParser(description="Optimized Mistral prompting for paraphrase detection")
    parser.add_argument("--data_file", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Output results file")
    parser.add_argument("--few_shot_file", type=str, help="Training file for few-shot examples")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--prompt_style", choices=["comprehensive", "concise", "chain_of_thought", "demographic_aware"], 
                       default="comprehensive", help="Prompt style to use")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                       help="Model name or path")
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token for gated models")
    
    args = parser.parse_args()
    
    # Initialize optimized evaluator
    evaluator = OptimizedMistralPrompter(
        model_name=args.model_name, 
        device=args.device,
        use_auth_token=args.auth_token
    )
    
    # Load few-shot examples if requested
    few_shot_examples = None
    if args.few_shot_file:
        print(f"Loading {args.num_shots} optimized few-shot examples...")
        few_shot_examples = load_optimized_few_shot_examples(
            args.few_shot_file, args.num_shots
        )
        print(f"Loaded {len(few_shot_examples)} examples")
    
    # Evaluate dataset
    results = evaluate_dataset_optimized(
        evaluator, args.data_file, args.output_file, 
        few_shot_examples, args.prompt_style
    )
    
    # Calculate metrics
    metrics = calculate_optimized_metrics(results)
    print(f"\nOptimized Evaluation Results ({args.prompt_style} prompting):")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")


if __name__ == "__main__":
    main() 