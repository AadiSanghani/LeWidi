#!/usr/bin/env python3
"""
Enhanced Paraphrase Detection Training with Mistral-7B-Instruct-v0.2 Integration
This script combines transformer-based models with LLM prompting for maximum efficiency.
"""

import argparse
import json
import os
import sys
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


class MistralPromptingModule:
    """Mistral-7B-Instruct-v0.2 prompting module for paraphrase detection."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="auto", use_auth_token=None):
        """Initialize Mistral model for prompting."""
        print(f"Loading Mistral model: {model_name}")
        
        # Alternative models if Mistral is not accessible
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
        print(f"Mistral model {self.model_name} loaded successfully!")

    def generate_paraphrase_rating(self, question1: str, question2: str, 
                                  demographic_context: Optional[Dict] = None,
                                  few_shot_examples: Optional[List] = None) -> int:
        """Generate paraphrase rating using Mistral prompting."""
        
        # Build few-shot examples if provided
        few_shot_text = ""
        if few_shot_examples:
            for ex in few_shot_examples:
                few_shot_text += f"""
Question 1: {ex['question1']}
Question 2: {ex['question2']}
Rating: {ex['rating']}

"""
        
        # Build demographic context if provided
        demographic_text = ""
        if demographic_context:
            demo_parts = []
            if 'age' in demographic_context:
                demo_parts.append(f"Age: {demographic_context['age']}")
            if 'gender' in demographic_context:
                demo_parts.append(f"Gender: {demographic_context['gender']}")
            if 'education' in demographic_context:
                demo_parts.append(f"Education: {demographic_context['education']}")
            if 'country' in demographic_context:
                demo_parts.append(f"Country: {demographic_context['country']}")
            
            if demo_parts:
                demographic_text = f"\nAnnotator Context: {', '.join(demo_parts)}"
        
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
Question 2: {question2}{demographic_text}

Rate the similarity as a single integer from -5 to +5."""

        response = self.generate_response(prompt, max_new_tokens=10)
        
        # Extract rating from response
        try:
            numbers = re.findall(r'-?\d+', response)
            if numbers:
                rating = int(numbers[0])
                rating = max(-5, min(5, rating))  # Clamp to valid range
                return rating
        except:
            pass
        
        return 0  # Default to neutral if no valid rating found

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


class EnhancedParDemogModel(torch.nn.Module):
    """Enhanced RoBERTa-Large model with demographic embeddings, SBERT embeddings, and Mistral prompting."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 8, sbert_dim: int = 384, 
                 dropout_rate: float = 0.3, num_classes: int = 11, use_mistral: bool = True,
                 mistral_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        super().__init__()
        from transformers import AutoModel
        from sentence_transformers import SentenceTransformer

        # RoBERTa-Large as the main model
        if "roberta-large" not in base_name.lower():
            print(f"Warning: Expected roberta-large but got {base_name}. Using roberta-large as base model.")
            base_name = "roberta-large"
        
        self.roberta_model = AutoModel.from_pretrained(base_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # SBERT model for additional embeddings
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        # Mistral prompting module
        self.use_mistral = use_mistral
        if use_mistral:
            try:
                self.mistral_module = MistralPromptingModule(mistral_model_name)
                print("Mistral prompting module initialized successfully!")
            except Exception as e:
                print(f"Warning: Failed to initialize Mistral module: {e}")
                self.use_mistral = False
        
        self.num_classes = num_classes
        
        # Create demographic embeddings
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        # Calculate combined dimension
        roberta_dim = self.roberta_model.config.hidden_size
        total_dem_dim = len(vocab_sizes) * dem_dim
        self.combined_dim = roberta_dim + sbert_dim + total_dem_dim
        
        # Add Mistral confidence dimension if using Mistral
        if self.use_mistral:
            self.combined_dim += 1  # Add 1 dimension for Mistral confidence
        
        print(f"Enhanced model dimensions:")
        print(f"  - RoBERTa-Large: {roberta_dim}")
        print(f"  - SBERT: {sbert_dim}")
        print(f"  - Demographics ({len(vocab_sizes)} fields): {total_dem_dim}")
        if self.use_mistral:
            print(f"  - Mistral confidence: 1")
        print(f"  - Total combined: {self.combined_dim}")

        # Classification head with dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.combined_dim, num_classes)

    def forward(self, input_ids, attention_mask, texts, **demographic_inputs):
        # Get RoBERTa-Large embeddings
        roberta_outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        roberta_pooled = roberta_outputs.pooler_output
        
        # Get SBERT embeddings
        sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, device=roberta_pooled.device)
        
        # Get demographic embeddings
        demographic_embeds = []
        for field, ids in demographic_inputs.items():
            if field.endswith("_ids"):
                field_name = field[:-4]
                if field_name in self.demographic_embeddings:
                    embed = self.demographic_embeddings[field_name](ids)
                    demographic_embeds.append(embed)
        
        # Combine embeddings
        combined_embeddings = [roberta_pooled, sbert_embeddings]
        combined_embeddings.extend(demographic_embeds)
        
        # Add Mistral confidence if available
        if self.use_mistral and hasattr(self, 'mistral_confidence'):
            mistral_conf = self.mistral_confidence.unsqueeze(1)
            combined_embeddings.append(mistral_conf)
        
        # Concatenate all embeddings
        combined = torch.cat(combined_embeddings, dim=1)
        
        # Apply dropout and classification
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        return logits

    def get_mistral_predictions(self, texts: List[str], demographic_contexts: Optional[List[Dict]] = None,
                               few_shot_examples: Optional[List] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Mistral predictions and confidence scores."""
        if not self.use_mistral:
            return None, None
        
        predictions = []
        confidence_scores = []
        
        for i, text in enumerate(texts):
            # Extract questions from text
            if "[SEP]" in text:
                question1, question2 = text.split("[SEP]", 1)
                question1 = question1.strip()
                question2 = question2.strip()
            else:
                question1 = text
                question2 = ""
            
            # Get demographic context
            demo_context = demographic_contexts[i] if demographic_contexts and i < len(demographic_contexts) else None
            
            # Get Mistral prediction
            rating = self.mistral_module.generate_paraphrase_rating(
                question1, question2, demo_context, few_shot_examples
            )
            
            # Convert rating to class index (-5 to 5 -> 0 to 10)
            class_idx = rating + 5
            
            # Create one-hot encoding
            one_hot = torch.zeros(self.num_classes)
            one_hot[class_idx] = 1.0
            
            predictions.append(one_hot)
            
            # Simple confidence based on rating extremity
            confidence = min(1.0, abs(rating) / 5.0 + 0.5)
            confidence_scores.append(confidence)
        
        if predictions:
            predictions_tensor = torch.stack(predictions)
            confidence_tensor = torch.tensor(confidence_scores, dtype=torch.float32)
            return predictions_tensor, confidence_tensor
        
        return None, None


class EnhancedParDataset(Dataset):
    """Enhanced PyTorch dataset with demographic information and Mistral integration."""
    
    def __init__(self, path: str, annot_meta_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.dists = []
        self.labels = []
        self.demographic_ids = defaultdict(list)
        self.demographic_contexts = []  # Store full demographic context
        self.UNK_IDX = 0
        
        # Define active demographic fields
        self.active_field_keys = {
            "age": "age",
            "gender": "gender", 
            "education": "education_level",
            "english": "first_language_english",
            "country": "country_of_birth"
        }
        
        # Load annotator metadata
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            annot_meta = json.load(f)
        
        # Build vocabularies
        self.vocab = {}
        for field in self.active_field_keys:
            self.vocab[field] = {"<UNK>": self.UNK_IDX}
            
        # First pass: build vocabularies
        for ann_id, ann_data in annot_meta.items():
            for field, json_key in self.active_field_keys.items():
                if field == "age":
                    age_bin = self.get_age_bin(ann_data.get(json_key))
                    if age_bin not in self.vocab[field]:
                        self.vocab[field][age_bin] = len(self.vocab[field])
                else:
                    val = str(ann_data.get(json_key, "")).strip()
                    if val == "":
                        val = "<UNK>"
                    if val not in self.vocab[field]:
                        self.vocab[field][val] = len(self.vocab[field])

        self.vocab_sizes = {field: len(v) for field, v in self.vocab.items()}
        
        # Load data
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            question1 = ex["text"].get("Question1", "")
            question2 = ex["text"].get("Question2", "")
            full_text = f"{question1} [SEP] {question2}".strip()

            soft_label = ex.get("soft_label", {})
            if not soft_label or soft_label == "":
                continue
            
            # Convert to 11-class distribution
            soft_label_list = [float(soft_label.get(str(i), 0.0)) for i in range(-5, 6)]
            dist = np.array(soft_label_list, dtype=np.float32)
            if dist.sum() == 0:
                continue
            dist /= dist.sum()
            hard_label = int(np.argmax(dist))

            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []

            # Create examples for each annotator
            for ann_tag in ann_list:
                ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                meta = annot_meta.get(ann_num, {})
                
                # Get demographic info
                annotator_demog_ids = {}
                demographic_context = {}
                
                for field, json_key in self.active_field_keys.items():
                    if field == "age":
                        age_bin = self.get_age_bin(meta.get(json_key))
                        idx = self.vocab[field].get(age_bin, self.UNK_IDX)
                        demographic_context[field] = age_bin
                    else:
                        val = str(meta.get(json_key, "")).strip()
                        if val == "":
                            val = "<UNK>"
                        idx = self.vocab[field].get(val, self.UNK_IDX)
                        demographic_context[field] = val
                    annotator_demog_ids[field] = idx

                # Add this annotator's example
                self.texts.append(full_text)
                self.dists.append(dist)
                self.labels.append(hard_label)
                
                for field in self.active_field_keys:
                    self.demographic_ids[field].append(annotator_demog_ids[field])
                
                self.demographic_contexts.append(demographic_context)

            # If no annotators, create one example with UNK demographic info
            if not ann_list:
                self.texts.append(full_text)
                self.dists.append(dist)
                self.labels.append(hard_label)
                
                for field in self.active_field_keys:
                    self.demographic_ids[field].append(self.UNK_IDX)
                
                self.demographic_contexts.append({field: "<UNK>" for field in self.active_field_keys})
                    
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            "texts": self.texts[idx],
            "demographic_context": self.demographic_contexts[idx],
        }
        
        # Add demographic information
        for field in self.active_field_keys:
            item[f"{field}_ids"] = torch.tensor(self.demographic_ids[field][idx], dtype=torch.long)
        
        return item


def enhanced_collate_fn(batch):
    """Enhanced collate function that handles demographic contexts."""
    input_ids = [x["input_ids"] for x in batch]
    attn = [x["attention_mask"] for x in batch]
    labels = torch.stack([x["labels"] for x in batch])
    dists = torch.stack([x["dist"] for x in batch])
    texts = [x["texts"] for x in batch]
    demographic_contexts = [x["demographic_context"] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    result = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
        "texts": texts,
        "demographic_contexts": demographic_contexts,
    }
    
    # Add demographic fields
    demographic_fields = [k for k in batch[0].keys() if k.endswith("_ids") and k not in ["input_ids"]]
    for field in demographic_fields:
        result[field] = torch.stack([x[field] for x in batch])
    
    return result


def enhanced_evaluate(model, dataloader, device, use_mistral=True):
    """Enhanced evaluation with Mistral integration."""
    model.eval()
    total_dist = 0.0
    n_examples = 0
    all_predictions = []
    all_targets = []
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get Mistral predictions if enabled
            mistral_preds = None
            mistral_conf = None
            if use_mistral and hasattr(model, 'mistral_module'):
                mistral_preds, mistral_conf = model.get_mistral_predictions(
                    batch["texts"], 
                    batch.get("demographic_contexts", None)
                )
                if mistral_conf is not None:
                    model.mistral_confidence = mistral_conf.to(device)
            
            # Prepare demographic inputs
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )
            
            # Combine with Mistral predictions if available
            if mistral_preds is not None:
                # Weighted combination of transformer and Mistral predictions
                transformer_probs = torch.softmax(logits, dim=-1)
                mistral_probs = mistral_preds.to(device)
                
                # Use confidence-weighted combination
                alpha = 0.7  # Weight for transformer predictions
                combined_probs = alpha * transformer_probs + (1 - alpha) * mistral_probs
                logits = torch.log(combined_probs + 1e-8)
            
            # Get embeddings for PCA analysis
            with torch.no_grad():
                roberta_outputs = model.roberta_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                roberta_pooled = roberta_outputs.pooler_output
                
                sbert_embeddings = model.sbert_model.encode(batch["texts"], convert_to_tensor=True, device=roberta_pooled.device)
                
                demographic_embeds = []
                for field, ids in demographic_inputs.items():
                    if field.endswith("_ids"):
                        field_name = field[:-4]
                        if field_name in model.demographic_embeddings:
                            embed = model.demographic_embeddings[field_name](ids)
                            demographic_embeds.append(embed)
                
                combined_embeddings = [roberta_pooled, sbert_embeddings]
                combined_embeddings.extend(demographic_embeds)
                
                if hasattr(model, 'mistral_confidence'):
                    combined_embeddings.append(model.mistral_confidence.unsqueeze(1))
                
                combined = torch.cat(combined_embeddings, dim=1)
                
                all_embeddings.extend(combined.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
            
            p_hat = torch.softmax(logits, dim=-1)
            dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
            total_dist += dist.sum().item()
            n_examples += dist.numel()
            
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    return total_dist / n_examples if n_examples else 0.0, all_predictions, all_targets, all_embeddings, all_labels


def enhanced_train(args):
    """Enhanced training function with Mistral integration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    train_ds = EnhancedParDataset(args.train_file, args.annot_meta, tokenizer, args.max_length)
    val_ds = EnhancedParDataset(args.val_file, args.annot_meta, tokenizer, args.max_length) if args.val_file else None

    print(f"Training samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")
    print(f"Using demographic fields: {list(train_ds.active_field_keys.keys())}")
    print(f"Demographic vocabulary sizes: {train_ds.vocab_sizes}")

    # Create data loaders
    sampler = build_sampler(train_ds.labels) if args.balance else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=enhanced_collate_fn,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=enhanced_collate_fn) 
        if val_ds else None
    )

    # Create enhanced model
    model = EnhancedParDemogModel(
        base_name=args.model_name,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
        sbert_dim=args.sbert_dim,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
        use_mistral=args.use_mistral,
        mistral_model_name=args.mistral_model_name,
    )
    model.to(device)

    # Optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = AdamW(grouped_params, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
                   optimiser,
                   num_warmup_steps=warmup_steps,
                   num_training_steps=total_steps)

    # Training loop
    best_metric = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    train_loss_history = []
    val_dist_history = []
    lr_history = []

    print(f"Enhanced training with Mistral integration: {args.use_mistral}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get Mistral predictions if enabled
            if args.use_mistral and hasattr(model, 'mistral_module'):
                mistral_preds, mistral_conf = model.get_mistral_predictions(
                    batch["texts"], 
                    batch.get("demographic_contexts", None)
                )
                if mistral_conf is not None:
                    model.mistral_confidence = mistral_conf.to(device)
            
            # Prepare demographic inputs
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )
            
            # Combine with Mistral predictions if available
            if args.use_mistral and hasattr(model, 'mistral_module') and 'mistral_preds' in locals():
                if mistral_preds is not None:
                    transformer_probs = torch.softmax(logits, dim=-1)
                    mistral_probs = mistral_preds.to(device)
                    
                    # Weighted combination
                    alpha = 0.7
                    combined_probs = alpha * transformer_probs + (1 - alpha) * mistral_probs
                    logits = torch.log(combined_probs + 1e-8)

            # Loss calculation
            loss = F.cross_entropy(logits, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            epoch_loss += loss.item()
            step_count += 1
            
            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step_count
                tqdm.write(f"Epoch {epoch} step {step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

        # Validation
        if val_loader:
            val_dist, predictions, targets, embeddings, labels = enhanced_evaluate(
                model, val_loader, device, args.use_mistral
            )
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            
            # Generate PCA plot
            generate_pca_plot(
                embeddings, labels, args.output_dir, 
                model_name="enhanced_paraphrase_model", epoch=epoch
            )
            
            if val_dist < best_metric:
                best_metric = val_dist
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best model
                save_path = os.path.join(args.output_dir, "best_enhanced_model")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                
                # Save metadata
                metadata = {
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "use_mistral": args.use_mistral,
                    "mistral_model_name": args.mistral_model_name,
                    "training_config": {
                        "lr": args.lr,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "model_name": args.model_name,
                        "patience": args.patience,
                        "dem_dim": args.dem_dim,
                        "sbert_dim": args.sbert_dim,
                        "dropout_rate": args.dropout_rate
                    },
                    "vocab_sizes": train_ds.vocab_sizes,
                    "active_fields": list(train_ds.active_field_keys.keys())
                }
                
                with open(os.path.join(save_path, "training_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"Saved best enhanced model to {save_path}")
                
            else:
                epochs_no_improve += 1
                
            val_dist_history.append(val_dist)
            
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break
        
        train_loss_history.append(epoch_loss / step_count)
        lr_history.append(scheduler.get_last_lr()[0])

    # Save final model
    final_path = os.path.join(args.output_dir, "final_enhanced_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))

    print(f"Enhanced training completed!")
    print(f"Best validation distance: {best_metric:.4f} at epoch {best_epoch}")
    print(f"Mistral integration: {'Enabled' if args.use_mistral else 'Disabled'}")


def build_sampler(labels):
    """Returns a WeightedRandomSampler to alleviate class imbalance."""
    counts = Counter(labels)
    total = float(sum(counts.values()))
    num_classes = len(counts)
    class_weights = {c: total / (num_classes * cnt) for c, cnt in counts.items()}
    weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def generate_pca_plot(embeddings, labels, output_dir, model_name="model", epoch=None, save_data=True):
    """Generate PCA plot for enhanced model."""
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Generating PCA plot for {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
    
    # Perform PCA
    pca = PCA(n_components=min(50, embeddings.shape[1], len(embeddings)))
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Print variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"PCA Variance Analysis:")
    print(f"  PC1 explains {variance_explained[0]:.3f} of variance")
    print(f"  PC2 explains {variance_explained[1]:.3f} of variance")
    print(f"  First 2 components explain {cumulative_variance[1]:.3f} of total variance")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: PCA scatter plot
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axes[0, 0].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
    
    axes[0, 0].set_xlabel(f'PC1 ({variance_explained[0]:.3f} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({variance_explained[1]:.3f} variance)')
    axes[0, 0].set_title(f'Enhanced PCA Visualization - {model_name}' + (f' (Epoch {epoch})' if epoch else ''))
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Variance explained
    components_to_plot = min(20, len(variance_explained))
    axes[0, 1].bar(range(1, components_to_plot + 1), variance_explained[:components_to_plot])
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Variance Explained')
    axes[0, 1].set_title('Variance Explained by Each PC')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative variance
    axes[1, 0].plot(range(1, components_to_plot + 1), cumulative_variance[:components_to_plot], 'b-o')
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
    axes[1, 0].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Cumulative Variance Explained')
    axes[1, 0].set_title('Cumulative Variance Explained')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[1, 1].bar(unique_labels, counts)
    axes[1, 1].set_xlabel('Class Label')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    epoch_suffix = f"_epoch_{epoch}" if epoch else ""
    plot_filename = f"enhanced_pca_plot_{model_name}{epoch_suffix}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhanced PCA plot → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced paraphrase detection training with Mistral-7B-Instruct-v0.2 integration")
    parser.add_argument("--train_file", type=str, default="dataset/Paraphrase/Paraphrase_train.json", help="Path to Paraphrase_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/Paraphrase/Paraphrase_dev.json", help="Path to Paraphrase_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="RoBERTa-Large model name")
    parser.add_argument("--output_dir", type=str, default="runs/enhanced_outputs_par")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes (Likert scale -5 to 5)")
    parser.add_argument("--use_mistral", action="store_true", help="Enable Mistral-7B-Instruct-v0.2 integration")
    parser.add_argument("--mistral_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Mistral model name")
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token for gated models")

    args = parser.parse_args()
    enhanced_train(args) 