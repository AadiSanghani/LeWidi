import argparse
import json
import os
import sys
import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW, RAdam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


class CrossAttentionLayer(torch.nn.Module):
    """Cross-attention between text and demographic embeddings."""
    
    def __init__(self, text_dim, dem_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert self.head_dim * num_heads == text_dim
        
        self.q_proj = torch.nn.Linear(text_dim, text_dim)
        self.k_proj = torch.nn.Linear(dem_dim, text_dim)
        self.v_proj = torch.nn.Linear(dem_dim, text_dim)
        self.out_proj = torch.nn.Linear(text_dim, text_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(text_dim)
        self.norm2 = torch.nn.LayerNorm(text_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(text_dim, text_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(text_dim * 4, text_dim)
        )
        
    def forward(self, text_emb, dem_emb):
        B, L, D = text_emb.shape
        
        q = self.q_proj(text_emb).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dem_emb).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dem_emb).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        text_emb = self.norm1(text_emb + attn_output)
        ffn_output = self.ffn(text_emb)
        text_emb = self.norm2(text_emb + ffn_output)
        
        return text_emb


class EnsembleParDemogModel(torch.nn.Module):
    """Ensemble of multiple models with different architectures."""
    
    def __init__(self, model_configs: List[Dict], vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.num_classes = num_classes
        
        for config in model_configs:
            model = self._create_single_model(config, vocab_sizes, dem_dim, dropout_rate, num_classes)
            self.models.append(model)
    
    def _create_single_model(self, config: Dict, vocab_sizes: dict, dem_dim: int, 
                           dropout_rate: float, num_classes: int):
        """Create a single model based on configuration."""
        base_name = config['base_name']
        model_type = config.get('model_type', 'cross_attention')
        
        if model_type == 'cross_attention':
            return CrossAttentionParModel(base_name, vocab_sizes, dem_dim, dropout_rate, num_classes)
        elif model_type == 'simple_concat':
            return SimpleConcatParModel(base_name, vocab_sizes, dem_dim, dropout_rate, num_classes)
        elif model_type == 'demographic_only':
            return DemographicOnlyModel(base_name, vocab_sizes, dem_dim, dropout_rate, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, *, texts, **demographic_inputs):
        """Forward pass through all models and ensemble predictions."""
        all_logits = []
        
        for model in self.models:
            logits = model(texts=texts, **demographic_inputs)
            all_logits.append(logits)
        
        # Average logits from all models
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        return ensemble_logits


class CrossAttentionParModel(torch.nn.Module):
    """Advanced model with cross-attention."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.text_model = SentenceTransformer(base_name)
        self.num_classes = num_classes
        
        # Demographic embeddings
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        emb_dim = self.text_model.get_sentence_embedding_dimension()
        if emb_dim is None:
            emb_dim = 768  # Default fallback
        emb_dim = int(emb_dim)
        
        # Project text embeddings
        self.text_proj = torch.nn.Linear(emb_dim, 768)
        
        # Cross-attention layers
        self.cross_attention_layers = torch.nn.ModuleList([
            CrossAttentionLayer(768, dem_dim * len(vocab_sizes), num_heads=12, dropout=dropout_rate)
            for _ in range(2)
        ])
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, *, texts, **demographic_inputs):
        text_embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        # Convert to regular tensor for backpropagation
        text_embeddings = text_embeddings.clone().detach().requires_grad_(True)
        text_embeddings = self.text_proj(text_embeddings)
        
        # Add sequence dimension for cross-attention (batch_size, 1, embedding_dim)
        text_embeddings = text_embeddings.unsqueeze(1)
        
        # Get demographic embeddings
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)
        
        if demographic_vectors:
            dem_emb = torch.cat(demographic_vectors, dim=-1)
            
            for cross_attn in self.cross_attention_layers:
                text_embeddings = cross_attn(text_embeddings, dem_emb)
        
        # Remove sequence dimension and pool
        text_embeddings = text_embeddings.squeeze(1)
        logits = self.classifier(text_embeddings)
        return logits


class SimpleConcatParModel(torch.nn.Module):
    """Simple concatenation model."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.text_model = SentenceTransformer(base_name)
        self.num_classes = num_classes
        
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        emb_dim = self.text_model.get_sentence_embedding_dimension()
        if emb_dim is None:
            emb_dim = 768  # Default fallback
        emb_dim = int(emb_dim)
        
        num_demog_fields = len(vocab_sizes)
        total_dim = emb_dim + num_demog_fields * dem_dim
        
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, *, texts, **demographic_inputs):
        text_embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        # Convert to regular tensor for backpropagation
        text_embeddings = text_embeddings.clone().detach().requires_grad_(True)
        
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        if demographic_vectors:
            concat = torch.cat([text_embeddings] + demographic_vectors, dim=-1)
        else:
            concat = text_embeddings
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class DemographicOnlyModel(torch.nn.Module):
    """Model that focuses heavily on demographic information."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.text_model = SentenceTransformer(base_name)
        self.num_classes = num_classes
        
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        emb_dim = self.text_model.get_sentence_embedding_dimension()
        if emb_dim is None:
            emb_dim = 768  # Default fallback
        emb_dim = int(emb_dim)
        
        # Smaller text projection, larger demographic influence
        self.text_proj = torch.nn.Linear(emb_dim, 256)
        num_demog_fields = len(vocab_sizes)
        dem_dim_total = num_demog_fields * dem_dim
        
        total_dim = 256 + dem_dim_total
        
        self.demographic_attention = torch.nn.MultiheadAttention(dem_dim_total, num_heads=8, dropout=dropout_rate)
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, *, texts, **demographic_inputs):
        text_embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        # Convert to regular tensor for backpropagation
        text_embeddings = text_embeddings.clone().detach().requires_grad_(True)
        text_embeddings = self.text_proj(text_embeddings)
        
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        if demographic_vectors:
            dem_emb = torch.cat(demographic_vectors, dim=-1)
            # Apply attention to demographic embeddings
            dem_emb = dem_emb.unsqueeze(1)  # Add sequence dimension
            dem_emb, _ = self.demographic_attention(dem_emb, dem_emb, dem_emb)
            dem_emb = dem_emb.squeeze(1)
            
            concat = torch.cat([text_embeddings, dem_emb], dim=-1)
        else:
            concat = text_embeddings
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class LLMPromptedDataset(Dataset):
    """Dataset with LLM prompting for enhanced understanding."""
    
    PAD_IDX = 0
    UNK_IDX = 1

    FIELD_KEYS = {
        "age": "Age",
        "gender": "Gender",
        "ethnicity": "Ethnicity simplified",
        "country_birth": "Country of birth",
        "country_residence": "Country of residence",
        "nationality": "Nationality",
        "student": "Student status",
        "employment": "Employment status",
    }

    PROMPT_TEMPLATES = [
        "Question 1: {q1}\nQuestion 2: {q2}\nAre these questions asking about the same thing?",
        "Q1: {q1}\nQ2: {q2}\nRate how similar these questions are:",
        "Compare these two questions:\n{q1}\n{q2}\nHow similar are they?",
        "Question A: {q1}\nQuestion B: {q2}\nAssess their similarity:",
        "Text 1: {q1}\nText 2: {q2}\nEvaluate similarity:",
    ]

    @staticmethod
    def get_age_bin(age):
        if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
            return "<UNK>"
        try:
            age = float(age)
            if age < 20: return "18-19"
            elif age < 25: return "20-24"
            elif age < 30: return "25-29"
            elif age < 35: return "30-34"
            elif age < 40: return "35-39"
            elif age < 50: return "40-49"
            else: return "50+"
        except (ValueError, TypeError):
            return "<UNK>"

    def __init__(self, path: str, annot_meta_path: str, max_length: int = 512, 
                 use_prompts: bool = True, augment: bool = True):
        self.max_length = max_length
        self.use_prompts = use_prompts
        self.augment = augment
        self.active_field_keys = self.FIELD_KEYS

        self.texts = []
        self.labels = []
        self.dists = []
        self.demographic_ids = {field: [] for field in self.active_field_keys}

        with open(annot_meta_path, "r", encoding="utf-8") as f:
            annot_meta = json.load(f)

        self.annot_meta = annot_meta

        # Build vocabulary
        self.vocab = {
            field: {"<PAD>": self.PAD_IDX, "<UNK>": self.UNK_IDX}
            for field in self.active_field_keys
        }

        for ann_data in annot_meta.values():
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
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            question1 = ex["text"].get("Question1", "")
            question2 = ex["text"].get("Question2", "")
            
            # Generate different text formats
            text_formats = []
            
            if self.use_prompts:
                # Add LLM-style prompts
                for template in self.PROMPT_TEMPLATES:
                    text_formats.append(template.format(q1=question1, q2=question2))
            
            # Add standard formats
            text_formats.extend([
                f"{question1} [SEP] {question2}",
                f"Question 1: {question1} Question 2: {question2}",
                f"Q1: {question1} Q2: {question2}",
                f"{question1} ||| {question2}",
                f"Text A: {question1} Text B: {question2}",
            ])

            soft_label = ex.get("soft_label", {})
            if not soft_label or soft_label == "":
                continue
            
            soft_label_list = [float(soft_label.get(str(i), 0.0)) for i in range(-5, 6)]
            dist = np.array(soft_label_list, dtype=np.float32)
            if dist.sum() == 0:
                continue
            dist /= dist.sum()
            hard_label = int(np.argmax(dist))

            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            if not ann_list:
                ann_list = []

            # Create examples for each annotator
            for ann_tag in ann_list:
                ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                meta = annot_meta.get(ann_num, {})
                
                annotator_demog_ids = {}
                for field, json_key in self.active_field_keys.items():
                    if field == "age":
                        age_bin = self.get_age_bin(meta.get(json_key))
                        idx = self.vocab[field].get(age_bin, self.UNK_IDX)
                    else:
                        val = str(meta.get(json_key, "")).strip()
                        if val == "":
                            val = "<UNK>"
                        idx = self.vocab[field].get(val, self.UNK_IDX)
                    annotator_demog_ids[field] = idx

                # Add examples with different text formats
                for text_format in text_formats:
                    self.texts.append(text_format)
                    self.dists.append(dist)
                    self.labels.append(hard_label)
                    
                    for field in self.active_field_keys:
                        self.demographic_ids[field].append(annotator_demog_ids[field])

            # If no annotators, create examples with UNK demographics
            if not ann_list:
                for text_format in text_formats:
                    self.texts.append(text_format)
                    self.dists.append(dist)
                    self.labels.append(hard_label)
                    
                    for field in self.active_field_keys:
                        self.demographic_ids[field].append(self.UNK_IDX)

        print(f"Dataset created with {len(self.texts)} examples (expanded from {len(data)} original examples)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        result = {
            "texts": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
        }
        
        for field in self.active_field_keys:
            result[f"{field}_ids"] = torch.tensor(self.demographic_ids[field][idx], dtype=torch.long)
        
        return result


def ensemble_collate_fn(batch):
    """Collate function for ensemble dataset."""
    texts = [b["texts"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    result = {
        "texts": texts,
        "labels": labels,
        "dist": dists,
    }
    
    demographic_keys = [k for k in batch[0].keys() 
                        if k.endswith("_ids") and k not in ["input_ids"]]
    for key in demographic_keys:
        tensors = [b[key] for b in batch]
        stacked = torch.stack(tensors)
        result[key] = stacked

    return result


def enhanced_loss_function(p_hat, targets, labels, lambda_l1=0.4, lambda_kl=0.3, lambda_ce=0.3):
    """Enhanced loss function combining L1, KL divergence, and cross-entropy."""
    l1_loss = torch.sum(torch.abs(p_hat - targets), dim=-1).mean()
    kl_loss = F.kl_div(torch.log(p_hat + 1e-12), targets, reduction="batchmean")
    ce_loss = F.cross_entropy(p_hat, labels)
    
    total_loss = lambda_l1 * l1_loss + lambda_kl * kl_loss + lambda_ce * ce_loss
    
    return total_loss, {
        'l1_loss': l1_loss.item(),
        'kl_loss': kl_loss.item(),
        'ce_loss': ce_loss.item(),
        'total_loss': total_loss.item()
    }


def ensemble_evaluate(model, dataloader, device):
    """Evaluate ensemble model with multiple metrics."""
    model.eval()
    total_metrics = defaultdict(float)
    n_examples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(texts=batch["texts"], **demographic_inputs)
            p_hat = torch.softmax(logits, dim=-1)
            
            # Calculate multiple metrics
            l1_dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1).mean()
            kl_loss = F.kl_div(torch.log(p_hat + 1e-12), batch["dist"], reduction="batchmean")
            ce_loss = F.cross_entropy(logits, batch["labels"])
            
            total_metrics['l1_dist'] += l1_dist.item()
            total_metrics['kl_loss'] += kl_loss.item()
            total_metrics['ce_loss'] += ce_loss.item()
            n_examples += 1
            
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= n_examples
    
    # Convert defaultdict to regular dict to avoid type issues
    result_metrics = dict(total_metrics)
    result_metrics['predictions'] = all_predictions
    result_metrics['targets'] = all_targets
    
    return result_metrics


def train_ensemble(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_ds = LLMPromptedDataset(args.train_file, args.annot_meta, args.max_length, 
                                 use_prompts=args.use_prompts, augment=args.augment)
    val_ds = LLMPromptedDataset(args.val_file, args.annot_meta, args.max_length, 
                               use_prompts=args.use_prompts, augment=False) if args.val_file else None

    print(f"Training samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")
    print(f"Using demographic fields: {list(train_ds.active_field_keys.keys())}")
    print(f"Demographic vocabulary sizes: {train_ds.vocab_sizes}")

    # Model configurations for ensemble - Better models for paraphrase detection
    model_configs = [
        {'base_name': 'all-mpnet-base-v2', 'model_type': 'cross_attention'},  # Best SBERT for semantic similarity
        {'base_name': 'roberta-base', 'model_type': 'simple_concat'},  # RoBERTa base
        {'base_name': 'sentence-transformers/all-MiniLM-L6-v2', 'model_type': 'demographic_only'},  # Fast SBERT
        {'base_name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'model_type': 'cross_attention'},  # Paraphrase-specific SBERT
    ]

    print(f"Creating ensemble with {len(model_configs)} models:")
    for i, config in enumerate(model_configs):
        print(f"  Model {i+1}: {config['base_name']} ({config['model_type']})")

    # Create ensemble model
    model = EnsembleParDemogModel(
        model_configs=model_configs,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
    )
    model.to(device)

    # Enhanced optimizer with different learning rates for each model
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = []
    
    for model_idx, single_model in enumerate(model.models):
        lr_multiplier = 1.0 if model_idx == 0 else 0.8  # Slightly lower LR for ensemble models
        
        for name, param in single_model.named_parameters():
            if "text_model" in name:
                lr = args.lr * 0.1 * lr_multiplier
            elif "demographic_embeddings" in name:
                lr = args.lr * lr_multiplier
            else:
                lr = args.lr * 2.0 * lr_multiplier
            
            if any(nd in name for nd in no_decay):
                grouped_params.append({"params": param, "weight_decay": 0.0, "lr": lr})
            else:
                grouped_params.append({"params": param, "weight_decay": args.weight_decay, "lr": lr})

    if args.optimizer == "adamw":
        optimizer = AdamW(grouped_params)
    else:
        optimizer = RAdam(grouped_params)

    # Data loaders
    sampler = WeightedRandomSampler(
        weights=[1.0] * len(train_ds),  # Equal weights for now
        num_samples=len(train_ds),
        replacement=True
    ) if args.balance else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=ensemble_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                  collate_fn=ensemble_collate_fn, num_workers=4) if val_ds else None
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_metric = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    train_loss_history = []
    val_dist_history = []
    lr_history = []

    print(f"Total training steps: {total_steps}")
    print(f"Initial learning rate: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(texts=batch["texts"], **demographic_inputs)
            p_hat = torch.softmax(logits, dim=-1)

            # Enhanced loss function
            loss, loss_components = enhanced_loss_function(
                p_hat, batch["dist"], batch["labels"],
                lambda_l1=args.lambda_l1, lambda_kl=args.lambda_kl, lambda_ce=args.lambda_ce
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Track losses
            for key, value in loss_components.items():
                epoch_losses[key] += value
            step_count += 1
            
            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_losses['total_loss'] / step_count
                avg_l1 = epoch_losses['l1_loss'] / step_count
                avg_kl = epoch_losses['kl_loss'] / step_count
                avg_ce = epoch_losses['ce_loss'] / step_count
                tqdm.write(f"Epoch {epoch} step {step}: total_loss={avg_loss:.4f}, l1_loss={avg_l1:.4f}, kl_loss={avg_kl:.4f}, ce_loss={avg_ce:.4f}, lr={current_lr:.2e}")

        if val_loader:
            val_metrics = ensemble_evaluate(model, val_loader, device)
            val_dist = val_metrics['l1_dist']
            print(f"Validation metrics after epoch {epoch}:")
            print(f"  L1 Distance: {val_metrics['l1_dist']:.4f}")
            print(f"  KL Loss: {val_metrics['kl_loss']:.4f}")
            print(f"  CE Loss: {val_metrics['ce_loss']:.4f}")
            
            if val_dist < best_metric:
                best_metric = val_dist
                best_epoch = epoch
                epochs_no_improve = 0
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                print(f"New best model saved to {save_path}")
            else:
                epochs_no_improve += 1

        train_loss_history.append(epoch_losses['total_loss'] / step_count)
        if val_loader:
            val_dist_history.append(val_dist)
        lr_history.append(scheduler.get_last_lr()[0])

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    final_path = os.path.join(args.output_dir, "last_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))

    print(f"\nTraining completed. Best validation distance: {best_metric:.4f}")
    if val_dist_history:
        print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble training for Paraphrase detection with LLM prompting.")
    parser.add_argument("--train_file", type=str, default="dataset/Paraphrase/Paraphrase_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/Paraphrase/Paraphrase_dev.json")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_par_ensemble")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=6, help="Smaller batch size for ensemble")
    parser.add_argument("--epochs", type=int, default=12, help="More epochs for ensemble")
    parser.add_argument("--lr", type=float, default=8e-6, help="Lower learning rate for ensemble")
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--lambda_l1", type=float, default=0.4, help="Weight for L1 loss")
    parser.add_argument("--lambda_kl", type=float, default=0.3, help="Weight for KL loss")
    parser.add_argument("--lambda_ce", type=float, default=0.3, help="Weight for cross-entropy loss")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json")
    parser.add_argument("--dem_dim", type=int, default=32, help="Demographic embedding dimension")
    parser.add_argument("--patience", type=int, default=6, help="Patience for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--use_prompts", action="store_true", help="Use LLM-style prompts")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--optimizer", choices=["adamw", "radam"], default="adamw")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_ensemble(args) 