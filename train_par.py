import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ParDemogModel(torch.nn.Module):
    """RoBERTa-Large model with demographic embeddings and SBERT embeddings for annotator-aware paraphrase detection."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 8, sbert_dim: int = 384, dropout_rate: float = 0.3, num_classes: int = 11):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer

        # RoBERTa-Large as the main model
        self.roberta_model = AutoModel.from_pretrained(base_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # SBERT model for additional embeddings
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Freeze SBERT model parameters
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        
        # Create embeddings dynamically based on vocab_sizes
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        # Get embedding dimension from RoBERTa
        roberta_dim = self.roberta_model.config.hidden_size  # Usually 1024 for RoBERTa-Large
        
        # SBERT dimension (from all-MiniLM-L6-v2)
        self.sbert_dim = sbert_dim
        
        num_demog_fields = len(vocab_sizes)
        total_dim = roberta_dim + sbert_dim + num_demog_fields * dem_dim
        
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(total_dim, num_classes)

    def forward(self, *, input_ids, attention_mask, texts, **demographic_inputs):
        # Get RoBERTa embeddings
        roberta_outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        roberta_embeddings = roberta_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get SBERT embeddings
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True)
        
        # Get demographic embeddings dynamically
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        # Concatenate all vectors: RoBERTa + SBERT + demographics
        all_vectors = [roberta_embeddings, sbert_embeddings] + demographic_vectors
        concat = torch.cat(all_vectors, dim=-1)
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class ParDataset(Dataset):
    """Dataset with demographic embeddings for paraphrase detection."""

    PAD_IDX = 0  # padding
    UNK_IDX = 1  # unknown / missing

    FIELD_KEYS = {
        "age": "Age",  # Will be binned
        "gender": "Gender",
        "ethnicity": "Ethnicity simplified",
        "country_birth": "Country of birth",
        "country_residence": "Country of residence",
        "nationality": "Nationality",
        "student": "Student status",
        "employment": "Employment status",
    }

    # Reduced field set for better performance
    REDUCED_FIELD_KEYS = {
        "age": "Age",  # Will be binned
        "gender": "Gender", 
        "ethnicity": "Ethnicity simplified",
        "country_residence": "Country of residence",
        "employment": "Employment status",
    }

    @staticmethod
    def get_age_bin(age):
        """Convert age to age bin."""
        if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
            return "<UNK>"
        try:
            age = float(age)
            if age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            else:
                return "55+"
        except (ValueError, TypeError):
            return "<UNK>"

    def __init__(
        self,
        path: str,
        annot_meta_path: str,
        tokenizer=None,
        max_length: int = 512,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Always use reduced demographics for better performance
        self.active_field_keys = self.REDUCED_FIELD_KEYS

        self.texts = []
        self.labels = []
        self.dists = []
        
        # Initialize storage for active fields only - now storing single values per example
        self.demographic_ids = {field: [] for field in self.active_field_keys}

        with open(annot_meta_path, "r", encoding="utf-8") as f:
            annot_meta = json.load(f)

        self.annot_meta = annot_meta

        self.vocab = {
            field: {"<PAD>": self.PAD_IDX, "<UNK>": self.UNK_IDX}
            for field in self.active_field_keys
        }

        # Build vocabulary from all annotators
        for ann_data in annot_meta.values():
            for field, json_key in self.active_field_keys.items():
                if field == "age":
                    # Special handling for age - convert to age bin
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
            full_text = f"{question1} [SEP] {question2}".strip()

            soft_label = ex.get("soft_label", {})
            if not soft_label or soft_label == "":
                continue
            
            # Convert to 11-class distribution (-5 to 5)
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

            # Create separate examples for each annotator
            for ann_tag in ann_list:
                ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                meta = annot_meta.get(ann_num, {})
                
                # Get demographic info for this specific annotator
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

                # Add this annotator's example
                self.texts.append(full_text)
                self.dists.append(dist)
                self.labels.append(hard_label)
                
                # Store single demographic IDs for this annotator
                for field in self.active_field_keys:
                    self.demographic_ids[field].append(annotator_demog_ids[field])

            # If no annotators, create one example with UNK demographic info
            if not ann_list:
                self.texts.append(full_text)
                self.dists.append(dist)
                self.labels.append(hard_label)
                
                # Store UNK demographic IDs
                for field in self.active_field_keys:
                    self.demographic_ids[field].append(self.UNK_IDX)

        print(f"Dataset created with {len(self.texts)} examples (expanded from {len(data)} original examples)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize text if tokenizer is available
        if self.tokenizer is not None:
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            result = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "texts": self.texts[idx],  # Keep original text for SBERT
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            }
        else:
            result = {
                "texts": self.texts[idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            }
        
        # Add demographic fields dynamically
        for field in self.active_field_keys:
            result[f"{field}_ids"] = torch.tensor(self.demographic_ids[field][idx], dtype=torch.long)
        
        return result


def collate_fn(batch):
    """Handles tokenized inputs and demographic values for paraphrase dataset."""

    texts = [b["texts"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    result = {
        "texts": texts,
        "labels": labels,
        "dist": dists,
    }
    
    # Handle tokenized inputs if present
    if "input_ids" in batch[0]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad token id = 1
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
    
    # Dynamically handle demographic fields - now single values, not lists
    demographic_keys = [k for k in batch[0].keys() 
                        if k.endswith("_ids") and k not in ["input_ids"]]
    for key in demographic_keys:
        tensors = [b[key] for b in batch]
        # Stack single values instead of padding lists
        stacked = torch.stack(tensors)
        result[key] = stacked

    return result


def build_sampler(labels):
    """Create WeightedRandomSampler to balance classes."""
    counts = Counter(labels)
    total = float(sum(counts.values()))
    num_classes = len(counts)
    class_weights = {c: total / (num_classes * cnt) for c, cnt in counts.items()}
    weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model, dataloader, device):
    """Return mean Manhattan (L1) distance between predicted and true distributions."""
    model.eval()
    total_dist = 0.0
    n_examples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare demographic inputs dynamically (exclude texts)
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )
            p_hat = torch.softmax(logits, dim=-1)
            dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
            total_dist += dist.sum().item()
            n_examples += dist.numel()
            
            # Store predictions and targets for analysis
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    return total_dist / n_examples if n_examples else 0.0, all_predictions, all_targets


def analyze_predictions(predictions, targets, epoch, output_dir):
    """Analyze prediction distributions and save plots."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate prediction bias (for class 5, which is the highest rating)
    pred_bias = np.mean(predictions[:, 10] - targets[:, 10])  # bias toward class 5
    pred_std = np.std(predictions[:, 10])
    target_std = np.std(targets[:, 10])
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Prediction vs Target scatter (for highest class)
    axes[0, 0].scatter(targets[:, 10], predictions[:, 10], alpha=0.5, s=1)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('True P(rating=5)')
    axes[0, 0].set_ylabel('Predicted P(rating=5)')
    axes[0, 0].set_title(f'Epoch {epoch}: Prediction vs Target')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction distribution
    axes[0, 1].hist(predictions[:, 10], bins=50, alpha=0.7, label='Predictions')
    axes[0, 1].hist(targets[:, 10], bins=50, alpha=0.7, label='Targets')
    axes[0, 1].set_xlabel('P(rating=5)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Epoch {epoch}: Distribution Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = predictions[:, 10] - targets[:, 10]
    axes[1, 0].hist(errors, bins=50, alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Epoch {epoch}: Error Distribution (bias={pred_bias:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error vs Target
    axes[1, 1].scatter(targets[:, 10], errors, alpha=0.5, s=1)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('True P(rating=5)')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title(f'Epoch {epoch}: Error vs Target')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'pred_bias': pred_bias,
        'pred_std': pred_std,
        'target_std': target_std,
        'mean_error': np.mean(np.abs(errors))
    }


def plot_training_metrics(train_loss_history, val_dist_history, lr_history, analysis_history, best_epoch, best_metric, output_dir):
    """Plot and save training metrics and curves."""
    epochs_range = list(range(1, len(train_loss_history) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss vs validation distance
    ax_comb = axes[0, 0]
    ax_comb.plot(epochs_range, train_loss_history, marker='o', color='blue', label='Train Loss')
    if val_dist_history:
        ax_comb_twin = ax_comb.twinx()
        ax_comb_twin.plot(epochs_range, val_dist_history, marker='s', color='orange', label='Val L1 Dist')
        ax_comb_twin.set_ylabel('Validation Distance', color='orange')
        ax_comb_twin.tick_params(axis='y', labelcolor='orange')
    ax_comb.set_title('Training Loss vs Validation Distance')
    ax_comb.set_xlabel('Epoch')
    ax_comb.set_ylabel('Training Loss', color='blue')
    ax_comb.tick_params(axis='y', labelcolor='blue')
    ax_comb.grid(True, alpha=0.3)

    # Validation distance with best epoch marker
    if val_dist_history:
        axes[0, 1].plot(epochs_range, val_dist_history, marker='o', color='orange')
        axes[0, 1].set_title('Validation Manhattan Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].scatter([best_epoch], [val_dist_history[best_epoch - 1]], color='red', s=100, zorder=5)
        axes[0, 1].text(best_epoch, val_dist_history[best_epoch - 1], f'  best={val_dist_history[best_epoch - 1]:.3f}', fontsize=10)

    # Learning rate schedule
    axes[1, 0].plot(epochs_range, lr_history, marker='o', color='green')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Analysis metrics
    if analysis_history:
        biases = [a['pred_bias'] for a in analysis_history]
        errors = [a['mean_error'] for a in analysis_history]
        axes[1, 1].plot(epochs_range, biases, marker='o', label='Prediction Bias', color='purple')
        axes[1, 1].plot(epochs_range, errors, marker='s', label='Mean Abs Error', color='brown')
        axes[1, 1].set_title('Prediction Analysis')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Separate combined plot for quick reference
    if val_dist_history:
        fig2, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs_range, train_loss_history, marker='o', color='blue', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, val_dist_history, marker='s', color='orange', label='Val L1 Dist')
        ax2.set_ylabel('Validation Distance', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax1.set_title('Train Loss vs Val Distance')
        ax1.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'loss_vs_val.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)

    # Save metrics to JSON
    metrics = {
        'train_loss': [float(x) for x in train_loss_history],
        'val_distance': [float(x) for x in val_dist_history] if val_dist_history else [],
        'learning_rates': [float(x) for x in lr_history],
        'analysis': [
            {
                'pred_bias': float(a['pred_bias']),
                'pred_std': float(a['pred_std']),
                'target_std': float(a['target_std']),
                'mean_error': float(a['mean_error'])
            } for a in analysis_history
        ],
        'best_epoch': int(best_epoch) if val_dist_history else None,
        'best_metric': float(best_metric) if val_dist_history else None
    }
    
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_ds = ParDataset(args.train_file, args.annot_meta, tokenizer, args.max_length)
    val_ds = ParDataset(args.val_file, args.annot_meta, tokenizer, args.max_length) if args.val_file else None

    print(f"Training samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")
    print(f"Using reduced demographic fields: {list(train_ds.active_field_keys.keys())}")
    print(f"Demographic vocabulary sizes: {train_ds.vocab_sizes}")

    sampler = build_sampler(train_ds.labels) if args.balance else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_ds else None
    )

    model = ParDemogModel(
        base_name=args.model_name,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
        sbert_dim=args.sbert_dim,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
    )
    model.to(device)

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

    best_metric = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    train_loss_history = []
    val_dist_history = []
    lr_history = []
    analysis_history = []

    print(f"Total training steps: {total_steps}")
    print(f"Initial learning rate: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_kl_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare demographic inputs dynamically (exclude texts)
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )

            p_hat = torch.softmax(logits, dim=-1)
            l1_loss = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1).mean()
            kl_loss = F.kl_div(torch.log(p_hat + 1e-12), batch["dist"], reduction="batchmean")
            loss = (1 - args.lambda_kl) * l1_loss + args.lambda_kl * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            epoch_loss += loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_kl_loss += kl_loss.item()
            step_count += 1
            
            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step_count
                avg_l1 = epoch_l1_loss / step_count
                avg_kl = epoch_kl_loss / step_count
                tqdm.write(f"Epoch {epoch} step {step}: total_loss={avg_loss:.4f}, l1_loss={avg_l1:.4f}, kl_loss={avg_kl:.4f}, lr={current_lr:.2e}")

        if val_loader:
            val_dist, predictions, targets = evaluate(model, val_loader, device)
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            
            analysis = analyze_predictions(predictions, targets, epoch, args.output_dir)
            analysis_history.append(analysis)
            
            print(f"Epoch {epoch} Analysis:")
            print(f"  Prediction bias: {analysis['pred_bias']:.4f}")
            print(f"  Mean absolute error: {analysis['mean_error']:.4f}")
            print(f"  Prediction std: {analysis['pred_std']:.4f}")
            print(f"  Target std: {analysis['target_std']:.4f}")
            
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

        train_loss_history.append(epoch_loss / step_count)
        if val_loader:
            val_dist_history.append(val_dist)
        lr_history.append(scheduler.get_last_lr()[0])

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    final_path = os.path.join(args.output_dir, "last_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))

    visualize_demog_embeddings(model, train_ds, args.output_dir)
    plot_training_metrics(train_loss_history, val_dist_history, lr_history, analysis_history, best_epoch, best_metric, args.output_dir)

    print(f"\nTraining completed. Best validation distance: {best_metric:.4f}")
    if val_dist_history:
        print(f"Best epoch: {best_epoch}")


def visualize_demog_embeddings(model, dataset: ParDataset, output_dir: str):
    """Save 2-D PCA scatter plots of demographic embeddings."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    # Get field names and their display names
    field_display_names = {
        "age": "Age",
        "gender": "Gender", 
        "ethnicity": "Ethnicity simplified",
        "country_birth": "Country of birth",
        "country_residence": "Country of residence", 
        "nationality": "Nationality",
        "student": "Student status",
        "employment": "Employment status"
    }

    for field_name, emb_layer in model.demographic_embeddings.items():
        if field_name not in dataset.vocab:
            continue
            
        display_name = field_display_names.get(field_name, field_name.replace("_", " ").title())
        
        emb = emb_layer.weight.detach().cpu().numpy()
        if emb.shape[0] <= 2:
            continue
        emb = emb[2:]  # Skip PAD and UNK
        labels = list(dataset.vocab[field_name].keys())[2:]  # Skip PAD and UNK
        if len(labels) != emb.shape[0]:
            continue
        
        X = emb - emb.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = X.dot(Vt.T[:, :2])

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=40)
        for i, label in enumerate(labels):
            if i % max(1, len(labels)//30) == 0:
                plt.text(coords[i, 0], coords[i, 1], label, fontsize=8, alpha=0.7)
        plt.title(f"{display_name} Embeddings (PCA-2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{field_name}_emb_pca.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {display_name} embedding visualisation → {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-Large with SBERT embeddings and demographic features on Paraphrase detection (soft labels).")
    parser.add_argument("--train_file", type=str, default="dataset/Paraphrase/Paraphrase_train.json", help="Path to Paraphrase_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/Paraphrase/Paraphrase_dev.json", help="Path to Paraphrase_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="RoBERTa model name")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_par")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--lambda_kl", type=float, default=0.3, help="Weight for KL in mixed loss (0=L1 only)")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes (Likert scale -5 to 5)")

    args = parser.parse_args()
    train(args) 