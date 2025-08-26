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


class VariErrNLIDemogModel(torch.nn.Module):
    """RoBERTa-Large model with demographic embeddings and SBERT embeddings for annotator-aware NLI."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 8, sbert_dim: int = 384, dropout_rate: float = 0.3):
        super().__init__()
        from transformers import AutoModel
        from sentence_transformers import SentenceTransformer

        # RoBERTa-Large as the main model (ensure we're using roberta-large)
        if "roberta-large" not in base_name.lower():
            print(f"Warning: Expected roberta-large but got {base_name}. Using roberta-large as base model.")
            base_name = "roberta-large"
        
        self.text_model = AutoModel.from_pretrained(base_name)
        
        # SBERT model for additional embeddings (pretrained)
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Freeze SBERT model parameters to use as a pretrained feature extractor
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        # Create demographic embeddings dynamically based on vocab_sizes
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        # Get embedding dimension from RoBERTa-Large (should be 1024)
        hidden_size = self.text_model.config.hidden_size
        print(f"RoBERTa-Large hidden size: {hidden_size}")
        print(f"SBERT embedding dimension: {sbert_dim}")
        
        num_demog_fields = len(vocab_sizes)
        total_dim = hidden_size + sbert_dim + num_demog_fields * dem_dim
        print(f"Total concatenated dimension: {total_dim} (RoBERTa: {hidden_size} + SBERT: {sbert_dim} + Demographics: {num_demog_fields * dem_dim})")
        
        # Layer normalization and dropout for regularization
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(total_dim, 3)  # NLI has 3 classes

    def forward(self, *, input_ids, attention_mask, texts, **demographic_inputs):
        # Get RoBERTa embeddings from the main model
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Get SBERT embeddings (frozen pretrained model)
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True)
            # Ensure SBERT embeddings are on the same device as RoBERTa embeddings
            sbert_embeddings = sbert_embeddings.to(pooled.device)
        
        # Get demographic embeddings dynamically
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        # Concatenate all vectors: RoBERTa + SBERT + demographics
        all_vectors = [pooled, sbert_embeddings] + demographic_vectors
        concat = torch.cat(all_vectors, dim=-1)
            
        # Apply layer normalization and dropout for regularization
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class VariErrNLIDataset(Dataset):
    """Dataset with demographic embeddings for VariErr NLI detection."""

    PAD_IDX = 0  # padding
    UNK_IDX = 1  # unknown / missing

    # Fields available in VariErrNLI annotator metadata
    FIELD_KEYS = {
        "gender": "Gender",
        "age": "Age",  # Will be binned
        "nationality": "Nationality",
        "education": "Education",
    }

    # Reduced field set for better performance (same as full set for VariErrNLI)
    REDUCED_FIELD_KEYS = {
        "gender": "Gender", 
        "nationality": "Nationality",
        "education": "Education",
    }

    @staticmethod
    def get_age_bin(age):
        """Convert age to age bin."""
        if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
            return "<UNK>"
        try:
            age_int = int(age)
            if age_int < 25:
                return "18-24"
            elif age_int < 35:
                return "25-34"
            elif age_int < 45:
                return "35-44"
            elif age_int < 55:
                return "45-54"
            else:
                return "55+"
        except (ValueError, TypeError):
            return "<UNK>"

    def __init__(self, path: str, tokenizer: AutoTokenizer, annot_meta_path: str, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Always use reduced demographics for better performance
        self.active_field_keys = self.REDUCED_FIELD_KEYS

        self.texts = []
        self.labels = []
        self.dists = []
        
        # Initialize storage for active fields only - storing single values per example
        self.demographic_ids = {field: [] for field in self.active_field_keys}

        # Load annotator metadata
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            self.annot_meta = json.load(f)

        # Label mapping for NLI
        self.label_to_id = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # Build vocabularies for demographic fields
        self.vocab = {
            field: {"<PAD>": self.PAD_IDX, "<UNK>": self.UNK_IDX}
            for field in self.active_field_keys
        }

        # Build vocabulary from all annotators
        for ann_data in self.annot_meta.values():
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

        # Load data
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex_id, ex in data.items():
            # Build input text for NLI: context + [SEP] + statement
            context = ex["text"]["context"]
            statement = ex["text"]["statement"]
            full_text = f"{context} {tokenizer.sep_token} {statement}".strip()

            # Create soft distribution from annotations
            annotators = ex.get("annotators", "").split(",")
            annotations = ex.get("annotations", {})
            
            # Create soft distribution from ratings
            label_counts = Counter()
            for annotator in annotators:
                annotator = annotator.strip()
                if annotator in annotations:
                    label = annotations[annotator]
                    if label in self.label_to_id:
                        label_counts[self.label_to_id[label]] += 1

            # Create soft distribution
            total_annotations = sum(label_counts.values())
            if total_annotations > 0:
                dist = np.array([0.0] * 3, dtype=np.float32)
                for label_id, count in label_counts.items():
                    dist[label_id] = count / total_annotations
            else:
                dist = np.array([1/3, 1/3, 1/3], dtype=np.float32)  # Uniform if no valid annotations

            dist /= dist.sum()
            hard_label = int(np.argmax(dist))

            ann_list = [a.strip() for a in annotators if a.strip()] if annotators else []

            # Create separate examples for each annotator
            for ann_tag in ann_list:
                ann_num = ann_tag  # VariErrNLI uses Ann1, Ann2, etc. directly
                meta = self.annot_meta.get(ann_num, {})
                
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
        
        # Add demographic fields dynamically
        for field in self.active_field_keys:
            result[f"{field}_ids"] = torch.tensor(self.demographic_ids[field][idx], dtype=torch.long)
        
        return result


def collate_fn(batch):
    """Pads text and handles single demographic values."""

    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    texts = [b["texts"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad token id = 1
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    result = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "texts": texts,
        "labels": labels,
        "dist": dists,
    }
    
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
            
            # Prepare demographic inputs dynamically (exclude input_ids, attention_mask, texts)
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
    
    # Calculate prediction bias for each class
    pred_bias_contradiction = np.mean(predictions[:, 0] - targets[:, 0])
    pred_bias_entailment = np.mean(predictions[:, 1] - targets[:, 1])
    pred_bias_neutral = np.mean(predictions[:, 2] - targets[:, 2])
    
    pred_std = np.std(predictions, axis=0)
    target_std = np.std(targets, axis=0)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Class distribution comparison
    class_names = ['Contradiction', 'Entailment', 'Neutral']
    pred_means = np.mean(predictions, axis=0)
    target_means = np.mean(targets, axis=0)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, pred_means, width, label='Predictions', alpha=0.7)
    axes[0, 0].bar(x + width/2, target_means, width, label='Targets', alpha=0.7)
    axes[0, 0].set_xlabel('Classes')
    axes[0, 0].set_ylabel('Mean Probability')
    axes[0, 0].set_title(f'Epoch {epoch}: Class Distribution Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction vs Target scatter for dominant class
    dominant_class = np.argmax(target_means)
    axes[0, 1].scatter(targets[:, dominant_class], predictions[:, dominant_class], alpha=0.5, s=1)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel(f'True P({class_names[dominant_class]})')
    axes[0, 1].set_ylabel(f'Predicted P({class_names[dominant_class]})')
    axes[0, 1].set_title(f'Epoch {epoch}: {class_names[dominant_class]} Prediction vs Target')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution for each class
    errors = predictions - targets
    colors = ['red', 'green', 'blue']
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        axes[1, 0].hist(errors[:, i], bins=30, alpha=0.5, label=f'{class_name}', color=color)
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Epoch {epoch}: Error Distribution by Class')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confusion matrix (argmax predictions)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    confusion = np.zeros((3, 3))
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1
    
    im = axes[1, 1].imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title(f'Epoch {epoch}: Confusion Matrix')
    tick_marks = np.arange(3)
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].set_yticklabels(class_names)
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, int(confusion[i, j]), ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    accuracy = np.sum(pred_labels == true_labels) / len(true_labels)
    
    return {
        'pred_bias_contradiction': pred_bias_contradiction,
        'pred_bias_entailment': pred_bias_entailment,
        'pred_bias_neutral': pred_bias_neutral,
        'pred_std': pred_std.tolist(),
        'target_std': target_std.tolist(),
        'mean_error': np.mean(np.abs(errors)),
        'accuracy': accuracy
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
        accuracies = [a['accuracy'] for a in analysis_history]
        mean_errors = [a['mean_error'] for a in analysis_history]
        axes[1, 1].plot(epochs_range, accuracies, marker='o', label='Accuracy', color='purple')
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(epochs_range, mean_errors, marker='s', label='Mean Abs Error', color='brown')
        axes[1, 1].set_title('Accuracy and Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy', color='purple')
        ax_twin.set_ylabel('Mean Abs Error', color='brown')
        axes[1, 1].tick_params(axis='y', labelcolor='purple')
        ax_twin.tick_params(axis='y', labelcolor='brown')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics to JSON
    metrics = {
        'train_loss': [float(x) for x in train_loss_history],
        'val_distance': [float(x) for x in val_dist_history] if val_dist_history else [],
        'learning_rates': [float(x) for x in lr_history],
        'analysis': analysis_history,
        'best_epoch': int(best_epoch) if val_dist_history else None,
        'best_metric': float(best_metric) if val_dist_history else None
    }
    
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def visualize_demog_embeddings(model, dataset: VariErrNLIDataset, output_dir: str):
    """Save 2-D PCA scatter plots of demographic embeddings."""
    os.makedirs(output_dir, exist_ok=True)

    # Get field names and their display names
    field_display_names = {
        "gender": "Gender", 
        "nationality": "Nationality",
        "education": "Education"
    }

    for field_name, emb_layer in model.demographic_embeddings.items():
        if field_name not in dataset.vocab:
            continue
            
        display_name = field_display_names.get(field_name, field_name.replace("_", " ").title())
        
        emb = emb_layer.weight.detach().cpu().numpy()
        if emb.shape[0] <= 2:
            print(f"Skipping {display_name} - not enough embeddings (only {emb.shape[0]})")
            continue
        emb = emb[2:]  # Skip PAD and UNK
        labels = list(dataset.vocab[field_name].keys())[2:]  # Skip PAD and UNK
        if len(labels) != emb.shape[0]:
            print(f"Skipping {display_name} - mismatch between labels ({len(labels)}) and embeddings ({emb.shape[0]})")
            continue
        
        # Check if we have enough unique embeddings for PCA
        if emb.shape[0] < 2:
            print(f"Skipping {display_name} - need at least 2 unique values for visualization")
            continue
        
        X = emb - emb.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Check if we can do 2D PCA
        if Vt.shape[0] < 2:
            print(f"Skipping {display_name} - only {Vt.shape[0]} principal component(s) available")
            continue
        
        # Project to 2D (or 1D if only 1 component available)
        n_components = min(2, Vt.shape[0], X.shape[0])
        coords = X.dot(Vt.T[:, :n_components])
        
        plt.figure(figsize=(8, 6))
        
        if n_components == 2:
            # Standard 2D plot
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=40)
            for i, label in enumerate(labels):
                if i % max(1, len(labels)//30) == 0:
                    plt.text(coords[i, 0], coords[i, 1], label, fontsize=8, alpha=0.7)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"{display_name} Embeddings (PCA-2D)")
        else:
            # 1D plot (project to y-axis, use indices for x-axis)
            x_pos = np.arange(len(coords))
            plt.scatter(x_pos, coords[:, 0], alpha=0.7, s=40)
            for i, label in enumerate(labels):
                if i % max(1, len(labels)//10) == 0:  # Show more labels in 1D
                    plt.text(x_pos[i], coords[i, 0], label, fontsize=8, alpha=0.7, rotation=45)
            plt.xlabel("Index")
            plt.ylabel("PC1")
            plt.title(f"{display_name} Embeddings (PCA-1D)")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{field_name}_emb_pca.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {display_name} embedding visualisation → {fname}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = VariErrNLIDataset(args.train_file, tokenizer, args.annot_meta, args.max_length)
    val_ds = VariErrNLIDataset(args.val_file, tokenizer, args.annot_meta, args.max_length) if args.val_file else None

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

    model = VariErrNLIDemogModel(
        base_name=args.model_name,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
        sbert_dim=args.sbert_dim,
        dropout_rate=args.dropout_rate,
    )
    model.to(device)

    frozen_layers = []
    if getattr(args, "freeze_layers", 0) > 0:
        for layer in model.text_model.encoder.layer[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        frozen_layers = list(range(args.freeze_layers))
        if frozen_layers:
            print(f"Frozen transformer layers: {frozen_layers} for first {args.freeze_epochs} epoch(s)")

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
        if frozen_layers and epoch > args.freeze_epochs:
            for layer_idx in frozen_layers:
                for p in model.text_model.encoder.layer[layer_idx].parameters():
                    p.requires_grad = True
            print(f"Unfroze layers {frozen_layers} at start of epoch {epoch}")
            frozen_layers = []

        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare demographic inputs dynamically (exclude input_ids, attention_mask, texts)
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )

            p_hat = torch.softmax(logits, dim=-1)
            # Soft cross-entropy: -sum(target_dist * log(predicted_dist))
            cross_entropy_loss = -torch.sum(batch["dist"] * torch.log(p_hat + 1e-12), dim=-1).mean()
            loss = cross_entropy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            epoch_loss += loss.item()
            step_count += 1
            
            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step_count
                tqdm.write(f"Epoch {epoch} step {step}: cross_entropy_loss={avg_loss:.4f}, lr={current_lr:.2e}")

        if val_loader:
            val_dist, predictions, targets = evaluate(model, val_loader, device)
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            
            analysis = analyze_predictions(predictions, targets, epoch, args.output_dir)
            analysis_history.append(analysis)
            
            print(f"Epoch {epoch} Analysis:")
            print(f"  Accuracy: {analysis['accuracy']:.4f}")
            print(f"  Mean absolute error: {analysis['mean_error']:.4f}")
            print(f"  Prediction bias (C/E/N): {analysis['pred_bias_contradiction']:.4f}/{analysis['pred_bias_entailment']:.4f}/{analysis['pred_bias_neutral']:.4f}")
            
            if val_dist < best_metric:
                best_metric = val_dist
                best_epoch = epoch
                epochs_no_improve = 0
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                tokenizer.save_pretrained(save_path)
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
    tokenizer.save_pretrained(final_path)

    visualize_demog_embeddings(model, train_ds, args.output_dir)
    plot_training_metrics(train_loss_history, val_dist_history, lr_history, analysis_history, best_epoch, best_metric, args.output_dir)

    print(f"\nTraining completed. Best validation distance: {best_metric:.4f}")
    if val_dist_history:
        print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-Large with SBERT and demographic embeddings on VariErr NLI using cross-entropy loss with soft labels.")
    parser.add_argument("--train_file", type=str, default="dataset/VariErrNLI/VariErrNLI_train.json", help="Path to VariErrNLI_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/VariErrNLI/VariErrNLI_dev.json", help="Path to VariErrNLI_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="HF model name")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_varier_nli")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")

    parser.add_argument("--annot_meta", type=str, default="dataset/VariErrNLI/VariErrNLI_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of epochs to freeze layers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")

    args = parser.parse_args()
    train(args)