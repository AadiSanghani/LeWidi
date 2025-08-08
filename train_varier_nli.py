def analyze_predictions(predictions, targets, epoch, output_dir):
    """Analyze prediction distributions and save plots for NLI."""
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
    import matplotlib.pyplot as plt
    import numpy as np
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

    # Generate demographic distribution tables
    generate_demographic_distribution_table(train_ds, args.output_dir, "train")
    if val_ds:
        generate_demographic_distribution_table(val_ds, args.output_dir, "validation")

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
            print(f"Frozen transformer layers:import argparse")
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


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
    all_embeddings = []
    all_labels = []
    
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
            
            # Get embeddings for PCA analysis by extracting the final layer features
            with torch.no_grad():
                # Get RoBERTa embeddings
                roberta_outputs = model.text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                roberta_pooled = roberta_outputs.last_hidden_state[:, 0]  # [CLS] token
                
                # Get SBERT embeddings
                sbert_embeddings = model.sbert_model.encode(batch["texts"], convert_to_tensor=True)
                sbert_embeddings = sbert_embeddings.to(roberta_pooled.device)
                
                # Get demographic embeddings
                demographic_embeds = []
                for field, emb_layer in model.demographic_embeddings.items():
                    field_key = f"{field}_ids"
                    if field_key in demographic_inputs:
                        embed = emb_layer(demographic_inputs[field_key])
                        demographic_embeds.append(embed)
                
                # Combine embeddings (same as in forward pass but before norm/dropout/classifier)
                combined_embeddings = [roberta_pooled, sbert_embeddings] + demographic_embeds
                combined = torch.cat(combined_embeddings, dim=-1)
                
                all_embeddings.extend(combined.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
            
            p_hat = torch.softmax(logits, dim=-1)
            dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
            total_dist += dist.sum().item()
            n_examples += dist.numel()
            
            # Store predictions and targets for analysis
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    return total_dist / n_examples if n_examples else 0.0, all_predictions, all_targets, all_embeddings, all_labels


def generate_pca_plot(embeddings, labels, output_dir, model_name="model", epoch=None, save_data=True):
    """
    Universal PCA plot generation function that creates and saves PCA visualizations.
    
    Args:
        embeddings: numpy array of embeddings/features
        labels: numpy array of corresponding labels
        output_dir: directory to save plots and data
        model_name: name identifier for the model/experiment
        epoch: epoch number (optional, for training plots)
        save_data: whether to save the data as JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays if needed
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Generating PCA plot for {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
    
    # Perform PCA
    pca = PCA(n_components=min(50, embeddings.shape[1], len(embeddings)))  # Limit to reasonable number of components
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Print variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"PCA Variance Analysis:")
    print(f"  PC1 explains {variance_explained[0]:.3f} of variance")
    print(f"  PC2 explains {variance_explained[1]:.3f} of variance")
    print(f"  First 2 components explain {cumulative_variance[1]:.3f} of total variance")
    print(f"  First 10 components explain {cumulative_variance[min(9, len(cumulative_variance)-1)]:.3f} of total variance")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: PCA scatter plot
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    class_names = ["Contradiction", "Entailment", "Neutral"]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if label < len(class_names) else f'Class {label}'
        axes[0, 0].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          c=[colors[i]], label=label_name, alpha=0.6, s=20)
    
    axes[0, 0].set_xlabel(f'PC1 ({variance_explained[0]:.3f} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({variance_explained[1]:.3f} variance)')
    axes[0, 0].set_title(f'PCA Visualization - {model_name}' + (f' (Epoch {epoch})' if epoch else ''))
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
    class_labels = [class_names[label] if label < len(class_names) else f'Class {label}' for label in unique_labels]
    axes[1, 1].bar(class_labels, counts)
    axes[1, 1].set_xlabel('Class Label')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    epoch_suffix = f"_epoch_{epoch}" if epoch else ""
    plot_filename = f"pca_plot_{model_name}{epoch_suffix}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PCA plot → {plot_path}")
    
    # Save data as JSON if requested
    if save_data:
        pca_data = {
            'pca_embeddings': pca_embeddings[:, :10].tolist(),  # Save first 10 components to keep file size reasonable
            'labels': labels.tolist(),
            'variance_explained': variance_explained.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'total_samples': len(embeddings),
            'original_dimensions': embeddings.shape[1],
            'pca_components_saved': min(10, pca_embeddings.shape[1]),
            'model_name': model_name,
            'epoch': epoch,
            'class_names': class_names
        }
        
        data_filename = f"pca_data_{model_name}{epoch_suffix}.json"
        data_path = os.path.join(output_dir, data_filename)
        
        with open(data_path, 'w') as f:
            json.dump(pca_data, f, indent=2)
        
        print(f"Saved PCA data → {data_path}")
    
    return variance_explained, cumulative_variance


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


def generate_demographic_distribution_table(dataset, output_dir, dataset_name="dataset"):
    """Generate and save demographic distribution tables for each demographic field."""
    os.makedirs(output_dir, exist_ok=True)
    
    demographic_stats = {}
    
    for field in dataset.active_field_keys.keys():
        field_ids = dataset.demographic_ids[field]
        
        # Create reverse mapping from ID to label
        id_to_label = {v: k for k, v in dataset.vocab[field].items()}
        
        # Count occurrences
        field_counts = Counter(field_ids)
        
        # Create distribution table
        distribution = {}
        for id_val, count in field_counts.items():
            label = id_to_label.get(id_val, f"ID_{id_val}")
            distribution[label] = count
        
        demographic_stats[field] = {
            'distribution': distribution,
            'total_annotators': len(field_ids),
            'unique_values': len(field_counts)
        }
        
        print(f"\n{field.upper()} Distribution in {dataset_name}:")
        print(f"Total annotators: {len(field_ids)}")
        print(f"Unique values: {len(field_counts)}")
        for label, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(field_ids)) * 100
            print(f"  {label}: {