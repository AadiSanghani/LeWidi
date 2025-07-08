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
from scipy.stats import wasserstein_distance


class CSCDemogModel(torch.nn.Module):
    """RoBERTa model with demographic embeddings for annotator-aware sarcasm detection."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 8, dropout_rate: float = 0.3):
        super().__init__()
        from transformers import AutoModel

        self.text_model = AutoModel.from_pretrained(base_name)
        
        # Create embeddings dynamically based on vocab_sizes
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        hidden_size = self.text_model.config.hidden_size
        num_demog_fields = len(vocab_sizes)
        total_dim = hidden_size + num_demog_fields * dem_dim
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(total_dim, 7)  # 7 classes for CSC (0-6)

    def forward(self, *, input_ids, attention_mask, **demographic_inputs):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

        # Get demographic embeddings dynamically
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        # Concatenate all vectors
        if demographic_vectors:
            concat = torch.cat([pooled] + demographic_vectors, dim=-1)
        else:
            concat = pooled
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class CSCDataset(Dataset):
    """Dataset for CSC sarcasm detection with demographic embeddings."""

    PAD_IDX = 0  # padding
    UNK_IDX = 1  # unknown / missing

    FIELD_KEYS = {
        "age": "Age",
        "gender": "Gender",
    }

    @staticmethod
    def get_age_bin(age):
        """Convert age to age bin."""
        if age is None or str(age).strip() == "" or str(age) in ["DATA_EXPIRED", "nan", "CONSENT_REVOKED"]:
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
        tokenizer: AutoTokenizer,
        annot_meta_path: str,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Use all available demographic fields from CSC data
        self.active_field_keys = self.FIELD_KEYS

        self.texts = []
        self.labels = []
        self.dists = []
        
        # Initialize storage for active fields only
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
                    # Handle CSC-specific missing/invalid values
                    if val == "" or val in ["DATA_EXPIRED", "nan", "CONSENT_REVOKED"]:
                        val = "<UNK>"
                    if val not in self.vocab[field]:
                        self.vocab[field][val] = len(self.vocab[field])

        self.vocab_sizes = {field: len(v) for field, v in self.vocab.items()}
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            context = ex["text"].get("context", "")
            response = ex["text"].get("response", "")
            full_text = f"{context} {tokenizer.sep_token} {response}".strip()

            soft_label = ex.get("soft_label", {})
            if not soft_label or soft_label == "":
                continue
            
            # Handle 7-class soft labels (0-6)
            dist = np.zeros(7, dtype=np.float32)
            for class_str, prob in soft_label.items():
                class_idx = int(class_str)
                if 0 <= class_idx <= 6:
                    dist[class_idx] = float(prob)
            
            if dist.sum() == 0:
                continue
            dist /= dist.sum()  # Normalize
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
                        # Handle CSC-specific missing/invalid values
                        if val == "" or val in ["DATA_EXPIRED", "nan", "CONSENT_REVOKED"]:
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
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    result = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
    }
    
    # Dynamically handle demographic fields
    demographic_keys = [k for k in batch[0].keys() 
                        if k.endswith("_ids") and k not in ["input_ids"]]
    for key in demographic_keys:
        tensors = [b[key] for b in batch]
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


def wasserstein_distance_numpy(p, q):
    """Compute Wasserstein distance between two discrete distributions."""
    # For discrete distributions, Wasserstein distance is the minimum cost to transform one to the other
    # For 1D case, it's the L1 distance between CDFs
    p_cdf = np.cumsum(p)
    q_cdf = np.cumsum(q)
    return np.sum(np.abs(p_cdf - q_cdf))


def evaluate(model, dataloader, device):
    """Return mean Wasserstein distance between predicted and true distributions."""
    model.eval()
    total_dist = 0.0
    n_examples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Prepare demographic inputs dynamically
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **demographic_inputs
            )
            p_hat = torch.softmax(logits, dim=-1)
            
            # Calculate Wasserstein distance for each example
            for i in range(p_hat.size(0)):
                pred_dist = p_hat[i].cpu().numpy()
                true_dist = batch["dist"][i].cpu().numpy()
                w_dist = wasserstein_distance_numpy(pred_dist, true_dist)
                total_dist += w_dist
                n_examples += 1
            
            # Store predictions and targets for analysis
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    return total_dist / n_examples if n_examples else 0.0, all_predictions, all_targets


def analyze_predictions(predictions, targets, epoch, output_dir):
    """Analyze prediction distributions and save plots."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate prediction statistics
    pred_means = np.mean(predictions, axis=0)
    target_means = np.mean(targets, axis=0)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Class distribution comparison
    class_labels = [f'Class {i}' for i in range(7)]
    x = np.arange(7)
    width = 0.35
    
    axes[0, 0].bar(x - width/2, target_means, width, label='True Distribution', alpha=0.7)
    axes[0, 0].bar(x + width/2, pred_means, width, label='Predicted Distribution', alpha=0.7)
    axes[0, 0].set_xlabel('Sarcasm Classes')
    axes[0, 0].set_ylabel('Mean Probability')
    axes[0, 0].set_title(f'Epoch {epoch}: Mean Class Distributions')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction vs Target for each class
    for class_idx in [0, 3, 6]:  # Show a few classes
        axes[0, 1].scatter(targets[:, class_idx], predictions[:, class_idx], 
                          alpha=0.5, s=1, label=f'Class {class_idx}')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Probability')
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].set_title(f'Epoch {epoch}: Prediction vs Target')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution for middle class (class 3)
    errors = predictions[:, 3] - targets[:, 3]
    axes[1, 0].hist(errors, bins=50, alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Prediction Error (Class 3)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Epoch {epoch}: Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class-wise mean absolute errors
    class_errors = np.mean(np.abs(predictions - targets), axis=0)
    axes[1, 1].bar(x, class_errors, alpha=0.7)
    axes[1, 1].set_xlabel('Sarcasm Classes')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title(f'Epoch {epoch}: Class-wise Errors')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_labels)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'pred_means': pred_means.tolist(),
        'target_means': target_means.tolist(),
        'class_errors': class_errors.tolist(),
        'mean_error': float(np.mean(class_errors))
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
        ax_comb_twin.plot(epochs_range, val_dist_history, marker='s', color='orange', label='Val Wasserstein Dist')
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
        axes[0, 1].set_title('Validation Wasserstein Distance')
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
        errors = [a['mean_error'] for a in analysis_history]
        axes[1, 1].plot(epochs_range, errors, marker='o', label='Mean Class Error', color='purple')
        axes[1, 1].set_title('Prediction Analysis')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics to JSON
    metrics = {
        'train_loss': [float(x) for x in train_loss_history],
        'val_distance': [float(x) for x in val_dist_history] if val_dist_history else [],
        'learning_rates': [float(x) for x in lr_history],
        'analysis': analysis_history,  # Already converted to native Python types in analyze_predictions
        'best_epoch': int(best_epoch) if val_dist_history else None,
        'best_metric': float(best_metric) if val_dist_history and best_metric != float("inf") else None
    }
    
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = CSCDataset(args.train_file, tokenizer, args.annot_meta, args.max_length)
    val_ds = CSCDataset(args.val_file, tokenizer, args.annot_meta, args.max_length) if args.val_file else None

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

    model = CSCDemogModel(
        base_name=args.model_name,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
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
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Prepare demographic inputs dynamically
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
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
            print(f"Validation Wasserstein distance after epoch {epoch}: {val_dist:.4f}")
            
            analysis = analyze_predictions(predictions, targets, epoch, args.output_dir)
            analysis_history.append(analysis)
            
            print(f"Epoch {epoch} Analysis:")
            print(f"  Mean class error: {analysis['mean_error']:.4f}")
            
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


def visualize_demog_embeddings(model, dataset: CSCDataset, output_dir: str):
    """Save 2-D PCA scatter plots of demographic embeddings."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    # Get field names and their display names (CSC only has Age and Gender)
    field_display_names = {
        "age": "Age",
        "gender": "Gender"
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
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on CSC sarcasm detection using cross-entropy loss with soft labels.")
    parser.add_argument("--train_file", type=str, default="dataset/CSC/CSC_train.json", help="Path to CSC_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/CSC/CSC_dev.json", help="Path to CSC_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="HF model name")
    parser.add_argument("--output_dir", type=str, default="runs_csc/outputs_csc")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--annot_meta", type=str, default="dataset/CSC/CSC_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of epochs to freeze layers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")

    args = parser.parse_args()
    train(args) 