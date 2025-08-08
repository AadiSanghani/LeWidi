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


class VariErrNLISimpleModel(torch.nn.Module):
    """RoBERTa-Large classifier for NLI (no SBERT, no demographics)."""
    
    def __init__(self, base_name: str, dropout_rate: float = 0.3):
        super().__init__()
        from transformers import AutoModel

        if "roberta-large" not in base_name.lower():
            print(f"Warning: Expected roberta-large but got {base_name}. Using roberta-large as base model.")
            base_name = "roberta-large"
        
        self.text_model = AutoModel.from_pretrained(base_name)
        
        hidden_size = self.text_model.config.hidden_size
        print(f"RoBERTa-Large hidden size: {hidden_size}")
        
        # Layer normalization and dropout for regularization
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_size, 3)  # NLI has 3 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class VariErrNLISimpleDataset(Dataset):
    """Dataset for VariErr NLI detection without demographic embeddings."""

    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.texts = []
        self.labels = []
        self.dists = []

        # Label mapping for NLI
        self.label_to_id = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

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

            # Add this example
            self.texts.append(full_text)
            self.dists.append(dist)
            self.labels.append(hard_label)

        print(f"Dataset created with {len(self.texts)} examples")

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
        
        return result


def collate_fn(batch):
    """Pads text for simple model without demographics."""

    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad token id = 1
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    result = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
    }

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
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
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
    
    im = axes[1, 1].imshow(confusion, interpolation='nearest', cmap='Blues')
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
        'pred_bias_contradiction': float(pred_bias_contradiction),
        'pred_bias_entailment': float(pred_bias_entailment),
        'pred_bias_neutral': float(pred_bias_neutral),
        'pred_std': pred_std.tolist(),
        'target_std': target_std.tolist(),
        'mean_error': float(np.mean(np.abs(errors))),
        'accuracy': float(accuracy)
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
        'analysis': [
            {
                'pred_bias_contradiction': float(a['pred_bias_contradiction']),
                'pred_bias_entailment': float(a['pred_bias_entailment']),
                'pred_bias_neutral': float(a['pred_bias_neutral']),
                'pred_std': a['pred_std'],
                'target_std': a['target_std'],
                'mean_error': float(a['mean_error']),
                'accuracy': float(a['accuracy'])
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = VariErrNLISimpleDataset(args.train_file, tokenizer, args.max_length)
    val_ds = VariErrNLISimpleDataset(args.val_file, tokenizer, args.max_length) if args.val_file else None

    print(f"Training samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")

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

    model = VariErrNLISimpleModel(
        base_name=args.model_name,
        dropout_rate=args.dropout_rate,
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
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
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

    # Save training metadata for submission generation
    metadata = {
        "training_config": {
            "model_name": args.model_name,
            "dropout_rate": args.dropout_rate,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio
        },
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "model_type": "simple_no_demographics"
    }
    
    with open(os.path.join(final_path, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save to best_model directory if it exists
    best_model_path = os.path.join(args.output_dir, "best_model")
    if os.path.exists(best_model_path):
        with open(os.path.join(best_model_path, "training_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

    # Skip plotting for simple script

    print(f"\nTraining completed. Best validation distance: {best_metric:.4f}")
    if val_dist_history:
        print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-Large with SBERT embeddings on VariErr NLI using cross-entropy loss with soft labels (no demographic embeddings).")
    parser.add_argument("--train_file", type=str, default="dataset/VariErrNLI/VariErrNLI_train.json", help="Path to VariErrNLI_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/VariErrNLI/VariErrNLI_dev.json", help="Path to VariErrNLI_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="HF model name")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_varier_nli_simple")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")


    args = parser.parse_args()
    train(args) 