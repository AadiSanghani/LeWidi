import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MPDataset(Dataset):
    """PyTorch dataset for the MP irony-detection data (binary soft labels)."""

    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []  # concatenated post + sep + reply
        self.labels = []  # hard labels (arg-max) used for CE and optional sampler
        self.dists = []   # 2-dim soft label distributions

        # Load the JSON file (outer keys are string IDs)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            # Build input text (post + SEP + reply)
            post = ex["text"].get("post", "")
            reply = ex["text"].get("reply", "")
            full_text = f"{post} {tokenizer.sep_token} {reply}".strip()

            # Extract soft label distribution; some files (e.g., test) have empty string
            soft_label = ex.get("soft_label", {})
            if not soft_label or soft_label == "":
                # Skip examples with no labels (test set will be handled separately)
                continue

            p0 = float(soft_label.get("0.0", 0.0))
            p1 = float(soft_label.get("1.0", 0.0))
            if p0 + p1 == 0:  # guard against empty/malformed
                continue
            dist = np.array([p0, p1], dtype=np.float32)
            dist /= dist.sum()

            hard_label = int(np.argmax(dist))

            self.texts.append(full_text)
            self.dists.append(dist)
            self.labels.append(hard_label)

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
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
        }


def collate_fn(batch):
    """Pads batch and stacks tensors."""
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad id = 1
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
    }


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
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
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
    
    # Calculate prediction bias
    pred_bias = np.mean(predictions[:, 1] - targets[:, 1])  # bias toward class 1
    pred_std = np.std(predictions[:, 1])
    target_std = np.std(targets[:, 1])
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Prediction vs Target scatter
    axes[0, 0].scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=1)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('True P(irony)')
    axes[0, 0].set_ylabel('Predicted P(irony)')
    axes[0, 0].set_title(f'Epoch {epoch}: Prediction vs Target')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction distribution
    axes[0, 1].hist(predictions[:, 1], bins=50, alpha=0.7, label='Predictions')
    axes[0, 1].hist(targets[:, 1], bins=50, alpha=0.7, label='Targets')
    axes[0, 1].set_xlabel('P(irony)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Epoch {epoch}: Distribution Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = predictions[:, 1] - targets[:, 1]
    axes[1, 0].hist(errors, bins=50, alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Epoch {epoch}: Error Distribution (bias={pred_bias:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error vs Target
    axes[1, 1].scatter(targets[:, 1], errors, alpha=0.5, s=1)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('True P(irony)')
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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = MPDataset(args.train_file, tokenizer, args.max_length)
    val_ds = MPDataset(args.val_file, tokenizer, args.max_length) if args.val_file else None

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

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    # Optimiser with proper parameter grouping
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = AdamW(grouped_params, lr=args.lr)

    # Add proper learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_metric = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    # Track metrics for plotting
    train_loss_history = []
    val_dist_history = []
    lr_history = []
    analysis_history = []

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Initial learning rate: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_kl_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            p_hat = torch.softmax(logits, dim=-1)
            l1_loss = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1).mean()
            kl_loss = F.kl_div(torch.log(p_hat + 1e-12), batch["dist"], reduction="batchmean")
            loss = (1 - args.lambda_kl) * l1_loss + args.lambda_kl * kl_loss

            loss.backward()
            
            # Gradient clipping
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

        # Validation
        if val_loader:
            val_dist, predictions, targets = evaluate(model, val_loader, device)
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            
            # Analyze predictions
            analysis = analyze_predictions(predictions, targets, epoch, args.output_dir)
            analysis_history.append(analysis)
            
            print(f"Epoch {epoch} Analysis:")
            print(f"  Prediction bias: {analysis['pred_bias']:.4f}")
            print(f"  Mean absolute error: {analysis['mean_error']:.4f}")
            print(f"  Prediction std: {analysis['pred_std']:.4f}")
            print(f"  Target std: {analysis['target_std']:.4f}")
            
            if val_dist < best_metric:
                best_metric = val_dist
                save_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"New best model saved to {save_path}")

        # Collect metrics for plotting
        train_loss_history.append(epoch_loss / step_count)
        if val_loader:
            val_dist_history.append(val_dist)
        lr_history.append(scheduler.get_last_lr()[0])

    # Save final model
    model.save_pretrained(os.path.join(args.output_dir, "last_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "last_model"))

    # Plot training curves
    epochs_range = list(range(1, len(train_loss_history) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(epochs_range, train_loss_history, marker='o')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation distance
    if val_dist_history:
        axes[0, 1].plot(epochs_range, val_dist_history, marker='o', color='orange')
        axes[0, 1].set_title('Validation Manhattan Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(val_dist_history) + 1
        best_val = min(val_dist_history)
        axes[0, 1].scatter([best_epoch], [best_val], color='red', s=100, zorder=5)
        axes[0, 1].text(best_epoch, best_val, f'  best={best_val:.3f}', fontsize=10)
    
    # Learning rate
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
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
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
    
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed. Best validation distance: {best_metric:.4f}")
    if val_dist_history:
        print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on MP irony detection (binary soft labels).")
    parser.add_argument("--train_file", type=str, required=True, help="Path to MP_train.json")
    parser.add_argument("--val_file", type=str, default=None, help="Path to MP_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="HF model name")
    parser.add_argument("--output_dir", type=str, default="outputs_mp")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Fraction of total steps for warmup")
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--lambda_kl", type=float, default=0.3, help="Weight for KL in mixed loss (0=L1 only)")

    args = parser.parse_args()
    train(args) 