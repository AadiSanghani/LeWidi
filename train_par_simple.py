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


class ParSimpleModel(torch.nn.Module):
    """RoBERTa-Large model with SBERT embeddings for paraphrase detection (no demographic embeddings)."""
    
    def __init__(self, base_name: str, sbert_dim: int = 384, dropout_rate: float = 0.3, num_classes: int = 11):
        super().__init__()
        from transformers import AutoModel
        from sentence_transformers import SentenceTransformer

        # RoBERTa-Large as the main model
        if "roberta-large" not in base_name.lower():
            print(f"Warning: Expected roberta-large but got {base_name}. Using roberta-large as base model.")
            base_name = "roberta-large"
        
        self.roberta_model = AutoModel.from_pretrained(base_name)
        
        # SBERT model for additional embeddings (pretrained)
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Freeze SBERT model parameters to use as a pretrained feature extractor
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        
        # Get embedding dimension from RoBERTa-Large (should be 1024)
        roberta_dim = self.roberta_model.config.hidden_size
        print(f"RoBERTa-Large hidden size: {roberta_dim}")
        print(f"SBERT embedding dimension: {sbert_dim}")
        
        total_dim = roberta_dim + sbert_dim
        print(f"Total combined embedding dimension: {total_dim}")
        print(f"  - RoBERTa-Large: {roberta_dim}")
        print(f"  - SBERT: {sbert_dim}")

        # Classification head with dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(total_dim, num_classes)

    def forward(self, input_ids, attention_mask, texts):
        # Get RoBERTa-Large embeddings
        roberta_outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        roberta_pooled = roberta_outputs.pooler_output  # [CLS] token representation
        
        # Get SBERT embeddings for the texts (batch processing)
        sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, device=roberta_pooled.device)
        
        # Combine RoBERTa and SBERT embeddings
        combined = torch.cat([roberta_pooled, sbert_embeddings], dim=1)
        
        # Apply dropout and classification
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        return logits


class ParSimpleDataset(Dataset):
    """PyTorch dataset for the Paraphrase detection data without demographic information."""
    
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.dists = []
        self.labels = []
        
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

            # Add this example
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
        
        # Create the item dictionary
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            "texts": self.texts[idx],  # Include the text for SBERT
        }
        
        return item


def collate_fn(batch):
    """Pads a batch of variable-length encoded examples."""
    input_ids = [x["input_ids"] for x in batch]
    attn = [x["attention_mask"] for x in batch]
    labels = torch.stack([x["labels"] for x in batch])
    dists = torch.stack([x["dist"] for x in batch])
    texts = [x["texts"] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad token id = 1
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    result = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
        "texts": texts,
    }
    
    return result


def build_sampler(labels):
    """Returns a WeightedRandomSampler to alleviate class imbalance."""
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
                texts=batch["texts"]
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
        'pred_bias': float(pred_bias),
        'pred_std': float(pred_std),
        'target_std': float(target_std),
        'mean_error': float(np.mean(np.abs(errors)))
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
        mean_errors = [a['mean_error'] for a in analysis_history]
        pred_biases = [a['pred_bias'] for a in analysis_history]
        axes[1, 1].plot(epochs_range, mean_errors, marker='o', label='Mean Abs Error', color='purple')
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(epochs_range, pred_biases, marker='s', label='Prediction Bias', color='brown')
        axes[1, 1].set_title('Error and Bias Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Abs Error', color='purple')
        ax_twin.set_ylabel('Prediction Bias', color='brown')
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_ds = ParSimpleDataset(args.train_file, tokenizer, args.max_length)
    val_ds = ParSimpleDataset(args.val_file, tokenizer, args.max_length) if args.val_file else None

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

    model = ParSimpleModel(
        base_name=args.model_name,
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
    print(f"Warmup steps: {warmup_steps}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience} epochs")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"]
            )

            # Use cross-entropy loss with hard labels
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
                
            val_dist_history.append(val_dist)
            
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break
        
        train_loss_history.append(epoch_loss / step_count)
        lr_history.append(scheduler.get_last_lr()[0])

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_path)

    # Save training metadata for submission generation
    metadata = {
        "training_config": {
            "model_name": args.model_name,
            "sbert_dim": args.sbert_dim,
            "dropout_rate": args.dropout_rate,
            "num_classes": args.num_classes,
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

    print(f"Training completed!")
    print(f"Best validation distance: {best_metric:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-Large with SBERT embeddings on Paraphrase detection using cross-entropy loss (no demographic embeddings).")
    parser.add_argument("--train_file", type=str, default="dataset/Paraphrase/Paraphrase_train.json", help="Path to Paraphrase_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/Paraphrase/Paraphrase_dev.json", help="Path to Paraphrase_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="RoBERTa-Large model name")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_par_simple")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes (Likert scale -5 to 5)")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")

    args = parser.parse_args()
    train(args) 