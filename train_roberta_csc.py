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
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F


class CSCDataset(Dataset):
    """PyTorch dataset for the CSC sarcasm-detection data"""

    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 128, n_bins: int = 7):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        self.dists = []
        self.n_bins = n_bins

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            # Build input text.
            context = ex["text"].get("context", "")
            response = ex["text"].get("response", "")
            if context:
                full_text = f"{context} {tokenizer.sep_token} {response}"
            else:
                full_text = response

            # Extract annotator ratings
            ratings = [int(v) for v in ex["annotations"].values() if v]
            if not ratings:
                # skip items with no labels
                continue

            # Use majority vote to assign a hard label (needed for the sampler only)
            vote = Counter(ratings).most_common(1)[0][0]
            label = vote - 1  # shift to 0-based

            self.texts.append(full_text)
            self.labels.append(label)
            # build soft distribution for Wasserstein training
            dist = np.zeros(self.n_bins, dtype=np.float32)
            for r in ratings:
                idx = r if self.n_bins == 7 else r - 1  
                if idx < self.n_bins:
                    dist[idx] += 1.0
            dist /= dist.sum()
            self.dists.append(dist)

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
        }
        return item


def collate_fn(batch):
    """Pads a batch of variable-length encoded examples."""
    input_ids = [x["input_ids"] for x in batch]
    attn = [x["attention_mask"] for x in batch]
    labels = torch.stack([x["labels"] for x in batch]) if "labels" in batch[0] else None
    dists = torch.stack([x["dist"] for x in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad token id = 1
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
    }


def build_sampler(labels):
    """Returns a WeightedRandomSampler to alleviate class imbalance."""
    counts = Counter(labels)
    num_classes = len(counts)
    total = float(sum(counts.values()))
    class_weights = {c: total / (num_classes * cnt) for c, cnt in counts.items()}
    weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model, dataloader, device):
    """Return mean 1-D Wasserstein distance over the dataloader."""
    model.eval()
    total_dist = 0.0
    n_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            p_hat = torch.softmax(logits, dim=-1)
            cdf_hat = torch.cumsum(p_hat, dim=-1)
            cdf_true = torch.cumsum(batch["dist"], dim=-1)
            dist = torch.sum(torch.abs(cdf_hat - cdf_true), dim=-1)
            total_dist += dist.sum().item()
            n_examples += dist.numel()

    return total_dist / n_examples if n_examples else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = CSCDataset(args.train_file, tokenizer, args.max_length, args.num_bins)
    val_ds = CSCDataset(args.val_file, tokenizer, args.max_length, args.num_bins) if args.val_file else None

    sampler = build_sampler(train_ds.labels) if args.balance else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_fn,
    )

    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        if val_ds
        else None
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_bins)
    model.to(device)

    # Optimiser & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimiser = AdamW(grouped_params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    decay_steps = args.lr_decay_steps  # already in optimizer steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        exponent = (current_step - warmup_steps) // max(1, decay_steps)
        return args.lr_decay_gamma ** exponent

    scheduler = LambdaLR(optimiser, lr_lambda)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    # Track metrics for plotting
    train_loss_history = []
    val_dist_history = []
    lr_history = []  # learning rate per epoch

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in prog:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            # Wasserstein loss between predicted and true distributions
            p_hat = torch.softmax(logits, dim=-1)
            cdf_hat = torch.cumsum(p_hat, dim=-1)
            cdf_true = torch.cumsum(batch["dist"], dim=-1)
            wass_loss = torch.sum(torch.abs(cdf_hat - cdf_true), dim=-1).mean()

            ce_loss = F.cross_entropy(logits, batch["labels"])
            loss = (1 - args.lambda_ce) * wass_loss + args.lambda_ce * ce_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            epoch_loss += loss.item()
            step += 1
            prog.set_postfix(loss=f"{epoch_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if val_loader:
            val_dist = evaluate(model, val_loader, device)
            print(f"\nValidation Wasserstein distance after epoch {epoch}: {val_dist:.4f}")
            improve = val_dist < best_acc or best_acc == 0.0
            if improve:
                best_acc = val_dist
                save_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved new best model to {save_path}")

        # Collect metrics for plotting
        train_loss_history.append(epoch_loss / step)
        if val_loader:
            val_dist_history.append(val_dist)
        lr_history.append(optimiser.param_groups[0]['lr'])

    # Save final checkpoint
    model.save_pretrained(os.path.join(args.output_dir, "last_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "last_model"))

    # Plot metrics with explicit epoch indices
    epochs_range = list(range(1, len(train_loss_history) + 1))
    num_cols = 3 if val_loader else 2
    plt.figure(figsize=(6 * num_cols, 4))
    plt.subplot(1, num_cols, 1)
    plt.plot(epochs_range, train_loss_history, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs_range[:: max(1, len(epochs_range) // 10)])  # at most 10 x-ticks

    col_idx = 2  # current subplot index

    if val_loader and val_dist_history:
        plt.subplot(1, num_cols, col_idx)
        plt.plot(epochs_range, val_dist_history, marker="o")
        plt.title("Validation Wasserstein Distance")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.xticks(epochs_range[:: max(1, len(epochs_range) // 10)])

        # Annotate best epoch
        best_epoch = int(np.argmin(val_dist_history)) + 1
        best_val = min(val_dist_history)
        plt.scatter([best_epoch], [best_val], color="red")
        plt.text(best_epoch, best_val, f"  best={best_val:.3f}@{best_epoch}")
        col_idx += 1

    # Plot LR curve
    plt.subplot(1, num_cols, col_idx)
    plt.plot(epochs_range, lr_history, marker="o")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.xticks(epochs_range[:: max(1, len(epochs_range) // 10)])

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_metrics.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on CSC sarcasm detection (6-way).")
    parser.add_argument("--train_file", type=str, required=True, help="Path to CSC_train.json")
    parser.add_argument("--val_file", type=str, default=None, help="Path to CSC_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="HF model name")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--balance", action="store_true", help="If set, use WeightedRandomSampler to mitigate class imbalance.")
    parser.add_argument("--num_bins", type=int, default=7, help="Number of label bins (e.g. 7 if ratings 0-6, 6 if 1-6).")
    parser.add_argument("--lr_decay_steps", type=int, default=1000, help="Optimizer steps between LR decays after warmup.")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.9, help="Gamma for learning rate decay")
    parser.add_argument("--lambda_ce", type=float, default=0.3, help="Weight for cross-entropy in mixed loss (0 means pure Wasserstein).")

    args = parser.parse_args()
    train(args) 