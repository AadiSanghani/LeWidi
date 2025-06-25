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
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            p_hat = torch.softmax(logits, dim=-1)
            dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
            total_dist += dist.sum().item()
            n_examples += dist.numel()
    return total_dist / n_examples if n_examples else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = MPDataset(args.train_file, tokenizer, args.max_length)
    val_ds = MPDataset(args.val_file, tokenizer, args.max_length) if args.val_file else None

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            p_hat = torch.softmax(logits, dim=-1)
            # Fix: Don't double-average the L1 loss
            l1_loss = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1).mean()
            ce_loss = F.cross_entropy(logits, batch["labels"])
            loss = (1 - args.lambda_ce) * l1_loss + args.lambda_ce * ce_loss

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            epoch_loss += loss.item()
            if step % 100 == 0:
                tqdm.write(f"Epoch {epoch} step {step}: loss {epoch_loss/step:.4f}, lr {scheduler.get_last_lr()[0]:.2e}")

        # Validation
        if val_loader:
            val_dist = evaluate(model, val_loader, device)
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            if val_dist < best_metric:
                best_metric = val_dist
                save_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"New best model saved to {save_path}")

    # Save final model
    model.save_pretrained(os.path.join(args.output_dir, "last_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "last_model"))


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
    parser.add_argument("--lambda_ce", type=float, default=0.3, help="Weight for CE in mixed loss (0=L1 only)")

    args = parser.parse_args()
    train(args) 