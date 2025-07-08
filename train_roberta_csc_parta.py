import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch.nn as nn

# source myenv/bin/activate

class CSCDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 128, n_bins: int = 6):
        self.contexts = []
        self.responses = []
        self.labels = []
        self.dists = []
        self.raw_annotations = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_bins = n_bins

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            context = ex["text"]["context"]
            response = ex["text"]["response"]
            ratings = [int(v) - 1 for v in ex["annotations"].values() if v]
            if not ratings:
                continue

            self.contexts.append(context)
            self.responses.append(response)
            label = max(set(ratings), key=ratings.count)
            self.labels.append(label)

            dist = np.zeros(n_bins)
            for r in ratings:
                dist[r] += 1
            dist /= len(ratings)
            self.dists.append(dist.tolist())
            self.raw_annotations.append(ratings)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        response = self.responses[idx]
        enc_context = self.tokenizer(context, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        enc_response = self.tokenizer(response, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        item = {
            "context_input_ids": enc_context["input_ids"].squeeze(0),
            "context_attention_mask": enc_context["attention_mask"].squeeze(0),
            "response_input_ids": enc_response["input_ids"].squeeze(0),
            "response_attention_mask": enc_response["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            "annotations": torch.tensor(self.raw_annotations[idx], dtype=torch.long),
        }
        return item

    def __len__(self):
        return len(self.labels)



def collate_fn(batch):
    context_ids = pad_sequence([x["context_input_ids"] for x in batch], batch_first=True, padding_value=1)
    context_mask = pad_sequence([x["context_attention_mask"] for x in batch], batch_first=True, padding_value=0)
    response_ids = pad_sequence([x["response_input_ids"] for x in batch], batch_first=True, padding_value=1)
    response_mask = pad_sequence([x["response_attention_mask"] for x in batch], batch_first=True, padding_value=0)

    labels = torch.stack([x["labels"] for x in batch])
    dists = torch.stack([x["dist"] for x in batch])
    annotations = [x["annotations"] for x in batch]

    return {
        "context_input_ids": context_ids,
        "context_attention_mask": context_mask,
        "response_input_ids": response_ids,
        "response_attention_mask": response_mask,
        "labels": labels,
        "dist": dists,
        "annotations": annotations,
    }

def build_sampler(labels):
    """Returns a WeightedRandomSampler to alleviate class imbalance."""
    counts = Counter(labels)
    num_classes = len(counts)
    total = float(sum(counts.values()))
    class_weights = {c: total / (num_classes * cnt) for c, cnt in counts.items()}
    weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class RobertaFusionModel(nn.Module):
    def __init__(self, model_name="roberta-base", num_bins=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_bins)
        )

    def forward(self, context_input_ids, context_attention_mask, response_input_ids, response_attention_mask):
        context_output = self.encoder(input_ids=context_input_ids, attention_mask=context_attention_mask).last_hidden_state[:, 0, :]
        response_output = self.encoder(input_ids=response_input_ids, attention_mask=response_attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat([context_output, response_output], dim=1)
        logits = self.classifier(self.dropout(combined))
        return logits


def evaluate(model, dataloader, device):
    model.eval()
    total_dist = 0.0
    n_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            logits = model(
                batch["context_input_ids"],
                batch["context_attention_mask"],
                batch["response_input_ids"],
                batch["response_attention_mask"]
            )
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

    model = RobertaFusionModel(args.model_name, args.num_bins)
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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
                response_input_ids=batch["response_input_ids"],
                response_attention_mask=batch["response_attention_mask"]
            )
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
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                tokenizer.save_pretrained(save_path)
                print(f"Saved new best model to {save_path}")

        # Collect metrics for plotting
        train_loss_history.append(epoch_loss / step)
        if val_loader:
            val_dist_history.append(val_dist)
        lr_history.append(optimiser.param_groups[0]['lr'])

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_path)


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

    parser.add_argument("--max_length", type=int, default=128, help="Max token length for context/response")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (adjust if using roberta-large)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (use if batch_size is low)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup proportion of total steps")

    parser.add_argument("--balance", action="store_true", help="If set, use class-balancing sampler for training")
    parser.add_argument("--num_bins", type=int, default=6, help="Number of label bins (6 for ratings 1–6, shifted to 0–5)")

    parser.add_argument("--lr_decay_steps", type=int, default=1000, help="Optimizer steps between LR decays after warmup")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.9, help="LR decay factor after every decay step")

    parser.add_argument("--lambda_ce", type=float, default=0.05, help="Weight for cross-entropy loss (0 = only Wasserstein)")

    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (saves memory, speeds up training)")
    args = parser.parse_args()
    train(args) 