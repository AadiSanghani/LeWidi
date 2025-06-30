import argparse
import json
import os
from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MPRegDataset(Dataset):
    """Dataset returning text encodings, annotator demographic id and probability label (float)."""

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        annot_meta: Optional[Dict] = None,
        location2idx: Optional[Dict[str, int]] = None,
        loc_field: str = "Country of residence",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.targets = []  # float p(irony)
        self.location_ids = []  # list[int] ids for all annotators

        if annot_meta is None or location2idx is None:
            raise ValueError("annot_meta and location2idx must be provided for MPRegDataset with location embedding")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            post = ex["text"].get("post", "")
            reply = ex["text"].get("reply", "")
            full_text = f"{post} {tokenizer.sep_token} {reply}".strip()

            # parse target soft label
            soft = ex.get("soft_label", {})
            if not soft or soft == "":
                continue
            p_irony = float(soft.get("1.0", 0.0))
            p0 = float(soft.get("0.0", 0.0))
            if p_irony + p0 == 0:  # malformed
                continue
            p_irony = p_irony / (p_irony + p0)

            # obtain *all* annotator IDs and map to location ids
            annotators_field = ex.get("annotators", "")
            ids_for_ex = []
            if annotators_field:
                for ann_tok in annotators_field.split(","):
                    ann_tok = ann_tok.strip()
                    if ann_tok.startswith("Ann"):
                        ann_id = ann_tok.replace("Ann", "")
                        loc_val = annot_meta.get(ann_id, {}).get(loc_field, "Unknown") or "Unknown"
                        loc_id = location2idx.get(loc_val, location2idx["Unknown"])
                        ids_for_ex.append(loc_id)

            if not ids_for_ex:
                ids_for_ex = [location2idx["Unknown"]]

            self.texts.append(full_text)
            self.targets.append(p_irony)
            self.location_ids.append(ids_for_ex)

    def __len__(self):
        return len(self.targets)

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
            "location_ids": torch.tensor(self.location_ids[idx], dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float),
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    targets = torch.stack([b["target"] for b in batch])

    # pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    # pad location id lists
    loc_lists = [b["location_ids"] for b in batch]
    max_l = max([len(t) for t in loc_lists])
    padded_loc = torch.zeros(len(batch), max_l, dtype=torch.long)
    mask = torch.zeros(len(batch), max_l, dtype=torch.bool)
    for i, t in enumerate(loc_lists):
        padded_loc[i, : len(t)] = t
        mask[i, : len(t)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "location_ids": padded_loc,
        "location_mask": mask,
        "target": targets,
    }


class RobertaProbRegressor(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        num_locations: int = 1,
        loc_emb_dim: int = 32,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.location_embed = nn.Embedding(num_locations, loc_emb_dim)

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + loc_emb_dim, 1),
        )

    def forward(self, input_ids, attention_mask, location_ids, location_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # [CLS]

        loc_emb_all = self.location_embed(location_ids)  # (B, L, D)
        mask = location_mask.unsqueeze(-1)  # (B, L, 1)
        loc_emb_all = loc_emb_all * mask  # zero out paddings
        lengths = mask.sum(dim=1).clamp(min=1)  # (B,1)
        loc_emb = loc_emb_all.sum(dim=1) / lengths  # mean

        combined = torch.cat([cls, loc_emb], dim=-1)
        logit = self.proj(combined).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob


def evaluate(model, dataloader, device):
    model.eval()
    total_dist = 0.0
    n_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            p_hat = model(batch["input_ids"], batch["attention_mask"], batch["location_ids"], batch["location_mask"])
            dist = 2.0 * torch.abs(p_hat - batch["target"])  # Manhattan for binary
            total_dist += dist.sum().item()
            n_examples += dist.numel()
    return total_dist / n_examples if n_examples else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- load annotator metadata and build location mapping ---
    with open(args.annot_meta_file, "r", encoding="utf-8") as f:
        annot_meta = json.load(f)

    # build mapping from location string to idx
    location2idx = {"Unknown": 0}
    for ann in annot_meta.values():
        loc_val = ann.get(args.loc_field, "Unknown") or "Unknown"
        if loc_val not in location2idx:
            location2idx[loc_val] = len(location2idx)

    print(f"Number of distinct locations (including Unknown): {len(location2idx)}")

    train_ds = MPRegDataset(
        args.train_file,
        tokenizer,
        args.max_length,
        annot_meta=annot_meta,
        location2idx=location2idx,
        loc_field=args.loc_field,
    )
    val_ds = (
        MPRegDataset(
            args.val_file,
            tokenizer,
            args.max_length,
            annot_meta=annot_meta,
            location2idx=location2idx,
            loc_field=args.loc_field,
        )
        if args.val_file
        else None
    )

    print(f"Train samples: {len(train_ds)}. Val samples: {len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_ds else None

    model = RobertaProbRegressor(
        args.model_name,
        dropout=args.dropout,
        num_locations=len(location2idx),
        loc_emb_dim=args.loc_emb_dim,
    )
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not n.startswith("location_embed")],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not n.startswith("location_embed")],
         "weight_decay": 0.0},
        {"params": model.location_embed.parameters(), "lr": args.lr * args.loc_lr_mult, "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped_params, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_metric = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            probs = model(batch["input_ids"], batch["attention_mask"], batch["location_ids"], batch["location_mask"])
            mae = torch.abs(probs - batch["target"]).mean()
            bce = F.binary_cross_entropy(probs, batch["target"])
            loss = (1 - args.lambda_bce) * mae + args.lambda_bce * bce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if step % 100 == 0:
                tqdm.write(f"Epoch {epoch} step {step}: loss={epoch_loss/step:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loader:
            val_dist = evaluate(model, val_loader, device)
            print(f"Validation Manhattan distance epoch {epoch}: {val_dist:.4f}")
            if val_dist < best_metric:
                best_metric = val_dist
                save_path = os.path.join(args.output_dir, "best_model")
                model.encoder.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                torch.save(model.state_dict(), os.path.join(save_path, "reg_head.pt"))
                print(f"New best model saved to {save_path}")

    # final save
    final_path = os.path.join(args.output_dir, "last_model")
    model.encoder.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    torch.save(model.state_dict(), os.path.join(final_path, "reg_head.pt"))
    print(f"Training complete. Best val distance: {best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression training for MP irony soft labels")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="outputs_mp_reg")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--annot_meta_file", type=str, default="dataset/MP/MP_annotators_meta.json")
    parser.add_argument("--loc_field", type=str, default="Country of residence")
    parser.add_argument("--loc_emb_dim", type=int, default=64)
    parser.add_argument("--loc_lr_mult", type=float, default=5.0)
    parser.add_argument("--lambda_bce", type=float, default=0.2)
    args = parser.parse_args()
    train(args) 