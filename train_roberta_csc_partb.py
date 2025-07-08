# Task B: Annotator-Specific Sarcasm Prediction (Improved with SBERT fallback)
import argparse
import json
import os
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


class CSCAnnotatorDataset(Dataset):
    def __init__(self, data_path, annotator_meta_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.annotator2id = {}

        with open(data_path, "r") as f:
            data = list(json.load(f).values())

        annotator_set = set()
        for ex in data:
            for ann_id, label in ex["annotations"].items():
                annotator_set.add(ann_id.strip())

        self.annotator2id = {a: i for i, a in enumerate(sorted(annotator_set))}

        for ex in data:
            context = ex["text"]["context"]
            response = ex["text"]["response"]
            for ann_id, label in ex["annotations"].items():
                ann_id = ann_id.strip()
                if label is None:
                    continue
                self.samples.append({
                    "text": f"{context} [SEP] {response}",
                    "annotator_id": self.annotator2id[ann_id],
                    "label": float(label)  # keep original 1–6 scale
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(s["text"], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "annotator_id": torch.tensor(s["annotator_id"], dtype=torch.long),
            "label": torch.tensor(s["label"], dtype=torch.float),
            "text": s["text"]
        }


def collate_fn(batch):
    input_ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=1)
    attention_mask = pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0)
    annotator_ids = torch.stack([x["annotator_id"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    texts = [x["text"] for x in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "annotator_ids": annotator_ids,
        "labels": labels,
        "texts": texts
    }


class SBERTAnnotatorFusion(nn.Module):
    def __init__(self, sbert_model="sentence-transformers/all-MiniLM-L6-v2", ann_embed_dim=64, num_annotators=500):
        super().__init__()
        self.encoder = SentenceTransformer(sbert_model)
        self.ann_embedding = nn.Embedding(num_annotators, ann_embed_dim)
        self.linear1 = nn.Linear(self.encoder.get_sentence_embedding_dimension() + ann_embed_dim, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, texts, annotator_ids):
        embeddings = self.encoder.encode(texts, convert_to_tensor=True, device=annotator_ids.device)
        a = self.ann_embedding(annotator_ids)
        x = torch.cat([embeddings, a], dim=1)
        x = F.relu(self.linear1(x))
        output = self.linear2(x).squeeze(1)
        return torch.clamp(output, min=1.0, max=6.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data = CSCAnnotatorDataset(args.train_file, args.annotator_meta, tokenizer)
    val_data = CSCAnnotatorDataset(args.val_file, args.annotator_meta, tokenizer) if args.val_file else None

    num_annotators = len(train_data.annotator2id)
    model = SBERTAnnotatorFusion(num_annotators=num_annotators).to(device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_data else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    patience, wait = 2, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            annotator_ids = batch["annotator_ids"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]

            preds = model(texts, annotator_ids)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Train MAE: {total_loss / len(train_loader):.4f}")

        if val_loader:
            model.eval()
            total_mae = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    annotator_ids = batch["annotator_ids"].to(device)
                    labels = batch["labels"].to(device)
                    texts = batch["texts"]
                    preds = model(texts, annotator_ids)
                    mae = loss_fn(preds, labels)
                    total_mae += mae.item()
            val_mae = total_mae / len(val_loader)
            print(f"Val MAE: {val_mae:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--annotator_meta", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(args)
