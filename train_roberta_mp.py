import argparse
import json
import os
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


class MPDataset(Dataset):
    """Dataset with demographic embeddings (country, employment, ethnicity)."""

    PAD_IDX = 0  # padding
    UNK_IDX = 1  # unknown / missing

    FIELD_KEYS = {
        "country": "Country of residence",
        "employment": "Employment status",
        "ethnicity": "Ethnicity simplified",
    }

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        annot_meta_path: str,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Containers
        self.texts = []
        self.labels = []
        self.dists = []
        self.country_ids = []
        self.employment_ids = []
        self.ethnicity_ids = []

        # -------------------- Build vocabularies --------------------
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            annot_meta = json.load(f)

        self.annot_meta = annot_meta

        # Initialize value→idx maps with PAD & UNK
        self.vocab = {
            field: {"<PAD>": self.PAD_IDX, "<UNK>": self.UNK_IDX}
            for field in self.FIELD_KEYS
        }

        # Populate vocabularies
        for ann_data in annot_meta.values():
            for field, json_key in self.FIELD_KEYS.items():
                val = str(ann_data.get(json_key, "")).strip()
                if val == "":
                    continue
                if val not in self.vocab[field]:
                    self.vocab[field][val] = len(self.vocab[field])

        # Store vocab sizes for later model init
        self.vocab_sizes = {field: len(v) for field, v in self.vocab.items()}

        # -------------------- Load main examples --------------------
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data.values():
            post = ex["text"].get("post", "")
            reply = ex["text"].get("reply", "")
            full_text = f"{post} {tokenizer.sep_token} {reply}".strip()

            soft_label = ex.get("soft_label", {})
            if not soft_label or soft_label == "":
                continue
            p0 = float(soft_label.get("0.0", 0.0))
            p1 = float(soft_label.get("1.0", 0.0))
            if p0 + p1 == 0:
                continue
            dist = np.array([p0, p1], dtype=np.float32)
            dist /= dist.sum()
            hard_label = int(np.argmax(dist))

            # --------------- Map annotator demographics ---------------
            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            if not ann_list:
                ann_list = []

            # For each field, collect indices list for this example
            field_lists = {field: [] for field in self.FIELD_KEYS}
            for ann_tag in ann_list:
                ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                meta = annot_meta.get(ann_num, {})
                for field, json_key in self.FIELD_KEYS.items():
                    val = str(meta.get(json_key, "")).strip()
                    idx = self.vocab[field].get(val, self.UNK_IDX)
                    field_lists[field].append(idx)

            # If no annotators or empty lists, use UNK
            for field in self.FIELD_KEYS:
                if not field_lists[field]:
                    field_lists[field] = [self.UNK_IDX]

            self.texts.append(full_text)
            self.dists.append(dist)
            self.labels.append(hard_label)
            self.country_ids.append(field_lists["country"])
            self.employment_ids.append(field_lists["employment"])
            self.ethnicity_ids.append(field_lists["ethnicity"])

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
            "country_ids": torch.tensor(self.country_ids[idx], dtype=torch.long),
            "employment_ids": torch.tensor(self.employment_ids[idx], dtype=torch.long),
            "ethnicity_ids": torch.tensor(self.ethnicity_ids[idx], dtype=torch.long),
        }


def _pad_list_tensors(list_of_1d):
    max_len = max(x.size(0) for x in list_of_1d)
    out = torch.zeros((len(list_of_1d), max_len), dtype=torch.long)
    for i, t in enumerate(list_of_1d):
        out[i, : t.size(0)] = t
    return out


def collate_fn(batch):
    """Pads text and demographic index lists."""

    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attn = pad_sequence(attn, batch_first=True, padding_value=0)

    country_tensor = _pad_list_tensors([b["country_ids"] for b in batch])
    employ_tensor = _pad_list_tensors([b["employment_ids"] for b in batch])
    ethnic_tensor = _pad_list_tensors([b["ethnicity_ids"] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "dist": dists,
        "country_ids": country_tensor,
        "employment_ids": employ_tensor,
        "ethnicity_ids": ethnic_tensor,
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
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                country_ids=batch["country_ids"],
                employment_ids=batch["employment_ids"],
                ethnicity_ids=batch["ethnicity_ids"],
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
    train_ds = MPDataset(args.train_file, tokenizer, args.annot_meta, args.max_length)
    val_ds = MPDataset(args.val_file, tokenizer, args.annot_meta, args.max_length) if args.val_file else None

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

    class MPDemogModel(torch.nn.Module):
        def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 16):
            super().__init__()
            from transformers import AutoModel

            self.text_model = AutoModel.from_pretrained(base_name)
            # Embeddings for each demographic field
            self.country_emb = torch.nn.Embedding(vocab_sizes["country"], dem_dim, padding_idx=MPDataset.PAD_IDX)
            self.emp_emb = torch.nn.Embedding(vocab_sizes["employment"], dem_dim, padding_idx=MPDataset.PAD_IDX)
            self.eth_emb = torch.nn.Embedding(vocab_sizes["ethnicity"], dem_dim, padding_idx=MPDataset.PAD_IDX)

            hidden_size = self.text_model.config.hidden_size
            total_dim = hidden_size + 3 * dem_dim
            self.norm = torch.nn.LayerNorm(total_dim)
            self.dropout = torch.nn.Dropout(0.5)  # Increased dropout
            self.classifier = torch.nn.Linear(total_dim, 2)

        def _avg_emb(self, emb_layer, idx_tensor):
            emb = emb_layer(idx_tensor)
            mask = (idx_tensor != MPDataset.PAD_IDX).unsqueeze(-1).float()
            summed = (emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts

        def forward(self, *, input_ids, attention_mask, country_ids, employment_ids, ethnicity_ids):
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]

            country_vec = self._avg_emb(self.country_emb, country_ids)
            emp_vec = self._avg_emb(self.emp_emb, employment_ids)
            eth_vec = self._avg_emb(self.eth_emb, ethnicity_ids)

            concat = torch.cat([pooled, country_vec, emp_vec, eth_vec], dim=-1)
            concat = self.norm(concat)
            concat = self.dropout(concat)
            logits = self.classifier(concat)
            return logits

    model = MPDemogModel(
        base_name=args.model_name,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
    )
    model.to(device)

    # ---------------- Layer Freezing Schedule ----------------
    frozen_layers = []
    if getattr(args, "freeze_layers", 0) > 0:
        for layer in model.text_model.encoder.layer[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        frozen_layers = list(range(args.freeze_layers))
        if frozen_layers:
            print(f"Frozen transformer layers: {frozen_layers} for first {args.freeze_epochs} epoch(s)")
    # ---------------------------------------------------------

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = AdamW(grouped_params, lr=args.lr)

    # Add proper learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
                   optimiser,
                   num_warmup_steps=warmup_steps,
                   num_training_steps=total_steps)

    best_metric = float("inf")
    epochs_no_improve = 0  # counter for early stopping
    best_epoch = 0  # track epoch of best validation metric
    os.makedirs(args.output_dir, exist_ok=True)

    # Track metrics for plotting
    train_loss_history = []
    val_dist_history = []
    lr_history = []
    analysis_history = []

    print(f"Total training steps: {total_steps}")
    print(f"Initial learning rate: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        # Unfreeze after the specified number of epochs
        if frozen_layers and epoch > args.freeze_epochs:
            for layer_idx in frozen_layers:
                for p in model.text_model.encoder.layer[layer_idx].parameters():
                    p.requires_grad = True
            print(f"Unfroze layers {frozen_layers} at start of epoch {epoch}")
            frozen_layers = []  # ensure we do not unfreeze again

        model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_kl_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"), 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                country_ids=batch["country_ids"],
                employment_ids=batch["employment_ids"],
                ethnicity_ids=batch["ethnicity_ids"],
            )

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
                best_epoch = epoch
                epochs_no_improve = 0
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                tokenizer.save_pretrained(save_path)
                print(f"New best model saved to {save_path}")
            else:
                epochs_no_improve += 1

        # Collect metrics for plotting
        train_loss_history.append(epoch_loss / step_count)
        if val_loader:
            val_dist_history.append(val_dist)
        lr_history.append(scheduler.get_last_lr()[0])

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save final model
    final_path = os.path.join(args.output_dir, "last_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_path)

    # Visualize demographic embeddings
    visualize_demog_embeddings(model, train_ds, args.output_dir)

    # Plot training curves
    epochs_range = list(range(1, len(train_loss_history) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (0,0) Combined training loss & validation distance for overfitting check
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

    # (0,1) Stand-alone validation distance plot with best epoch marker
    if val_dist_history:
        axes[0, 1].plot(epochs_range, val_dist_history, marker='o', color='orange')
        axes[0, 1].set_title('Validation Manhattan Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].scatter([best_epoch], [val_dist_history[best_epoch - 1]], color='red', s=100, zorder=5)
        axes[0, 1].text(best_epoch, val_dist_history[best_epoch - 1], f'  best={val_dist_history[best_epoch - 1]:.3f}', fontsize=10)

    # (1,0) Learning rate schedule
    axes[1, 0].plot(epochs_range, lr_history, marker='o', color='green')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1) Analysis metrics
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

    # Separate combined plot saved individually for quick reference
    if val_dist_history:
        fig2, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs_range, train_loss_history, marker='o', color='blue', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, val_dist_history, marker='s', color='orange', label='Val L1 Dist')
        ax2.set_ylabel('Validation Distance', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax1.set_title('Train Loss vs Val Distance')
        ax1.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.output_dir, 'loss_vs_val.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)

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

    # Optionally plot per-annotator error after training
    if args.plot_annotator_error:
        # Re-load best model for evaluation
        import torch
        from transformers import AutoTokenizer
        best_model_path = os.path.join(args.output_dir, "best_model", "pytorch_model.bin")
        if os.path.exists(best_model_path):
            model = MPDemogModel(
                base_name=args.model_name,
                vocab_sizes=train_ds.vocab_sizes,
                dem_dim=args.dem_dim,
            )
            model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            plot_per_annotator_error(
                model,
                tokenizer,
                args.val_file,
                args.annot_meta,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                args.output_dir,
            )
        else:
            print("Best model not found, skipping per-annotator error plot.")


def visualize_demog_embeddings(model, dataset: MPDataset, output_dir: str):
    """Save 2-D PCA scatter plots of demographic embeddings."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    emb_info = [
        ("country_emb", model.country_emb, dataset.vocab["country"], "Country of residence"),
        ("emp_emb", model.emp_emb, dataset.vocab["employment"], "Employment status"),
        ("eth_emb", model.eth_emb, dataset.vocab["ethnicity"], "Ethnicity simplified"),
    ]

    for name, emb_layer, vocab, title in emb_info:
        emb = emb_layer.weight.detach().cpu().numpy()
        if emb.shape[0] <= 2:
            continue
        # Exclude PAD/UNK
        emb = emb[2:]
        labels = list(vocab.keys())[2:]
        if len(labels) != emb.shape[0]:
            continue
        # PCA to 2D
        X = emb - emb.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = X.dot(Vt.T[:, :2])

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=40)
        for i, label in enumerate(labels):
            if i % max(1, len(labels)//30) == 0:  # avoid clutter
                plt.text(coords[i, 0], coords[i, 1], label, fontsize=8, alpha=0.7)
        plt.title(f"{title} Embeddings (PCA-2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{name}_pca.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {title} embedding visualisation → {fname}")


def plot_per_annotator_error(model, tokenizer, dev_file, annot_meta_file, device, output_dir):
    """Compute and plot per-annotator error rate for the Perspectivist Task (Task B)."""
    import json
    import os
    with open(dev_file, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(annot_meta_file, "r", encoding="utf-8") as f:
        annot_meta = json.load(f)

    annotator_errors = defaultdict(list)
    model.eval()
    for ex in dev_data.values():
        post = ex["text"].get("post", "")
        reply = ex["text"].get("reply", "")
        full_text = f"{post} {tokenizer.sep_token} {reply}".strip()
        ann_str = ex.get("annotators", "")
        ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
        if not ann_list:
            continue
        for ann_id in ann_list:
            meta = annot_meta.get(ann_id[3:] if ann_id.startswith("Ann") else ann_id, {})
            # Prepare demographic indices for this annotator
            def get_idx(field, json_key, vocab):
                val = str(meta.get(json_key, "")).strip()
                return vocab[field].get(val, MPDataset.UNK_IDX)
            # Use vocab from training set (assume model was trained with MPDataset)
            vocab = model.country_emb.weight.device  # hack to get vocab from model, but not available here
            # Instead, use a dummy MPDataset to get vocab
            # (Assume train_ds is available in global scope)
            # If not, skip this annotator
            try:
                country_idx = get_idx("country", "Country of residence", train_ds.vocab)
                emp_idx = get_idx("employment", "Employment status", train_ds.vocab)
                eth_idx = get_idx("ethnicity", "Ethnicity simplified", train_ds.vocab)
            except Exception:
                continue
            # Prepare tensors
            enc = tokenizer(full_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            country_ids = torch.tensor([[country_idx]], dtype=torch.long, device=device)
            emp_ids = torch.tensor([[emp_idx]], dtype=torch.long, device=device)
            eth_ids = torch.tensor([[eth_idx]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    country_ids=country_ids,
                    employment_ids=emp_ids,
                    ethnicity_ids=eth_ids,
                )
                pred_label = int(torch.argmax(logits, dim=-1).item())
            # True label for this annotator
            true_label = int(ex["annotations"][ann_id])
            error = int(pred_label != true_label)
            annotator_errors[ann_id].append(error)
    # Compute error rate per annotator
    annotator_error_rate = {ann: sum(errs)/len(errs) for ann, errs in annotator_errors.items() if errs}
    # Plot
    plt.figure(figsize=(16, 4))
    plt.bar(annotator_error_rate.keys(), annotator_error_rate.values())
    plt.xlabel("Annotator ID")
    plt.ylabel("Error Rate")
    plt.title("Per-Annotator Error Rate (Task B)")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "per_annotator_error.png"), dpi=150)
    plt.close()
    print(f"Saved per-annotator error plot to {os.path.join(output_dir, 'per_annotator_error.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on MP irony detection (binary soft labels).")
    parser.add_argument("--train_file", type=str, default="dataset/MP/MP_train.json", help="Path to MP_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/MP/MP_dev.json", help="Path to MP_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="HF model name")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_mp")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--lambda_kl", type=float, default=0.3, help="Weight for KL in mixed loss (0=L1 only)")
    parser.add_argument("--annot_meta", type=str, default="dataset/MP/MP_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=16, help="Dimension of each demographic embedding")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of epochs to freeze layers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--plot_annotator_error", action="store_true", help="Plot per-annotator error after training (Task B)")

    args = parser.parse_args()
    train(args) 