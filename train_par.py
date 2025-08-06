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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


class ParDemogModel(torch.nn.Module):
    """RoBERTa-Large model with demographic embeddings and SBERT embeddings for annotator-aware paraphrase detection."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 8, sbert_dim: int = 384, dropout_rate: float = 0.3, num_classes: int = 11):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer

        # RoBERTa-Large as the main model (ensure we're using roberta-large)
        if "roberta-large" not in base_name.lower():
            print(f"Warning: Expected roberta-large but got {base_name}. Using roberta-large as base model.")
            base_name = "roberta-large"
        
        self.roberta_model = AutoModel.from_pretrained(base_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # SBERT model for additional embeddings (pretrained)
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Freeze SBERT model parameters to use as a pretrained feature extractor
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        
        # Create demographic embeddings dynamically based on vocab_sizes
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        # Get embedding dimension from RoBERTa-Large (should be 1024)
        roberta_dim = self.roberta_model.config.hidden_size
        print(f"RoBERTa-Large hidden size: {roberta_dim}")
        
        # SBERT dimension (from all-MiniLM-L6-v2)
        total_dem_dim = len(vocab_sizes) * dem_dim
        self.combined_dim = roberta_dim + sbert_dim + total_dem_dim
        
        print(f"Total combined embedding dimension: {self.combined_dim}")
        print(f"  - RoBERTa-Large: {roberta_dim}")
        print(f"  - SBERT: {sbert_dim}")
        print(f"  - Demographics ({len(vocab_sizes)} fields): {total_dem_dim}")

        # Classification head with dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.combined_dim, num_classes)

    def forward(self, input_ids, attention_mask, texts, **demographic_inputs):
        # Get RoBERTa-Large embeddings
        roberta_outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        roberta_pooled = roberta_outputs.pooler_output  # [CLS] token representation
        
        # Get SBERT embeddings for the texts (batch processing)
        sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, device=roberta_pooled.device)
        
        # Get demographic embeddings and concatenate them
        demographic_embeds = []
        for field, ids in demographic_inputs.items():
            if field.endswith("_ids"):
                field_name = field[:-4]  # Remove "_ids" suffix
                if field_name in self.demographic_embeddings:
                    embed = self.demographic_embeddings[field_name](ids)
                    demographic_embeds.append(embed)
        
        # Combine all embeddings
        combined_embeddings = [roberta_pooled, sbert_embeddings]
        combined_embeddings.extend(demographic_embeds)
        
        # Concatenate all embeddings
        combined = torch.cat(combined_embeddings, dim=1)
        
        # Apply dropout and classification
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        return logits


class ParDataset(Dataset):
    """PyTorch dataset for the Paraphrase detection data with demographic information."""
    
    def __init__(self, path: str, annot_meta_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.dists = []
        self.labels = []
        self.demographic_ids = defaultdict(list)  # Store demographic IDs for each field
        self.UNK_IDX = 0  # Reserve index 0 for unknown/padding
        
        # Define active demographic fields and their JSON keys
        self.active_field_keys = {
            "age": "age",
            "gender": "gender", 
            "education": "education_level",
            "english": "first_language_english",
            "country": "country_of_birth"
        }
        
        # Load annotator metadata
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            annot_meta = json.load(f)
        
        # Build vocabularies for each demographic field
        self.vocab = {}
        for field in self.active_field_keys:
            self.vocab[field] = {"<UNK>": self.UNK_IDX}
            
        # First pass: build vocabularies
        for ann_id, ann_data in annot_meta.items():
            for field, json_key in self.active_field_keys.items():
                if field == "age":
                    # Special handling for age - convert to age bin
                    age_bin = self.get_age_bin(ann_data.get(json_key))
                    if age_bin not in self.vocab[field]:
                        self.vocab[field][age_bin] = len(self.vocab[field])
                else:
                    val = str(ann_data.get(json_key, "")).strip()
                    if val == "":
                        val = "<UNK>"
                    if val not in self.vocab[field]:
                        self.vocab[field][val] = len(self.vocab[field])

        self.vocab_sizes = {field: len(v) for field, v in self.vocab.items()}
        
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
                        if val == "":
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
                    
    def get_age_bin(self, age_val):
        """Convert age to binned categories."""
        if age_val is None:
            return "<UNK>"
        try:
            age = int(age_val)
            if age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            elif age < 65:
                return "55-64"
            else:
                return "65+"
        except (ValueError, TypeError):
            return "<UNK>"

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
        
        # Add demographic information dynamically
        for field in self.active_field_keys:
            item[f"{field}_ids"] = torch.tensor(self.demographic_ids[field][idx], dtype=torch.long)
        
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
    
    # Add demographic fields dynamically
    demographic_fields = [k for k in batch[0].keys() if k.endswith("_ids") and k not in ["input_ids"]]
    for field in demographic_fields:
        result[field] = torch.stack([x[field] for x in batch])
    
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
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare demographic inputs dynamically (exclude texts)
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )
            
            # Get embeddings for PCA analysis by extracting the final layer features
            # We'll get the combined embeddings before the classifier
            with torch.no_grad():
                # Get RoBERTa embeddings
                roberta_outputs = model.roberta_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                roberta_pooled = roberta_outputs.pooler_output
                
                # Get SBERT embeddings
                sbert_embeddings = model.sbert_model.encode(batch["texts"], convert_to_tensor=True, device=roberta_pooled.device)
                
                # Get demographic embeddings
                demographic_embeds = []
                for field, ids in demographic_inputs.items():
                    if field.endswith("_ids"):
                        field_name = field[:-4]
                        if field_name in model.demographic_embeddings:
                            embed = model.demographic_embeddings[field_name](ids)
                            demographic_embeds.append(embed)
                
                # Combine embeddings (same as in forward pass but before dropout/classifier)
                combined_embeddings = [roberta_pooled, sbert_embeddings]
                combined_embeddings.extend(demographic_embeds)
                combined = torch.cat(combined_embeddings, dim=1)
                
                all_embeddings.extend(combined.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
            
            p_hat = torch.softmax(logits, dim=-1)
            dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
            total_dist += dist.sum().item()
            n_examples += dist.numel()
            
            # Store predictions and targets for analysis
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    return total_dist / n_examples if n_examples else 0.0, all_predictions, all_targets, all_embeddings, all_labels


def generate_pca_plot(embeddings, labels, output_dir, model_name="model", epoch=None, save_data=True):
    """
    Universal PCA plot generation function that creates and saves PCA visualizations.
    
    Args:
        embeddings: numpy array of embeddings/features
        labels: numpy array of corresponding labels
        output_dir: directory to save plots and data
        model_name: name identifier for the model/experiment
        epoch: epoch number (optional, for training plots)
        save_data: whether to save the data as JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays if needed
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Generating PCA plot for {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
    
    # Perform PCA
    pca = PCA(n_components=min(50, embeddings.shape[1], len(embeddings)))  # Limit to reasonable number of components
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Print variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"PCA Variance Analysis:")
    print(f"  PC1 explains {variance_explained[0]:.3f} of variance")
    print(f"  PC2 explains {variance_explained[1]:.3f} of variance")
    print(f"  First 2 components explain {cumulative_variance[1]:.3f} of total variance")
    print(f"  First 10 components explain {cumulative_variance[min(9, len(cumulative_variance)-1)]:.3f} of total variance")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: PCA scatter plot
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axes[0, 0].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
    
    axes[0, 0].set_xlabel(f'PC1 ({variance_explained[0]:.3f} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({variance_explained[1]:.3f} variance)')
    axes[0, 0].set_title(f'PCA Visualization - {model_name}' + (f' (Epoch {epoch})' if epoch else ''))
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Variance explained
    components_to_plot = min(20, len(variance_explained))
    axes[0, 1].bar(range(1, components_to_plot + 1), variance_explained[:components_to_plot])
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Variance Explained')
    axes[0, 1].set_title('Variance Explained by Each PC')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative variance
    axes[1, 0].plot(range(1, components_to_plot + 1), cumulative_variance[:components_to_plot], 'b-o')
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
    axes[1, 0].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Cumulative Variance Explained')
    axes[1, 0].set_title('Cumulative Variance Explained')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[1, 1].bar(unique_labels, counts)
    axes[1, 1].set_xlabel('Class Label')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    epoch_suffix = f"_epoch_{epoch}" if epoch else ""
    plot_filename = f"pca_plot_{model_name}{epoch_suffix}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PCA plot → {plot_path}")
    
    # Save data as JSON if requested
    if save_data:
        pca_data = {
            'pca_embeddings': pca_embeddings[:, :10].tolist(),  # Save first 10 components to keep file size reasonable
            'labels': labels.tolist(),
            'variance_explained': variance_explained.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'total_samples': len(embeddings),
            'original_dimensions': embeddings.shape[1],
            'pca_components_saved': min(10, pca_embeddings.shape[1]),
            'model_name': model_name,
            'epoch': epoch
        }
        
        data_filename = f"pca_data_{model_name}{epoch_suffix}.json"
        data_path = os.path.join(output_dir, data_filename)
        
        with open(data_path, 'w') as f:
            json.dump(pca_data, f, indent=2)
        
        print(f"Saved PCA data → {data_path}")
    
    return variance_explained, cumulative_variance


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
        'pred_bias': pred_bias,
        'pred_std': pred_std,
        'target_std': target_std,
        'mean_error': np.mean(np.abs(errors))
    }


def generate_demographic_distribution_table(dataset, output_dir, dataset_name="dataset"):
    """Generate and save demographic distribution tables for each demographic field."""
    os.makedirs(output_dir, exist_ok=True)
    
    demographic_stats = {}
    
    for field in dataset.active_field_keys.keys():
        field_ids = dataset.demographic_ids[field]
        
        # Create reverse mapping from ID to label
        id_to_label = {v: k for k, v in dataset.vocab[field].items()}
        
        # Count occurrences
        field_counts = Counter(field_ids)
        
        # Create distribution table
        distribution = {}
        for id_val, count in field_counts.items():
            label = id_to_label.get(id_val, f"ID_{id_val}")
            distribution[label] = count
        
        demographic_stats[field] = {
            'distribution': distribution,
            'total_annotators': len(field_ids),
            'unique_values': len(field_counts)
        }
        
        print(f"\n{field.upper()} Distribution in {dataset_name}:")
        print(f"Total annotators: {len(field_ids)}")
        print(f"Unique values: {len(field_counts)}")
        for label, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(field_ids)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Save to JSON
    output_file = os.path.join(output_dir, f"{dataset_name}_demographic_distributions.json")
    with open(output_file, 'w') as f:
        json.dump(demographic_stats, f, indent=2)
    
    print(f"\nSaved demographic distributions → {output_file}")
    
    return demographic_stats


def run_demographic_ablations(args, base_model_path=None):
    """Run ablation studies by adding demographics one at a time."""
    print("Running demographic ablation studies...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    train_ds = ParDataset(args.train_file, args.annot_meta, tokenizer, args.max_length)
    val_ds = ParDataset(args.val_file, args.annot_meta, tokenizer, args.max_length) if args.val_file else None
    
    # Generate demographic distribution tables
    generate_demographic_distribution_table(train_ds, args.output_dir, "train")
    if val_ds:
        generate_demographic_distribution_table(val_ds, args.output_dir, "validation")
    
    # Define ablation configurations
    demographic_fields = list(train_ds.active_field_keys.keys())
    ablation_configs = [
        {"name": "baseline", "fields": []},  # No demographics
    ]
    
    # Add one demographic at a time
    for field in demographic_fields:
        ablation_configs.append({
            "name": f"with_{field}",
            "fields": [field]
        })
    
    # Add all demographics
    ablation_configs.append({
        "name": "all_demographics", 
        "fields": demographic_fields
    })
    
    ablation_results = {}
    
    for config in ablation_configs:
        print(f"\n{'='*50}")
        print(f"Running ablation: {config['name']}")
        print(f"Using demographic fields: {config['fields']}")
        print(f"{'='*50}")
        
        # Create modified dataset with only selected demographic fields
        modified_train_ds = create_ablation_dataset(train_ds, config['fields'])
        modified_val_ds = create_ablation_dataset(val_ds, config['fields']) if val_ds else None
        
        # Create data loaders
        sampler = build_sampler(modified_train_ds.labels) if args.balance else None
        train_loader = DataLoader(
            modified_train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
        )
        val_loader = (
            DataLoader(modified_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) 
            if modified_val_ds else None
        )
        
        # Create model with appropriate vocab sizes
        model = ParDemogModel(
            base_name=args.model_name,
            vocab_sizes=modified_train_ds.vocab_sizes,
            dem_dim=args.dem_dim,
            sbert_dim=args.sbert_dim,
            dropout_rate=args.dropout_rate,
            num_classes=args.num_classes,
        )
        model.to(device)
        
        # Train model (shortened for ablation)
        ablation_epochs = min(3, args.epochs)  # Shorter training for ablations
        result = train_ablation_model(
            model, train_loader, val_loader, device, 
            ablation_epochs, args.lr, config['name'], 
            os.path.join(args.output_dir, "ablations")
        )
        
        ablation_results[config['name']] = result
        
        # Generate PCA plot for this ablation
        if val_loader:
            _, _, _, embeddings, labels = evaluate(model, val_loader, device)
            generate_pca_plot(
                embeddings, labels, 
                os.path.join(args.output_dir, "ablations"), 
                model_name=config['name']
            )
    
    # Save ablation results
    ablation_output_file = os.path.join(args.output_dir, "ablation_results.json")
    with open(ablation_output_file, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\nSaved ablation results → {ablation_output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    for name, result in ablation_results.items():
        print(f"{name:20s}: Val Distance = {result.get('val_distance', 'N/A'):.4f}")
    
    return ablation_results


def create_ablation_dataset(original_dataset, selected_fields):
    """Create a modified dataset with only selected demographic fields."""
    if original_dataset is None:
        return None
    
    # Create a copy of the dataset
    modified_dataset = ParDataset.__new__(ParDataset)
    
    # Copy basic attributes
    modified_dataset.tokenizer = original_dataset.tokenizer
    modified_dataset.max_length = original_dataset.max_length
    modified_dataset.texts = original_dataset.texts.copy()
    modified_dataset.dists = original_dataset.dists.copy()
    modified_dataset.labels = original_dataset.labels.copy()
    modified_dataset.UNK_IDX = original_dataset.UNK_IDX
    
    # Filter demographic fields
    modified_dataset.active_field_keys = {
        field: original_dataset.active_field_keys[field] 
        for field in selected_fields 
        if field in original_dataset.active_field_keys
    }
    
    # Copy relevant demographic data
    modified_dataset.demographic_ids = defaultdict(list)
    for field in selected_fields:
        if field in original_dataset.demographic_ids:
            modified_dataset.demographic_ids[field] = original_dataset.demographic_ids[field].copy()
    
    # Copy relevant vocabularies
    modified_dataset.vocab = {
        field: original_dataset.vocab[field].copy()
        for field in selected_fields
        if field in original_dataset.vocab
    }
    
    modified_dataset.vocab_sizes = {
        field: len(modified_dataset.vocab[field])
        for field in modified_dataset.vocab
    }
    
    return modified_dataset


def train_ablation_model(model, train_loader, val_loader, device, epochs, lr, model_name, output_dir):
    """Train a model for ablation study (simplified training loop)."""
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_val_distance = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ablation {model_name} Epoch {epoch}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare demographic inputs
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )
            
            loss = F.cross_entropy(logits, batch["labels"])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validate
        if val_loader:
            val_distance, _, _, _, _ = evaluate(model, val_loader, device)
            if val_distance < best_val_distance:
                best_val_distance = val_distance
            print(f"Ablation {model_name} Epoch {epoch}: Loss = {avg_loss:.4f}, Val Distance = {val_distance:.4f}")
        else:
            print(f"Ablation {model_name} Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return {
        'final_loss': avg_loss,
        'val_distance': best_val_distance if val_loader else None,
        'epochs_trained': epochs,
        'demographic_fields': list(model.demographic_embeddings.keys()) if hasattr(model, 'demographic_embeddings') else []
    }


def save_training_metrics(train_loss_history, val_dist_history, lr_history, analysis_history, best_epoch, best_metric, output_dir):
    """Save training metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    print(f"Saved training metrics → {os.path.join(output_dir, 'training_metrics.json')}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_ds = ParDataset(args.train_file, args.annot_meta, tokenizer, args.max_length)
    val_ds = ParDataset(args.val_file, args.annot_meta, tokenizer, args.max_length) if args.val_file else None

    print(f"Training samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")
    print(f"Using reduced demographic fields: {list(train_ds.active_field_keys.keys())}")
    print(f"Demographic vocabulary sizes: {train_ds.vocab_sizes}")

    # Generate demographic distribution tables
    generate_demographic_distribution_table(train_ds, args.output_dir, "train")
    if val_ds:
        generate_demographic_distribution_table(val_ds, args.output_dir, "validation")

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

    model = ParDemogModel(
        base_name=args.model_name,
        vocab_sizes=train_ds.vocab_sizes,
        dem_dim=args.dem_dim,
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
            
            # Prepare demographic inputs dynamically (exclude texts)
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
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
            val_dist, predictions, targets, embeddings, labels = evaluate(model, val_loader, device)
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            
            # Generate PCA plot for this epoch
            generate_pca_plot(
                embeddings, labels, args.output_dir, 
                model_name="paraphrase_model", epoch=epoch
            )
            
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
                
                # Generate PCA plot for best model
                generate_pca_plot(
                    embeddings, labels, save_path, 
                    model_name="best_paraphrase_model"
                )
                
                # Save training metadata and vocabulary for Codabench
                metadata = {
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "training_config": {
                        "lr": args.lr,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "model_name": args.model_name,
                        "patience": args.patience,
                        "dem_dim": args.dem_dim,
                        "sbert_dim": args.sbert_dim,
                        "dropout_rate": args.dropout_rate
                    },
                    "vocab_sizes": train_ds.vocab_sizes,
                    "active_fields": list(train_ds.active_field_keys.keys())
                }
                
                with open(os.path.join(save_path, "training_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Save vocabulary mappings
                vocab_data = {
                    "field_mappings": train_ds.active_field_keys,
                    "vocabularies": train_ds.vocab,
                    "vocab_sizes": train_ds.vocab_sizes
                }
                
                with open(os.path.join(save_path, "vocabulary.json"), 'w') as f:
                    json.dump(vocab_data, f, indent=2)
                
                print(f"Saved best model to {save_path}")
                
            else:
                epochs_no_improve += 1
                
            val_dist_history.append(val_dist)
            
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break
        
        train_loss_history.append(epoch_loss / step_count)
        lr_history.append(scheduler.get_last_lr()[0])

    # Save training metrics
    save_training_metrics(train_loss_history, val_dist_history, lr_history, 
                         analysis_history, best_epoch, best_metric, args.output_dir)

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
    
    # Generate final PCA plot
    if val_loader:
        _, _, _, embeddings, labels = evaluate(model, val_loader, device)
        generate_pca_plot(
            embeddings, labels, final_path, 
            model_name="final_paraphrase_model"
        )

    print(f"Training completed!")
    print(f"Best validation distance: {best_metric:.4f} at epoch {best_epoch}")
    
    # Run demographic ablations if requested
    if hasattr(args, 'run_ablations') and args.run_ablations:
        run_demographic_ablations(args)

    # Generate submission files if requested
    if args.generate_submission:
        print("Generating submission files...")
        from generate_submission_par import generate_submission
        best_model_path = os.path.join(args.output_dir, "best_model")
        generate_submission(best_model_path, args.task, args.output_dir)
        print("In Codabench, actual test data will be used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-Large with pretrained SBERT embeddings and demographic embeddings on Paraphrase detection using cross-entropy loss.")
    parser.add_argument("--train_file", type=str, default="dataset/Paraphrase/Paraphrase_train.json", help="Path to Paraphrase_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/Paraphrase/Paraphrase_dev.json", help="Path to Paraphrase_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="RoBERTa-Large model name (recommended)")
    parser.add_argument("--output_dir", type=str, default="runs/outputs_par")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs without improvement for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes (Likert scale -5 to 5)")
    parser.add_argument("--generate_submission", action="store_true", help="Generate submission files after training")
    parser.add_argument("--task", choices=["A", "B"], default="A", help="Task for submission generation: A (soft) or B (perspectivist)")
    parser.add_argument("--run_ablations", action="store_true", help="Run demographic ablation studies")

    args = parser.parse_args()
    train(args)