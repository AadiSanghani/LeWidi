import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import os
from typing import List, Tuple, Optional, Dict
import warnings
import json
from tqdm import tqdm
from torch.optim import AdamW
warnings.filterwarnings('ignore')

MAX_ANNOTATORS = 5  # Maximum number of annotators for VariErrNLI
NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL2IDX = {label: idx for idx, label in enumerate(NLI_LABELS)}

class VariErrNLIDemogModel(nn.Module):
    """RoBERTa-Large model with demographic embeddings and SBERT embeddings for annotator-aware NLI."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, sbert_dim: int = 384, 
                 dropout_rate: float = 0.3, num_classes: int = 3, task_type: str = "soft_label"):
        super().__init__()
        
        # RoBERTa-Large as the main model
        self.roberta_model = AutoModel.from_pretrained(base_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # SBERT model for additional embeddings
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Freeze SBERT model parameters
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        self.task_type = task_type
        
        # Create embeddings dynamically based on vocab_sizes
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        # Get embedding dimension from RoBERTa
        roberta_dim = self.roberta_model.config.hidden_size  # Usually 1024 for RoBERTa-Large
        
        # SBERT dimension (from all-MiniLM-L6-v2)
        self.sbert_dim = sbert_dim
        
        num_demog_fields = len(vocab_sizes)
        total_dim = roberta_dim + sbert_dim + num_demog_fields * dem_dim
        
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        if task_type == "soft_label":
            self.classifier = torch.nn.Linear(total_dim, num_classes)
        else:
            # For perspectivist, we need to handle multiple annotators
            self.classifier = torch.nn.Linear(total_dim, num_classes)

    def forward(self, *, input_ids, attention_mask, texts, **demographic_inputs):
        # Get RoBERTa embeddings
        roberta_outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        roberta_embeddings = roberta_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get SBERT embeddings
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True)
        
        # Get demographic embeddings dynamically
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        # Concatenate all vectors: RoBERTa + SBERT + demographics
        all_vectors = [roberta_embeddings, sbert_embeddings] + demographic_vectors
        concat = torch.cat(all_vectors, dim=-1)
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits

class VariErrNLI_Dataset(Dataset):
    """Dataset class for VariErrNLI dataset with demographics support"""
    
    PAD_IDX = 0  # padding
    UNK_IDX = 1  # unknown / missing

    FIELD_KEYS = {
        "age": "Age",  # Will be binned
        "gender": "Gender",
        "nationality": "Nationality",
        "education": "Education",
    }

    @staticmethod
    def get_age_bin(age):
        """Convert age to age bin."""
        if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
            return "<UNK>"
        try:
            age = float(age)
            if age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            else:
                return "55+"
        except (ValueError, TypeError):
            return "<UNK>"

    def __init__(self, data_path: str, annot_meta_path: str, max_length: int = 512, 
                 dataset_type: str = "varierrnli", task_type: str = "soft_label",
                 tokenizer=None):
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.data = self.load_data(data_path)
        
        # Load annotator metadata
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            self.annot_meta = json.load(f)

        self.texts = []
        self.labels = []
        self.dists = []
        
        # Initialize storage for demographic fields
        self.demographic_ids = {field: [] for field in self.FIELD_KEYS}

        self.vocab = {
            field: {"<PAD>": self.PAD_IDX, "<UNK>": self.UNK_IDX}
            for field in self.FIELD_KEYS
        }

        # Build vocabulary from all annotators
        for ann_data in self.annot_meta.values():
            for field, json_key in self.FIELD_KEYS.items():
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
        
        # Process data
        self.process_data()
        
        print(f"Dataset created with {len(self.texts)} examples")
        print(f"Demographic vocabulary sizes: {self.vocab_sizes}")
    
    def load_data(self, data_path: str):
        import json
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .json.")
        return data
    
    def process_data(self):
        """Process the data and create examples for each annotator"""
        if isinstance(self.data, dict):
            data_items = list(self.data.values())
        else:
            data_items = self.data
            
        for ex in data_items:
            context = ex['text']['context']
            statement = ex['text']['statement']
            text = f"{context} [SEP] {statement}"
            
            # Get annotator list
            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            
            if self.task_type == "soft_label":
                # Create soft label from all annotators
                annotator_list = [a.strip() for a in ex.get('annotators', '').split(',')]
                annotations_dict = ex.get('annotations', {})
                multi_hot_vectors = []
                
                for ann in annotator_list:
                    if ann in annotations_dict:
                        label_str = annotations_dict[ann]
                        label_vec = [0, 0, 0]
                        for label in label_str.split(','):
                            label = label.strip()
                            if label in NLI_LABEL2IDX:
                                label_vec[NLI_LABEL2IDX[label]] = 1
                        multi_hot_vectors.append(label_vec)
                
                if multi_hot_vectors:
                    soft_label = torch.tensor(multi_hot_vectors, dtype=torch.float).mean(dim=0)
                else:
                    soft_label = torch.zeros(3, dtype=torch.float)
                
                # Create one example with soft label
                self.texts.append(text)
                self.dists.append(soft_label.numpy())
                self.labels.append(torch.argmax(soft_label).item())
                
                # Use UNK demographic info for soft label task
                for field in self.FIELD_KEYS:
                    self.demographic_ids[field].append(self.UNK_IDX)
                    
            else:
                # Perspectivist task - create separate examples for each annotator
                annotations_dict = ex.get('annotations', {})
                
                for ann_tag in ann_list:
                    ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                    meta = self.annot_meta.get(ann_num, {})
                    
                    # Get demographic info for this specific annotator
                    annotator_demog_ids = {}
                    for field, json_key in self.FIELD_KEYS.items():
                        if field == "age":
                            age_bin = self.get_age_bin(meta.get(json_key))
                            idx = self.vocab[field].get(age_bin, self.UNK_IDX)
                        else:
                            val = str(meta.get(json_key, "")).strip()
                            if val == "":
                                val = "<UNK>"
                            idx = self.vocab[field].get(val, self.UNK_IDX)
                        annotator_demog_ids[field] = idx
                    
                    # Get label for this annotator
                    if ann_tag in annotations_dict:
                        label_str = annotations_dict[ann_tag]
                        label_vec = [0, 0, 0]
                        for label in label_str.split(','):
                            label = label.strip()
                            if label in NLI_LABEL2IDX:
                                label_vec[NLI_LABEL2IDX[label]] = 1
                        
                        # Add this annotator's example
                        self.texts.append(text)
                        self.dists.append(torch.tensor(label_vec, dtype=torch.float).numpy())
                        self.labels.append(torch.argmax(torch.tensor(label_vec)).item())
                        
                        # Store demographic IDs for this annotator
                        for field in self.FIELD_KEYS:
                            self.demographic_ids[field].append(annotator_demog_ids[field])
                
                # If no annotators, create one example with UNK demographic info
                if not ann_list:
                    self.texts.append(text)
                    self.dists.append(torch.zeros(3, dtype=torch.float).numpy())
                    self.labels.append(0)  # Default to contradiction
                    
                    # Store UNK demographic IDs
                    for field in self.FIELD_KEYS:
                        self.demographic_ids[field].append(self.UNK_IDX)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Tokenize text if tokenizer is available
        if self.tokenizer is not None:
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
                "texts": self.texts[idx],  # Keep original text for SBERT
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            }
        else:
            result = {
                "texts": self.texts[idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "dist": torch.tensor(self.dists[idx], dtype=torch.float),
            }
        
        # Add demographic fields dynamically
        for field in self.FIELD_KEYS:
            result[f"{field}_ids"] = torch.tensor(self.demographic_ids[field][idx], dtype=torch.long)
        
        return result

def collate_fn(batch):
    """Handles tokenized inputs and demographic values for VariErrNLI dataset."""
    texts = [b["texts"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    dists = torch.stack([b["dist"] for b in batch])

    result = {
        "texts": texts,
        "labels": labels,
        "dist": dists,
    }
    
    # Handle tokenized inputs if present
    if "input_ids" in batch[0]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)  # RoBERTa pad token id = 1
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
    
    # Dynamically handle demographic fields
    demographic_keys = [k for k in batch[0].keys() 
                        if k.endswith("_ids") and k not in ["input_ids"]]
    for key in demographic_keys:
        tensors = [b[key] for b in batch]
        stacked = torch.stack(tensors)
        result[key] = stacked

    return result

def evaluate(model, dataloader, device, task_type="soft_label"):
    """Return mean Manhattan (L1) distance between predicted and true distributions."""
    model.eval()
    total_dist = 0.0
    n_examples = 0
    all_predictions = []
    all_targets = []
    
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
            
            if task_type == "soft_label":
                p_hat = torch.sigmoid(logits)  # Use sigmoid for binary classification
                dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
            else:
                p_hat = torch.softmax(logits, dim=-1)  # Use softmax for multi-class
                dist = torch.sum(torch.abs(p_hat - batch["dist"]), dim=-1)
                
            total_dist += dist.sum().item()
            n_examples += dist.numel()
            
            # Store predictions and targets for analysis
            all_predictions.extend(p_hat.cpu().numpy())
            all_targets.extend(batch["dist"].cpu().numpy())
    
    return total_dist / n_examples if n_examples else 0.0, all_predictions, all_targets

def main():
    # Configuration
    MAX_LENGTH = 512
    BATCH_SIZE = 8  # Reduced batch size for RoBERTa-large
    EPOCHS = 15
    LEARNING_RATE = 1e-5  # Lower learning rate for RoBERTa-large

    config = {
        'num_classes': 3,  # Entailment, Neutral, Contradiction
        'train_path': 'dataset/VariErrNLI/VariErrNLI_train.json',
        'val_path': 'dataset/VariErrNLI/VariErrNLI_dev.json',
        'annot_meta_path': 'dataset/VariErrNLI/VariErrNLI_annotators_meta.json',
        'model_name': 'roberta-large'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    for task_type in ['soft_label', 'perspectivist']:
        print(f"\n{'='*50}")
        print(f"Training on VARIERRNLI dataset - Task Type: {task_type}")
        print(f"{'='*50}")

        train_dataset = VariErrNLI_Dataset(
            config['train_path'], 
            config['annot_meta_path'],
            MAX_LENGTH, 
            'varierrnli', 
            task_type,
            tokenizer
        )
        val_dataset = VariErrNLI_Dataset(
            config['val_path'], 
            config['annot_meta_path'],
            MAX_LENGTH, 
            'varierrnli', 
            task_type,
            tokenizer
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        model = VariErrNLIDemogModel(
            base_name=config['model_name'],
            vocab_sizes=train_dataset.vocab_sizes,
            dem_dim=32,
            sbert_dim=384,
            dropout_rate=0.3,
            num_classes=config['num_classes'],
            task_type=task_type
        )
        model.to(device)

        # Optimizer setup
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(grouped_params, lr=LEARNING_RATE)

        total_steps = len(train_dataloader) * EPOCHS
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
                       optimizer,
                       num_warmup_steps=warmup_steps,
                       num_training_steps=total_steps)

        best_metric = float("inf")
        epochs_no_improve = 0
        best_epoch = 0
        save_path = f'models/varierrnli_{task_type}_roberta_large'
        os.makedirs(save_path, exist_ok=True)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Demographic vocabulary sizes: {train_dataset.vocab_sizes}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            step_count = 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{EPOCHS}"), 1):
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
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                step_count += 1
                
                if step % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / step_count
                    tqdm.write(f"Epoch {epoch} step {step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

            # Validation
            val_dist, predictions, targets = evaluate(model, val_dataloader, device, task_type)
            print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
            
            if val_dist < best_metric:
                best_metric = val_dist
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                print(f"New best model saved to {save_path} (metric: {best_metric:.4f})")
            else:
                epochs_no_improve += 1

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Training Loss: {epoch_loss / step_count:.4f}")
            print(f"  Validation Distance: {val_dist:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  Epochs without improvement: {epochs_no_improve}")

            if epochs_no_improve >= 3:  # Early stopping patience
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation distance: {best_metric:.4f} at epoch {best_epoch}")
                break

        print(f"\nTraining completed for {task_type}. Best validation distance: {best_metric:.4f}")
        print(f"Best epoch: {best_epoch}")

def generate_submission_files():
    """Generate Codabench submission files for VariErrNLI dataset using trained models."""
    import json
    from tqdm import tqdm
    
    # Configuration for VariErrNLI dataset
    test_file = 'dataset/VariErrNLI/VariErrNLI_dev.json'  # Use dev for testing, change to test when available
    max_length = 512
    num_bins = 3  # Contradiction, Entailment, Neutral
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    
    # Generate Task A (Soft Label) submission
    print("Generating Task A (Soft Label) submission...")
    soft_model_path = 'models/varierrnli_soft_label_roberta_large/pytorch_model.bin'
    
    # Load the saved state dict to get the actual vocabulary sizes
    saved_state = torch.load(soft_model_path, map_location='cpu')
    
    # Extract vocabulary sizes from the saved model
    vocab_sizes = {}
    for key in saved_state.keys():
        if key.startswith('demographic_embeddings.') and key.endswith('.weight'):
            field = key.split('.')[1]
            vocab_size = saved_state[key].shape[0]
            vocab_sizes[field] = vocab_size
    
    print(f"Detected vocabulary sizes from saved model: {vocab_sizes}")
    
    # Create model with the correct vocabulary sizes
    model = VariErrNLIDemogModel(
        base_name='roberta-large',
        vocab_sizes=vocab_sizes,
        dem_dim=32,
        sbert_dim=384,
        dropout_rate=0.3,
        num_classes=num_bins,
        task_type='soft_label'
    )
    model.load_state_dict(saved_state)
    model.to(device)
    model.eval()
    
    output_file = 'VariErrNLI_test_soft_roberta.tsv'
    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, ex in tqdm(enumerate(data), desc="Task A predictions"):
            context = ex['text']['context']
            statement = ex['text']['statement']
            text = f"{context} [SEP] {statement}"
            
            # Tokenize
            enc = tokenizer(
                text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            
            # Prepare demographic inputs (UNK for test)
            demographic_inputs = {}
            for field in vocab_sizes.keys():
                demographic_inputs[f"{field}_ids"] = torch.tensor([1])  # UNK
            
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                    texts=[text],
                    **{k: v.to(device) for k, v in demographic_inputs.items()}
                )
                probs = torch.sigmoid(logits).squeeze(0).cpu()
            
            out_probs = probs.tolist()
            if len(out_probs) < num_bins:
                pad = [0.0] * (num_bins - len(out_probs))
                out_probs = pad + out_probs
            out_probs = [round(p, 10) for p in out_probs]
            drift = 1.0 - sum(out_probs)
            if abs(drift) > 1e-10:
                # Find the index with maximum probability
                max_idx = 0
                max_prob = out_probs[0]
                for i in range(1, len(out_probs)):
                    if out_probs[i] > max_prob:
                        max_prob = out_probs[i]
                        max_idx = i
                out_probs[max_idx] = round(out_probs[max_idx] + drift, 10)
            prob_str = ",".join(f"{p:.10f}" for p in out_probs)
            out_f.write(f"{idx}\t[{prob_str}]\n")
    print(f"Saved Task A submission file to {output_file}")
    
    # Generate Task B (Perspectivist) submission
    print("Generating Task B (Perspectivist) submission...")
    pe_model_path = 'models/varierrnli_perspectivist_roberta_large/pytorch_model.bin'
    
    # Load the saved state dict to get the actual vocabulary sizes
    saved_state_pe = torch.load(pe_model_path, map_location='cpu')
    
    # Extract vocabulary sizes from the saved model
    vocab_sizes_pe = {}
    for key in saved_state_pe.keys():
        if key.startswith('demographic_embeddings.') and key.endswith('.weight'):
            field = key.split('.')[1]
            vocab_size = saved_state_pe[key].shape[0]
            vocab_sizes_pe[field] = vocab_size
    
    print(f"Detected vocabulary sizes from saved perspectivist model: {vocab_sizes_pe}")
    
    model_pe = VariErrNLIDemogModel(
        base_name='roberta-large',
        vocab_sizes=vocab_sizes_pe,
        dem_dim=32,
        sbert_dim=384,
        dropout_rate=0.3,
        num_classes=num_bins,
        task_type='perspectivist'
    )
    model_pe.load_state_dict(saved_state_pe)
    model_pe.to(device)
    model_pe.eval()
    
    output_file_pe = 'VariErrNLI_test_pe_roberta.tsv'
    with open(output_file_pe, "w", encoding="utf-8") as out_f:
        for idx, ex in tqdm(enumerate(data), desc="Task B predictions"):
            context = ex['text']['context']
            statement = ex['text']['statement']
            text = f"{context} [SEP] {statement}"
            ann_list = ex.get("annotators", "").split(",") if ex.get("annotators") else []
            
            annotator_preds = []
            for ann in ann_list:
                # Use UNK demographic info for test
                demographic_inputs = {}
                for field in vocab_sizes_pe.keys():
                    demographic_inputs[f"{field}_ids"] = torch.tensor([1])  # UNK
                
                # Tokenize
                enc = tokenizer(
                    text,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
                with torch.no_grad():
                    logits = model_pe(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                        texts=[text],
                        **{k: v.to(device) for k, v in demographic_inputs.items()}
                    )
                    preds = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                
                label_idx = torch.argmax(preds).item()
                if 0 <= label_idx < len(NLI_LABELS):
                    label = NLI_LABELS[label_idx]
                else:
                    label = "UNK"
                annotator_preds.append(label)
            
            preds_str = ", ".join(annotator_preds)
            out_f.write(f"{idx}\t[{preds_str}]\n")
    print(f"Saved Task B submission file to {output_file_pe}")
    print("To submit: zip -j res_roberta.zip", output_file, output_file_pe)

if __name__ == "__main__":
    main()
    # Uncomment to generate submission files after training:
    # generate_submission_files() 