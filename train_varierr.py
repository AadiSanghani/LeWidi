import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional, Dict
import warnings
import json
warnings.filterwarnings('ignore')

MAX_ANNOTATORS = 5  # Maximum number of annotators for VariErrNLI
NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL2IDX = {label: idx for idx, label in enumerate(NLI_LABELS)}

class DemographicsEncoder:
    """Encode demographic information into embeddings"""
    def __init__(self, metadata_path: str, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.metadata = self.load_metadata(metadata_path)
        self.gender_encoder = nn.Embedding(3, embedding_dim // 4)  # Male, Female, Unknown
        self.age_encoder = nn.Embedding(10, embedding_dim // 4)    # Age buckets
        self.nationality_encoder = nn.Embedding(20, embedding_dim // 4)  # Nationality
        self.education_encoder = nn.Embedding(10, embedding_dim // 4)    # Education level
        
        # Create mappings
        self.gender_map = {"Male": 0, "Female": 1, "Unknown": 2}
        self.nationality_map = self.create_nationality_map()
        self.education_map = self.create_education_map()
        
    def load_metadata(self, metadata_path: str) -> Dict:
        with open(metadata_path, 'r') as f:
            content = f.read()
            # Handle trailing commas in JSON by removing them
            import re
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            return json.loads(content)
    
    def create_nationality_map(self) -> Dict[str, int]:
        nationalities = set()
        for ann_data in self.metadata.values():
            nationalities.add(ann_data.get("Nationality", "Unknown"))
        return {nat: idx for idx, nat in enumerate(sorted(nationalities))}
    
    def create_education_map(self) -> Dict[str, int]:
        educations = set()
        for ann_data in self.metadata.values():
            educations.add(ann_data.get("Education", "Unknown"))
        return {edu: idx for idx, edu in enumerate(sorted(educations))}
    
    def get_age_bucket(self, age_str: str) -> int:
        try:
            age = int(age_str)
            if age < 20: return 0
            elif age < 25: return 1
            elif age < 30: return 2
            elif age < 35: return 3
            elif age < 40: return 4
            elif age < 45: return 5
            elif age < 50: return 6
            elif age < 55: return 7
            elif age < 60: return 8
            else: return 9
        except:
            return 0
    
    def encode_annotator(self, annotator_id: str) -> torch.Tensor:
        if annotator_id not in self.metadata:
            # Return zero embedding for unknown annotators
            return torch.zeros(self.embedding_dim)
        
        ann_data = self.metadata[annotator_id]
        
        gender_idx = self.gender_map.get(ann_data.get("Gender", "Unknown"), 2)
        age_idx = self.get_age_bucket(ann_data.get("Age", "25"))
        nationality_idx = self.nationality_map.get(ann_data.get("Nationality", "Unknown"), 0)
        education_idx = self.education_map.get(ann_data.get("Education", "Unknown"), 0)
        
        gender_emb = self.gender_encoder(torch.tensor(gender_idx))
        age_emb = self.age_encoder(torch.tensor(age_idx))
        nationality_emb = self.nationality_encoder(torch.tensor(nationality_idx))
        education_emb = self.education_encoder(torch.tensor(education_idx))
        
        return torch.cat([gender_emb, age_emb, nationality_emb, education_emb])

class VariErrNLI_Dataset(Dataset):
    """Dataset class for VariErrNLI dataset with demographics support"""
    def __init__(self, data_path: str, max_length: int = 512, 
                 dataset_type: str = "varierrnli", task_type: str = "soft_label",
                 metadata_path: Optional[str] = None):
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.task_type = task_type
        self.data = self.load_data(data_path)
        self.demographics_encoder = None
        if metadata_path and os.path.exists(metadata_path):
            self.demographics_encoder = DemographicsEncoder(metadata_path)
    
    def load_data(self, data_path: str):
        import json
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .json.")
        return data
    
    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, list):
            item = self.data[idx]
        else:
            item = list(self.data.values())[idx]
        if self.dataset_type == "varierrnli":
            return self.process_varierrnli_item(item)
        elif self.dataset_type == "par":
            return self.process_par_item(item)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def process_par_item(self, item):
        # Not used in this script
        return {}
    
    def process_varierrnli_item(self, item):
        context = item['text']['context']
        statement = item['text']['statement']
        text = f"{context} [SEP] {statement}"
        
        if self.task_type == "soft_label":
            annotator_list = [a.strip() for a in item['annotators'].split(',')]
            annotations_dict = item['annotations']
            multi_hot_vectors = []
            for ann in annotator_list:
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
            labels = soft_label
            annotator_ids = []
            demographic_embeddings = []
        else:
            annotator_list = [a.strip() for a in item['annotators'].split(',')]
            annotations_dict = item['annotations']
            annotator_labels = []
            
            # Create demographic embeddings for each annotator
            demographic_embeddings = []
            for ann in annotator_list:
                label_str = annotations_dict[ann]
                label_vec = [0, 0, 0]
                for label in label_str.split(','):
                    label = label.strip()
                    if label in NLI_LABEL2IDX:
                        label_vec[NLI_LABEL2IDX[label]] = 1
                annotator_labels.append(label_vec)
                
                # Create demographic embedding
                if self.demographics_encoder:
                    dem_emb = self.demographics_encoder.encode_annotator(ann)
                else:
                    dem_emb = torch.zeros(64)  # Default embedding dimension
                demographic_embeddings.append(dem_emb)
            
            # Truncate if too many annotators, pad if too few
            if len(annotator_labels) > MAX_ANNOTATORS:
                annotator_labels = annotator_labels[:MAX_ANNOTATORS]
                demographic_embeddings = demographic_embeddings[:MAX_ANNOTATORS]
            while len(annotator_labels) < MAX_ANNOTATORS:
                annotator_labels.append([-100, -100, -100])
                demographic_embeddings.append(torch.zeros(64))
            
            labels = torch.tensor(annotator_labels, dtype=torch.float)
            demographic_embeddings = torch.stack(demographic_embeddings)
            annotator_ids = torch.tensor([hash(ann) % 1000 for ann in annotator_list[:MAX_ANNOTATORS]], dtype=torch.long)
            while len(annotator_ids) < MAX_ANNOTATORS:
                annotator_ids = torch.cat([annotator_ids, torch.tensor([-1])])
        
        return {
            'text': text,
            'labels': labels,
            'annotator_ids': annotator_ids if self.task_type == "perspectivist" else torch.tensor([]),
            'demographic_embeddings': demographic_embeddings if self.task_type == "perspectivist" else torch.tensor([])
        }
    
    def create_soft_label(self, annotations: List, scale_range: Tuple[int, int], categorical: bool = False):
        # Not used anymore, but kept for compatibility
        num_classes = scale_range[1] - scale_range[0]
        soft_label = torch.zeros(num_classes)
        for ann in annotations:
            try:
                ann = int(ann)
            except (ValueError, TypeError):
                continue
            if scale_range[0] <= ann < scale_range[1]:
                soft_label[ann - scale_range[0]] += 1
        if soft_label.sum() > 0:
            soft_label = soft_label / soft_label.sum()
        return soft_label

class SBertForLeWiDi(nn.Module):
    def __init__(self, model_name: str, num_classes: int, task_type: str = "soft_label",
                 num_annotators: Optional[int] = None, embedding_dim: int = 256,
                 use_demographics: bool = False):
        super(SBertForLeWiDi, self).__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.use_demographics = use_demographics
        
        self.sbert = SentenceTransformer(model_name)
        self.dropout = nn.Dropout(0.1)
        
        emb_dim = self.sbert.get_sentence_embedding_dimension()
        if emb_dim is None:
            raise ValueError("SBert model did not return a valid embedding dimension.")
        emb_dim = int(emb_dim)
        
        # Enhanced embedding layer that can incorporate demographics
        if use_demographics and task_type == "perspectivist":
            self.embedding = nn.Linear(emb_dim + 64, embedding_dim)  # +64 for demographics
        else:
            self.embedding = nn.Linear(emb_dim, embedding_dim)
        
        if task_type == "soft_label":
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            if num_annotators is None:
                raise ValueError("num_annotators must be provided for perspectivist task_type.")
            self.classifier = nn.Linear(embedding_dim, num_classes * num_annotators)
    
    def forward(self, texts, labels=None, annotator_ids=None, demographic_embeddings=None):
        # texts: list of strings
        embeddings = self.sbert.encode(texts, convert_to_tensor=True)
        
        if self.use_demographics and self.task_type == "perspectivist" and demographic_embeddings is not None:
            # For perspectivist with demographics, we need to handle each sample-annotator pair
            batch_size = embeddings.shape[0]
            num_annotators = self.num_annotators or MAX_ANNOTATORS
            
            # Expand text embeddings to match annotator dimension
            # embeddings: [batch_size, emb_dim] -> [batch_size * num_annotators, emb_dim]
            expanded_embeddings = embeddings.unsqueeze(1).expand(-1, num_annotators, -1)
            expanded_embeddings = expanded_embeddings.reshape(-1, embeddings.shape[-1])
            
            # Reshape demographic embeddings: [batch_size, num_annotators, dem_dim] -> [batch_size * num_annotators, dem_dim]
            demographic_flat = demographic_embeddings.reshape(-1, demographic_embeddings.shape[-1])
            
            # Concatenate text and demographic embeddings
            embeddings = torch.cat([expanded_embeddings, demographic_flat], dim=1)
        else:
            # For soft_label or no demographics, use original embeddings
            pass
        
        x = self.dropout(embeddings)
        x = self.embedding(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            if self.task_type == "soft_label":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                # For perspectivist, handle the expanded logits properly
                if self.use_demographics and demographic_embeddings is not None:
                    # Logits are already [batch_size * num_annotators, num_classes]
                    # Labels are [batch_size, num_annotators], need to flatten
                    mask = (labels != -100)
                    # Flatten mask to match logits shape
                    flat_mask = mask.view(-1)
                    # Apply mask to both logits and labels
                    valid_logits = logits[flat_mask, :]  # Select valid logits (keep all classes)
                    valid_labels = labels[mask].view(-1, self.num_classes)  # Select valid labels
                    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)
                    # Reshape logits back for output
                    logits = logits.view(-1, self.num_annotators, self.num_classes)
                else:
                    # Original perspectivist logic without demographics
                    logits = logits.view(-1, self.num_annotators, self.num_classes)
                    mask = (labels != -100)
                    valid_logits = logits[mask].view(-1, self.num_classes)
                    valid_labels = labels[mask].view(-1, self.num_classes)
                    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': torch.sigmoid(logits)
        }

class LeWiDiTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5,
              warmup_steps=0.1, save_path=None):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * warmup_steps) if warmup_steps < 1 else warmup_steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        self.model.train()
        best_val_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                texts = batch['text']
                labels = batch['labels'].to(self.device)
                annotator_ids = batch.get('annotator_ids', None)
                demographic_embeddings = batch.get('demographic_embeddings', None)
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                if demographic_embeddings is not None:
                    demographic_embeddings = demographic_embeddings.to(self.device)
                outputs = self.model(texts, labels, annotator_ids, demographic_embeddings)
                loss = outputs['loss']
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, '
                          f'Loss: {loss.item():.4f}')
            val_loss = self.evaluate(val_dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_dataloader):.4f}, '
                  f'Val Loss: {val_loss:.4f}')
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
                print(f'Model saved to {save_path}')
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                texts = batch['text']
                labels = batch['labels'].to(self.device)
                annotator_ids = batch.get('annotator_ids', None)
                demographic_embeddings = batch.get('demographic_embeddings', None)
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                if demographic_embeddings is not None:
                    demographic_embeddings = demographic_embeddings.to(self.device)
                outputs = self.model(texts, labels, annotator_ids, demographic_embeddings)
                total_loss += outputs['loss'].item()
        self.model.train()
        return total_loss / len(dataloader)
    def manhattan_distance_evaluation(self, dataloader):
        self.model.eval()
        total_distance = 0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                texts = batch['text']
                labels = batch['labels'].to(self.device)
                outputs = self.model(texts)
                predictions = outputs['predictions']
                manhattan_dist = torch.sum(torch.abs(predictions - labels), dim=1)
                total_distance += manhattan_dist.sum().item()
                total_samples += len(predictions)
        self.model.train()
        return total_distance / total_samples
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
    def load_model(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

def main():
    MODEL_NAME = 'all-MiniLM-L6-v2'  # SBert model
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    config = {
        'num_classes': 3,  # Entailment, Neutral, Contradiction
        'train_path': 'dataset/VariErrNLI/VariErrNLI_train.json',
        'val_path': 'dataset/VariErrNLI/VariErrNLI_dev.json',
        'test_path': 'models/VariErrNLI_test.json'
    }

    for task_type in ['soft_label', 'perspectivist']:
        print(f"\n{'='*50}")
        print(f"Training on VARIERRNLI dataset - Task Type: {task_type}")
        print(f"{'='*50}")

        train_dataset = VariErrNLI_Dataset(
            config['train_path'], MAX_LENGTH, 'varierrnli', task_type,
            metadata_path='dataset/VariErrNLI/VariErrNLI_annotators_meta.json'
        )
        val_dataset = VariErrNLI_Dataset(
            config['val_path'], MAX_LENGTH, 'varierrnli', task_type,
            metadata_path='dataset/VariErrNLI/VariErrNLI_annotators_meta.json'
        )

        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = torch.stack([item['labels'] for item in batch])
            if task_type == "perspectivist":
                annotator_ids = torch.stack([item['annotator_ids'] for item in batch])
                demographic_embeddings = torch.stack([item['demographic_embeddings'] for item in batch])
                return {'text': texts, 'labels': labels, 'annotator_ids': annotator_ids, 'demographic_embeddings': demographic_embeddings}
            else:
                return {'text': texts, 'labels': labels}

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        num_annotators = MAX_ANNOTATORS if task_type == 'perspectivist' else None
        model = SBertForLeWiDi(
            MODEL_NAME, config['num_classes'], task_type, num_annotators, use_demographics=task_type == 'perspectivist'
        )
        trainer = LeWiDiTrainer(model)
        save_path = f'models/varierrnli_{task_type}_sbert'
        trainer.train(
            train_dataloader, val_dataloader, EPOCHS, LEARNING_RATE,
            save_path=save_path
        )
        if task_type == 'soft_label':
            manhattan_score = trainer.manhattan_distance_evaluation(val_dataloader)
            print(f"Manhattan Distance Score: {manhattan_score:.4f}")
        print(f"Model saved to {save_path}")

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
    # Generate Task A (Soft Label) submission
    print("Generating Task A (Soft Label) submission...")
    soft_model_path = 'models/varierrnli_soft_label_sbert'
    model = SBertForLeWiDi('all-MiniLM-L6-v2', num_classes=num_bins, task_type='soft_label')
    trainer = LeWiDiTrainer(model)
    trainer.load_model(soft_model_path)
    model.to(device)
    model.eval()
    output_file = 'VariErrNLI_test_soft.tsv'
    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, ex in tqdm(enumerate(data), desc="Task A predictions"):
            context = ex['text']['context']
            statement = ex['text']['statement']
            text = f"{context} [SEP] {statement}"
            with torch.no_grad():
                outputs = model([text])
                probs = outputs['predictions'].squeeze(0).cpu()
            out_probs = probs.tolist()
            if len(out_probs) < num_bins:
                pad = [0.0] * (num_bins - len(out_probs))
                out_probs = pad + out_probs
            out_probs = [round(p, 10) for p in out_probs]
            drift = 1.0 - sum(out_probs)
            if abs(drift) > 1e-10:
                idx_max = max(range(len(out_probs)), key=out_probs.__getitem__)
                out_probs[idx_max] = round(out_probs[idx_max] + drift, 10)
            prob_str = ",".join(f"{p:.10f}" for p in out_probs)
            out_f.write(f"{idx}\t[{prob_str}]\n")
    print(f"Saved Task A submission file to {output_file}")
    # Generate Task B (Perspectivist) submission
    print("Generating Task B (Perspectivist) submission...")
    pe_model_path = 'models/varierrnli_perspectivist_sbert'
    model_pe = SBertForLeWiDi('all-MiniLM-L6-v2', num_classes=num_bins, task_type='perspectivist', 
                              num_annotators=MAX_ANNOTATORS, use_demographics=True)
    trainer_pe = LeWiDiTrainer(model_pe)
    trainer_pe.load_model(pe_model_path)
    model_pe.to(device)
    model_pe.eval()
    
    # Load demographics encoder for inference
    demographics_encoder = DemographicsEncoder('dataset/VariErrNLI/VariErrNLI_annotators_meta.json')
    
    output_file_pe = 'VariErrNLI_test_pe.tsv'
    with open(output_file_pe, "w", encoding="utf-8") as out_f:
        for idx, ex in tqdm(enumerate(data), desc="Task B predictions"):
            context = ex['text']['context']
            statement = ex['text']['statement']
            text = f"{context} [SEP] {statement}"
            ann_list = ex.get("annotators", "").split(",") if ex.get("annotators") else []
            
            # Create demographic embeddings for each annotator
            demographic_embeddings = []
            for ann in ann_list:
                dem_emb = demographics_encoder.encode_annotator(ann.strip())
                demographic_embeddings.append(dem_emb)
            
            # Pad to MAX_ANNOTATORS
            while len(demographic_embeddings) < MAX_ANNOTATORS:
                demographic_embeddings.append(torch.zeros(64))
            
            demographic_embeddings = torch.stack(demographic_embeddings).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model_pe([text], demographic_embeddings=demographic_embeddings)
                preds = outputs['predictions'].squeeze(0).cpu()  # [MAX_ANNOTATORS, num_bins]
            annotator_preds = []
            if isinstance(preds, torch.Tensor) and preds.ndim == 2:
                n_annotators = min(len(ann_list), preds.shape[0])
                for i in range(n_annotators):
                    label_idx = torch.argmax(preds[i]).item()
                    label_idx = int(label_idx)
                    if 0 <= label_idx < len(NLI_LABELS):
                        label = NLI_LABELS[label_idx]
                    else:
                        label = "UNK"
                    annotator_preds.append(label)
            elif isinstance(preds, torch.Tensor) and preds.ndim == 1:
                # fallback: treat as one annotator
                label_idx = torch.argmax(preds).item()
                label_idx = int(label_idx)
                if 0 <= label_idx < len(NLI_LABELS):
                    label = NLI_LABELS[label_idx]
                else:
                    label = "UNK"
                annotator_preds.append(label)
            preds_str = ", ".join(annotator_preds)
            out_f.write(f"{idx}\t[{preds_str}]\n")
    print(f"Saved Task B submission file to {output_file_pe}")
    print("To submit: zip -j res.zip", output_file, output_file_pe)

if __name__ == "__main__":
    main()
    # Uncomment to generate submission files after training:
    # generate_submission_files()
    # predict_example()  # Uncomment to run example inference 