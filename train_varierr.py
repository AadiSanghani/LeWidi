import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

MAX_ANNOTATORS = 5  # Maximum number of annotators for VariErrNLI
NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL2IDX = {label: idx for idx, label in enumerate(NLI_LABELS)}

class VariErrNLI_Dataset(Dataset):
    """Dataset class for VariErrNLI dataset"""
    def __init__(self, data_path: str, max_length: int = 512, 
                 dataset_type: str = "varierrnli", task_type: str = "soft_label"):
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.task_type = task_type
        self.data = self.load_data(data_path)
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
        else:
            annotator_list = [a.strip() for a in item['annotators'].split(',')]
            annotations_dict = item['annotations']
            annotator_labels = []
            for ann in annotator_list:
                label_str = annotations_dict[ann]
                label_vec = [0, 0, 0]
                for label in label_str.split(','):
                    label = label.strip()
                    if label in NLI_LABEL2IDX:
                        label_vec[NLI_LABEL2IDX[label]] = 1
                annotator_labels.append(label_vec)
            # Truncate if too many annotators, pad if too few
            if len(annotator_labels) > MAX_ANNOTATORS:
                annotator_labels = annotator_labels[:MAX_ANNOTATORS]
            while len(annotator_labels) < MAX_ANNOTATORS:
                annotator_labels.append([-100, -100, -100])
            labels = torch.tensor(annotator_labels, dtype=torch.float)
            annotator_ids = []
        return {
            'text': text,
            'labels': labels,
            'annotator_ids': torch.tensor([])
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
                 num_annotators: Optional[int] = None, embedding_dim: int = 256):
        super(SBertForLeWiDi, self).__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.sbert = SentenceTransformer(model_name)
        self.dropout = nn.Dropout(0.1)
        emb_dim = self.sbert.get_sentence_embedding_dimension()
        if emb_dim is None:
            raise ValueError("SBert model did not return a valid embedding dimension.")
        emb_dim = int(emb_dim)
        self.embedding = nn.Linear(emb_dim, embedding_dim)
        if task_type == "soft_label":
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            if num_annotators is None:
                raise ValueError("num_annotators must be provided for perspectivist task_type.")
            self.classifier = nn.Linear(embedding_dim, num_classes * num_annotators)
    def forward(self, texts, labels=None, annotator_ids=None):
        embeddings = self.sbert.encode(texts, convert_to_tensor=True)
        x = self.dropout(embeddings)
        x = self.embedding(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.task_type == "soft_label":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
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
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                outputs = self.model(texts, labels, annotator_ids)
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
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                outputs = self.model(texts, labels, annotator_ids)
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
            config['train_path'], MAX_LENGTH, 'varierrnli', task_type
        )
        val_dataset = VariErrNLI_Dataset(
            config['val_path'], MAX_LENGTH, 'varierrnli', task_type
        )

        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = torch.stack([item['labels'] for item in batch])
            return {'text': texts, 'labels': labels}

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        num_annotators = MAX_ANNOTATORS if task_type == 'perspectivist' else None
        model = SBertForLeWiDi(
            MODEL_NAME, config['num_classes'], task_type, num_annotators
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

if __name__ == "__main__":
    main()
    # predict_example()  # Uncomment to run example inference 