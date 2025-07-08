import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional
import warnings
import csv
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MAX_ANNOTATORS = 5  # Maximum number of annotators for Paraphrase dataset

class Par_Dataset(Dataset):
    """Dataset class for Par dataset"""
    def __init__(self, data_path: str, max_length: int = 512, 
                 dataset_type: str = "par", task_type: str = "soft_label"):
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
        if self.dataset_type == "par":
            return self.process_par_item(item)
        elif self.dataset_type == "varierrnli":
            return self.process_varierrnli_item(item)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    def process_par_item(self, item):
        question1 = item['text']['Question1']
        question2 = item['text']['Question2']
        if self.task_type == "soft_label":
            soft_label_dict = item['soft_label']
            soft_label = [float(soft_label_dict[str(i)]) for i in range(-5, 6)]
            soft_label = torch.tensor(soft_label, dtype=torch.float)
            labels = soft_label
            annotator_ids = []
        else:
            annotator_list = [a.strip() for a in item['annotators'].split(',')]
            annotations_dict = item['annotations']
            # Build annotation list, pad to MAX_ANNOTATORS with -100
            annotations = [float(annotations_dict[ann]) for ann in annotator_list]
            while len(annotations) < MAX_ANNOTATORS:
                annotations.append(-100)
            labels = torch.tensor(annotations, dtype=torch.float)
            annotator_ids = []
        text = f"{question1} [SEP] {question2}"
        return {
            'text': text,
            'labels': labels,
            'annotator_ids': torch.tensor([])
        }
    def process_varierrnli_item(self, item):
        # Not used in this script
        return {}
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
        # texts: list of strings
        embeddings = self.sbert.encode(texts, convert_to_tensor=True)
        x = self.dropout(embeddings)
        x = self.embedding(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.task_type == "soft_label":
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, labels, reduction='batchmean')
            else:
                logits = logits.view(-1, self.num_annotators, self.num_classes)
                labels_shifted = labels.clone()
                mask = (labels != -100)
                labels_shifted[mask] = labels[mask] + 5
                labels_shifted = labels_shifted.long()
                loss = F.cross_entropy(
                    logits.view(-1, self.num_classes),
                    labels_shifted.view(-1),
                    ignore_index=-100
                )
        return {
            'loss': loss,
            'logits': logits,
            'predictions': F.softmax(logits, dim=-1) if self.task_type == "soft_label" else logits
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
    EPOCHS = 5  # Set to 5 epochs for real training
    LEARNING_RATE = 2e-5

    config = {
        'num_classes': 11,  # Likert scale -5 to 5
        'train_path': 'dataset/Paraphrase/Paraphrase_train.json',
        'val_path': 'dataset/Paraphrase/Paraphrase_dev.json',
        'test_path': 'models/Paraphrase_test.json'
    }

    for task_type in ['soft_label', 'perspectivist']:
        print(f"\n{'='*50}")
        print(f"Training on PARAPHRASE dataset - Task Type: {task_type}")
        print(f"{'='*50}")

        train_dataset = Par_Dataset(
            config['train_path'], MAX_LENGTH, 'par', task_type
        )
        val_dataset = Par_Dataset(
            config['val_path'], MAX_LENGTH, 'par', task_type
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
        save_path = f'models/par_{task_type}_sbert'
        trainer.train(
            train_dataloader, val_dataloader, EPOCHS, LEARNING_RATE,
            save_path=save_path
        )
        if task_type == 'soft_label':
            manhattan_score = trainer.manhattan_distance_evaluation(val_dataloader)
            print(f"Manhattan Distance Score: {manhattan_score:.4f}")
        print(f"Model saved to {save_path}")

def predict_example():
    """Example prediction function for Paraphrase"""
    tokenizer = SentenceTransformer('models/par_soft_label_sbert')
    model = SBertForLeWiDi('all-MiniLM-L6-v2', num_classes=11, task_type='soft_label')
    trainer = LeWiDiTrainer(model)
    trainer.load_model('models/par_soft_label_sbert')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    question1 = "How do I reset my password?"
    question2 = "What's the procedure for changing my login credentials?"
    text = f"{question1} [SEP] {question2}"
    model.eval()
    with torch.no_grad():
        outputs = model([text])
        predictions = outputs['predictions']
    print(f"Soft label distribution: {predictions.cpu().numpy()}")
    predicted_class = torch.argmax(predictions, dim=-1).item() - 5  # -5 for Likert scale
    print(f"Most likely rating: {predicted_class}")

def generate_submission_files():
    """Generate Codabench submission files for Paraphrase dataset using trained models."""
    import json
    import argparse
    from pathlib import Path
    
    # Configuration for Paraphrase dataset
    test_file = 'dataset/Paraphrase/Paraphrase_dev.json'  # Use dev for testing, change to test when available
    max_length = 512
    num_bins = 11  # Likert scale -5 to 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to list if it's a dict
    if isinstance(data, dict):
        data = list(data.values())
    
    # Generate Task A (Soft Label) submission
    print("Generating Task A (Soft Label) submission...")
    soft_model_path = 'models/par_soft_label_sbert'
    tokenizer = SentenceTransformer(soft_model_path)
    model = SBertForLeWiDi('all-MiniLM-L6-v2', num_classes=num_bins, task_type='soft_label')
    trainer = LeWiDiTrainer(model)
    trainer.load_model(soft_model_path)
    model.to(device)
    model.eval()
    
    output_file = 'Paraphrase_test_soft.tsv'
    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, ex in tqdm(enumerate(data), desc="Task A predictions"):
            # Build input text
            question1 = ex['text']['Question1']
            question2 = ex['text']['Question2']
            text = f"{question1} [SEP] {question2}"
            
            # Tokenize and predict
            enc = tokenizer.encode(text, convert_to_tensor=True).to(device)
            with torch.no_grad():
                outputs = model(enc)
                probs = outputs['predictions'].squeeze(0).cpu()
            
            # Format output
            out_probs = probs.tolist()
            # Ensure we output exactly num_bins probabilities
            if len(out_probs) < num_bins:
                pad = [0.0] * (num_bins - len(out_probs))
                out_probs = pad + out_probs
            
            # Round to 10 decimals and fix any rounding drift
            out_probs = [round(p, 10) for p in out_probs]
            drift = 1.0 - sum(out_probs)
            if abs(drift) > 1e-10:
                # add drift to the max prob to keep list summing to 1
                idx_max = max(range(len(out_probs)), key=out_probs.__getitem__)
                out_probs[idx_max] = round(out_probs[idx_max] + drift, 10)
            
            prob_str = ",".join(f"{p:.10f}" for p in out_probs)
            out_f.write(f"{idx}\t[{prob_str}]\n")
    
    print(f"Saved Task A submission file to {output_file}")
    
    # Generate Task B (Perspectivist) submission
    print("Generating Task B (Perspectivist) submission...")
    pe_model_path = 'models/par_perspectivist_sbert'
    tokenizer_pe = SentenceTransformer(pe_model_path)
    model_pe = SBertForLeWiDi('all-MiniLM-L6-v2', num_classes=num_bins, task_type='perspectivist', num_annotators=MAX_ANNOTATORS)
    trainer_pe = LeWiDiTrainer(model_pe)
    trainer_pe.load_model(pe_model_path)
    model_pe.to(device)
    model_pe.eval()
    
    output_file_pe = 'Paraphrase_test_pe.tsv'
    with open(output_file_pe, "w", encoding="utf-8") as out_f:
        for idx, ex in tqdm(enumerate(data), desc="Task B predictions"):
            # Build input text
            question1 = ex['text']['Question1']
            question2 = ex['text']['Question2']
            text = f"{question1} [SEP] {question2}"
            
            # Get annotator list
            ann_list = ex.get("annotators", "").split(",") if ex.get("annotators") else []
            
            # Tokenize and predict
            enc = tokenizer_pe.encode(text, convert_to_tensor=True).to(device)
            with torch.no_grad():
                outputs = model_pe(enc)
                # outputs['predictions']: [1, MAX_ANNOTATORS, num_bins]
                preds = outputs['predictions'].squeeze(0).cpu()  # [MAX_ANNOTATORS, num_bins]
            
            # For each annotator, get the predicted rating (argmax - 5 for Likert scale)
            annotator_preds = []
            for i in range(len(ann_list)):
                rating_idx = torch.argmax(preds[i]).item()
                rating = rating_idx - 5  # Convert to Likert scale -5 to 5
                annotator_preds.append(str(rating))
            
            preds_str = ", ".join(annotator_preds)
            out_f.write(f"{idx}\t[{preds_str}]\n")
    
    print(f"Saved Task B submission file to {output_file_pe}")
    print("To submit: zip -j res.zip", output_file, output_file_pe)

if __name__ == "__main__":
    main()
    # Uncomment to generate submission files after training:
    # generate_submission_files()
    
    # predict_example()  # Uncomment to run example inference 