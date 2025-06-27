import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaModel, RobertaConfig,
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Par_VariErrNLI(Dataset):
    """Dataset class for Par and VariErrNLI datasets"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                 dataset_type: str = "par", task_type: str = "soft_label"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.task_type = task_type
        
        # Load data
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str):
        """Load dataset based on type"""
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
            item = self.data.iloc[idx]
        
        if self.dataset_type == "par":
            return self.process_par_item(item)
        elif self.dataset_type == "varierrnli":
            return self.process_varierrnli_item(item)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def process_par_item(self, item):
        """Process Paraphrase Detection dataset item"""
        if isinstance(item, dict):
            question1 = item.get('question1', '')
            question2 = item.get('question2', '')
            annotations = item.get('annotations', [])
            annotator_ids = item.get('annotator_ids', [])
        else:
            question1 = item['question1']
            question2 = item['question2']
            annotations = eval(item['annotations']) if isinstance(item['annotations'], str) else item['annotations']
            annotator_ids = eval(item['annotator_ids']) if isinstance(item['annotator_ids'], str) else item['annotator_ids']
        
        # Tokenize the question pair
        text = f"{question1} [SEP] {question2}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process labels based on task type
        if self.task_type == "soft_label":
            # Create soft label distribution (Likert scale 1-5)
            soft_label = self.create_soft_label(annotations, scale_range=(1, 6))
        else:  # perspectivist
            # Store individual annotator labels
            soft_label = torch.tensor(annotations, dtype=torch.float)
            
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': soft_label,
            'annotator_ids': torch.tensor(annotator_ids) if annotator_ids else torch.tensor([])
        }
    
    def process_varierrnli_item(self, item):
        """Process VariErrNLI dataset item"""
        if isinstance(item, dict):
            premise = item.get('premise', '')
            hypothesis = item.get('hypothesis', '')
            annotations = item.get('annotations', [])
            annotator_ids = item.get('annotator_ids', [])
        else:
            premise = item['premise']
            hypothesis = item['hypothesis']
            annotations = eval(item['annotations']) if isinstance(item['annotations'], str) else item['annotations']
            annotator_ids = eval(item['annotator_ids']) if isinstance(item['annotator_ids'], str) else item['annotator_ids']
        
        # Tokenize premise-hypothesis pair
        text = f"{premise} [SEP] {hypothesis}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process labels based on task type
        if self.task_type == "soft_label":
            # Create soft label distribution for 3-class NLI (entailment, neutral, contradiction)
            soft_label = self.create_soft_label(annotations, scale_range=(0, 3), categorical=True)
        else:  # perspectivist
            soft_label = torch.tensor(annotations, dtype=torch.float)
            
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': soft_label,
            'annotator_ids': torch.tensor(annotator_ids) if annotator_ids else torch.tensor([])
        }
    
    def create_soft_label(self, annotations: List, scale_range: Tuple[int, int], categorical: bool = False):
        """Create soft label distribution from annotations"""
        if categorical:
            # For categorical labels (NLI)
            num_classes = scale_range[1] - scale_range[0]
            soft_label = torch.zeros(num_classes)
            for ann in annotations:
                if scale_range[0] <= ann < scale_range[1]:
                    soft_label[ann - scale_range[0]] += 1
        else:
            # For Likert scale (Paraphrase Detection)
            num_classes = scale_range[1] - scale_range[0]
            soft_label = torch.zeros(num_classes)
            for ann in annotations:
                if scale_range[0] <= ann < scale_range[1]:
                    soft_label[ann - scale_range[0]] += 1
        
        # Normalize to create probability distribution
        if soft_label.sum() > 0:
            soft_label = soft_label / soft_label.sum()
        
        return soft_label

class RoBERTaForLeWiDi(nn.Module):
    """RoBERTa model for LeWiDi tasks with soft labeling and perspectivist approaches"""
    
    def __init__(self, model_name: str, num_classes: int, task_type: str = "soft_label",
                 num_annotators: Optional[int] = None):
        super(RoBERTaForLeWiDi, self).__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        
        # Load RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        if task_type == "soft_label":
            # For Task A: Output probability distribution
            self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        else:
            if num_annotators is None:
                raise ValueError("num_annotators must be provided for perspectivist task_type.")
            self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes * num_annotators)
    
    def forward(self, input_ids, attention_mask, labels=None, annotator_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.task_type == "soft_label":
                # Soft label loss using KL divergence
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, labels, reduction='batchmean')
            else:
                # Perspectivist loss
                # Reshape logits for per-annotator predictions
                logits = logits.view(-1, self.num_annotators, self.num_classes)
                loss = F.cross_entropy(logits.view(-1, self.num_classes), 
                                     labels.long().view(-1), ignore_index=-100)
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': F.softmax(logits, dim=-1) if self.task_type == "soft_label" else logits
        }

class LeWiDiTrainer:
    """Trainer class for LeWiDi models"""
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5,
              warmup_steps=0.1, save_path=None):
        """Train the model"""
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * warmup_steps) if warmup_steps < 1 else warmup_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                annotator_ids = batch.get('annotator_ids', None)
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels, annotator_ids)
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # Validation
            val_loss = self.evaluate(val_dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_dataloader):.4f}, '
                  f'Val Loss: {val_loss:.4f}')
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
                print(f'Model saved to {save_path}')
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                annotator_ids = batch.get('annotator_ids', None)
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels, annotator_ids)
                total_loss += outputs['loss'].item()
        
        self.model.train()
        return total_loss / len(dataloader)
    
    def manhattan_distance_evaluation(self, dataloader):
        """Evaluate using Manhattan distance for soft labels"""
        self.model.eval()
        total_distance = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = outputs['predictions']
                
                # Calculate Manhattan distance
                manhattan_dist = torch.sum(torch.abs(predictions - labels), dim=1)
                total_distance += manhattan_dist.sum().item()
                total_samples += len(predictions)
        
        self.model.train()
        return total_distance / total_samples
    
    def save_model(self, path):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        self.tokenizer.save_pretrained(path)
        
    def load_model(self, path):
        """Load model"""
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

def main():
    """Main training function"""
    
    # Configuration
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Dataset configurations
    datasets_config = {
        'par': {
            'num_classes': 6,  # Likert scale 1-6
            'train_path': 'dataset/Paraphrase/Paraphrase_train.json', 
            'val_path': 'dataset/Paraphrase/Paraphrase_dev.json',
            'test_path': 'models/Paraphrase_test.json'
        },
        'varierrnli': {
            'num_classes': 3,  # Entailment, Neutral, Contradiction
            'train_path': 'dataset/VariErrNLI/VariErrNLI_train.json', 
            'val_path': 'dataset/VariErrNLI/VariErrNLI_dev.json',
            'test_path': 'models/VariErrNLI_test.json'
        }
    }
    
    # Train models for both datasets and both tasks
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*50}")
        print(f"Training on {dataset_name.upper()} dataset")
        print(f"{'='*50}")
        
        for task_type in ['soft_label', 'perspectivist']:
            print(f"\nTask Type: {task_type}")
            print(f"-" * 30)
            
            # Create datasets
            train_dataset = Par_VariErrNLI(
                config['train_path'], tokenizer, MAX_LENGTH, 
                dataset_name, task_type
            )
            val_dataset = Par_VariErrNLI(
                config['val_path'], tokenizer, MAX_LENGTH, 
                dataset_name, task_type
            )
            
            # Create dataloaders
            train_dataloader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False
            )
            
            # Initialize model
            num_annotators = 10 if task_type == 'perspectivist' else None  # Adjust as needed
            model = RoBERTaForLeWiDi(
                MODEL_NAME, config['num_classes'], task_type, num_annotators
            )
            
            # Initialize trainer
            trainer = LeWiDiTrainer(model, tokenizer)
            
            # Train model
            save_path = f'models/{dataset_name}_{task_type}_roberta'
            trainer.train(
                train_dataloader, val_dataloader, EPOCHS, LEARNING_RATE,
                save_path=save_path
            )
            
            # Evaluate with Manhattan distance (for soft label task)
            if task_type == 'soft_label':
                manhattan_score = trainer.manhattan_distance_evaluation(val_dataloader)
                print(f"Manhattan Distance Score: {manhattan_score:.4f}")
            
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

# Example usage for inference
def predict_example():
    """Example prediction function"""
    
    # Load trained model
    tokenizer = RobertaTokenizer.from_pretrained('models/par_soft_label_roberta')
    model = RoBERTaForLeWiDi('roberta-base', num_classes=6, task_type='soft_label')
    trainer = LeWiDiTrainer(model, tokenizer)
    trainer.load_model('models/par_soft_label_roberta')
    
    # Example paraphrase detection
    question1 = "How do I reset my password?"
    question2 = "What's the procedure for changing my login credentials?"
    
    text = f"{question1} [SEP] {question2}"
    encoding = tokenizer(
        text, truncation=True, padding='max_length', 
        max_length=512, return_tensors='pt'
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        predictions = outputs['predictions']
        
    print(f"Soft label distribution: {predictions.numpy()}")
    predicted_class = torch.argmax(predictions, dim=-1).item() + 1  # +1 for 1-6 scale
    print(f"Most likely rating: {predicted_class}")


predict_example()