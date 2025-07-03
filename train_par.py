import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
import os
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

class Par_Dataset(Dataset):
    """Dataset class for Par dataset"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                 dataset_type: str = "par", task_type: str = "soft_label"):
        self.tokenizer = tokenizer
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
        # Extract questions from the 'text' field
        question1 = item['text']['Question1']
        question2 = item['text']['Question2']
        # For soft_label task, use the 'soft_label' field
        if self.task_type == "soft_label":
            # The soft_label field is a dict with string keys from -5 to 5
            soft_label_dict = item['soft_label']
            # Ensure order from -5 to 5
            soft_label = [float(soft_label_dict[str(i)]) for i in range(-5, 6)]
            soft_label = torch.tensor(soft_label, dtype=torch.float)
            labels = soft_label
            annotator_ids = []
        else:
            # For perspectivist, extract annotation values in the order of 'annotators' field
            annotator_list = [a.strip() for a in item['annotators'].split(',')]
            annotations_dict = item['annotations']
            # Get annotation for each annotator in order, convert to float
            annotations = [float(annotations_dict[ann]) for ann in annotator_list]
            labels = torch.tensor(annotations, dtype=torch.float)
            annotator_ids = annotator_list
        text = f"{question1} [SEP] {question2}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'annotator_ids': torch.tensor([])  # Always a tensor, not a list
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

class RoBERTaForLeWiDi(nn.Module):
    def __init__(self, model_name: str, num_classes: int, task_type: str = "soft_label",
                 num_annotators: Optional[int] = None):
        super(RoBERTaForLeWiDi, self).__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        if task_type == "soft_label":
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
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, labels, reduction='batchmean')
            else:
                # logits: [batch_size, num_annotators * num_classes] -> [batch_size, num_annotators, num_classes]
                logits = logits.view(-1, self.num_annotators, self.num_classes)
                # labels: [batch_size, num_annotators] (float Likert values)
                # For cross_entropy, labels must be integer class indices in [0, num_classes-1]
                # Map Likert scale (-5 to 5) to class indices (0 to 10)
                labels = labels + 5  # shift -5..5 to 0..10
                labels = labels.long()
                loss = F.cross_entropy(
                    logits.view(-1, self.num_classes),      # [batch_size * num_annotators, num_classes]
                    labels.view(-1),                        # [batch_size * num_annotators]
                    ignore_index=-100
                )
        return {
            'loss': loss,
            'logits': logits,
            'predictions': F.softmax(logits, dim=-1) if self.task_type == "soft_label" else logits
        }

class LeWiDiTrainer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5,
              warmup_steps=0.1, save_path=None):
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                annotator_ids = batch.get('annotator_ids', None)
                if annotator_ids is not None:
                    annotator_ids = annotator_ids.to(self.device)
                outputs = self.model(input_ids, attention_mask, labels, annotator_ids)
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
                manhattan_dist = torch.sum(torch.abs(predictions - labels), dim=1)
                total_distance += manhattan_dist.sum().item()
                total_samples += len(predictions)
        self.model.train()
        return total_distance / total_samples
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        self.tokenizer.save_pretrained(path)
    def load_model(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

def main():
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 5  # Set to 5 epochs for real training
    LEARNING_RATE = 2e-5

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

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
            config['train_path'], tokenizer, MAX_LENGTH, 'par', task_type
        )
        val_dataset = Par_Dataset(
            config['val_path'], tokenizer, MAX_LENGTH, 'par', task_type
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        num_annotators = 10 if task_type == 'perspectivist' else None
        model = RoBERTaForLeWiDi(
            MODEL_NAME, config['num_classes'], task_type, num_annotators
        )
        trainer = LeWiDiTrainer(model, tokenizer)
        save_path = f'models/par_{task_type}_roberta'
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
    tokenizer = RobertaTokenizer.from_pretrained('models/par_soft_label_roberta')
    model = RoBERTaForLeWiDi('roberta-base', num_classes=11, task_type='soft_label')
    trainer = LeWiDiTrainer(model, tokenizer)
    trainer.load_model('models/par_soft_label_roberta')
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
    predicted_class = torch.argmax(predictions, dim=-1).item() - 5  # -5 for Likert scale
    print(f"Most likely rating: {predicted_class}")

if __name__ == "__main__":
    main()
    
    # predict_example()  # Uncomment to run example inference 