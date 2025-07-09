import argparse
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


class ParDemogModel(torch.nn.Module):
    """RoBERTa-Large model with demographic embeddings and SBERT embeddings for annotator-aware paraphrase detection."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 8, sbert_dim: int = 384, dropout_rate: float = 0.3, num_classes: int = 11):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer

        # RoBERTa-Large as the main model
        self.roberta_model = AutoModel.from_pretrained(base_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # SBERT model for additional embeddings
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Freeze SBERT model parameters
        for param in self.sbert_model.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        
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


class ParDataset:
    """Dataset class for loading test data with demographic processing."""
    
    PAD_IDX = 0  # padding
    UNK_IDX = 1  # unknown / missing

    # Reduced field set for better performance
    REDUCED_FIELD_KEYS = {
        "age": "Age",  # Will be binned
        "gender": "Gender", 
        "ethnicity": "Ethnicity simplified",
        "country_residence": "Country of residence",
        "employment": "Employment status",
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

    def __init__(self, test_file: str, annot_meta_path: str):
        self.active_field_keys = self.REDUCED_FIELD_KEYS
        
        # Load annotator metadata
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            self.annot_meta = json.load(f)

        # Build vocabulary from all annotators (same as training)
        self.vocab = {
            field: {"<PAD>": self.PAD_IDX, "<UNK>": self.UNK_IDX}
            for field in self.active_field_keys
        }

        for ann_data in self.annot_meta.values():
            for field, json_key in self.active_field_keys.items():
                if field == "age":
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
        
        # Load test data
        with open(test_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_demographic_ids(self, annotator_id):
        """Get demographic IDs for a specific annotator."""
        meta = self.annot_meta.get(annotator_id, {})
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
            
        return annotator_demog_ids


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load dataset to get vocabulary sizes
    dataset = ParDataset(args.test_file, args.annot_meta)
    print(f"Demographic vocabulary sizes: {dataset.vocab_sizes}")

    # Load model
    model = ParDemogModel(
        base_name=args.model_name,
        vocab_sizes=dataset.vocab_sizes,
        dem_dim=args.dem_dim,
        sbert_dim=args.sbert_dim,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
    )
    
    # Load trained weights
    model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Build output filename if not provided
    if args.output_tsv is None:
        dataset_name = Path(args.test_file).stem.replace("_clear", "")
        suffix = "_pe.tsv" if args.task == "B" else "_soft.tsv"
        args.output_tsv = f"{dataset_name}{suffix}"

    with open(args.output_tsv, "w", encoding="utf-8") as out_f:
        for ex_id, ex in tqdm(dataset.data.items(), desc="Predicting"):
            question1 = ex["text"].get("Question1", "")
            question2 = ex["text"].get("Question2", "")
            full_text = f"{question1} [SEP] {question2}".strip()

            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            
            if args.task == "A":
                # Task A: Generate soft label distribution
                # Use average of all annotators' demographics or UNK if none
                if ann_list:
                    # Get demographic IDs for the first annotator (or average if multiple)
                    ann_num = ann_list[0][3:] if ann_list[0].startswith("Ann") else ann_list[0]
                    demog_ids = dataset.get_demographic_ids(ann_num)
                else:
                    # Use UNK for all fields if no annotators
                    demog_ids = {field: dataset.UNK_IDX for field in dataset.active_field_keys}
                
                # Tokenize text for RoBERTa
                enc = model.tokenizer(
                    full_text,
                    max_length=512,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                # Prepare demographic inputs
                demographic_inputs = {f"{field}_ids": torch.tensor([demog_ids[field]], dtype=torch.long).to(device) 
                                    for field in dataset.active_field_keys}
                
                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        texts=[full_text],
                        **demographic_inputs
                    )
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                
                # Convert 11-class distribution (-5 to 5) to 7-class distribution (-3 to 3)
                # This is a simple mapping - you might want to adjust this based on your task
                if args.num_bins == 7:
                    # Map 11 classes to 7 classes
                    mapped_probs = torch.zeros(7)
                    # Map -5,-4 to -3; -3,-2 to -2; -1,0 to -1; 1,2 to 0; 3,4 to 1; 5 to 2
                    mapping = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]  # 11 -> 7 mapping
                    for i, prob in enumerate(probs):
                        mapped_probs[mapping[i]] += prob
                    probs = mapped_probs
                
                # Ensure we output exactly num_bins probabilities
                out_probs = probs.tolist()
                if len(out_probs) < args.num_bins:
                    pad = [0.0] * (args.num_bins - len(out_probs))
                    out_probs = pad + out_probs
                
                # Round to 10 decimals and fix any rounding drift
                out_probs = [round(p, 10) for p in out_probs]
                drift = 1.0 - sum(out_probs)
                if abs(drift) > 1e-10:
                    idx_max = max(range(len(out_probs)), key=lambda i: out_probs[i])
                    out_probs[idx_max] = round(out_probs[idx_max] + drift, 10)
                
                prob_str = ",".join(f"{p:.10f}" for p in out_probs)
                out_f.write(f"{ex_id}\t[{prob_str}]\n")
                
            else:
                # Task B: Generate individual predictions for each annotator
                predictions = []
                for ann_tag in ann_list:
                    ann_num = ann_tag[3:] if ann_tag.startswith("Ann") else ann_tag
                    demog_ids = dataset.get_demographic_ids(ann_num)
                    
                    # Tokenize text for RoBERTa
                    enc = model.tokenizer(
                        full_text,
                        max_length=512,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    
                    # Prepare demographic inputs
                    demographic_inputs = {f"{field}_ids": torch.tensor([demog_ids[field]], dtype=torch.long).to(device) 
                                        for field in dataset.active_field_keys}
                    
                    with torch.no_grad():
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            texts=[full_text],
                            **demographic_inputs
                        )
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                    
                    # Convert to rating (argmax + offset if needed)
                    rating_idx = torch.argmax(probs).item()
                    if args.num_bins == 7:
                        # Map from 11-class to 7-class rating
                        rating_mapping = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
                        rating = rating_mapping[rating_idx]
                    else:
                        rating = rating_idx
                    
                    predictions.append(str(rating))
                
                # If no annotators, use UNK demographics
                if not ann_list:
                    demog_ids = {field: dataset.UNK_IDX for field in dataset.active_field_keys}
                    demographic_inputs = {f"{field}_ids": torch.tensor([demog_ids[field]], dtype=torch.long).to(device) 
                                        for field in dataset.active_field_keys}
                    
                    # Tokenize text for RoBERTa
                    enc = model.tokenizer(
                        full_text,
                        max_length=512,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    
                    with torch.no_grad():
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            texts=[full_text],
                            **demographic_inputs
                        )
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                    
                    rating_idx = torch.argmax(probs).item()
                    if args.num_bins == 7:
                        rating_mapping = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
                        rating = rating_mapping[rating_idx]
                    else:
                        rating = rating_idx
                    
                    predictions.append(str(rating))
                
                preds = ", ".join(predictions)
                out_f.write(f"{ex_id}\t[{preds}]\n")

    print(f"Saved submission file to {args.output_tsv}")
    print(f"To submit: zip -j res.zip {args.output_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission TSV for Paraphrase dataset using RoBERTa + SBERT + Demographics model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory (containing pytorch_model.bin)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to *_test_clear.json file")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json", help="Path to annotator metadata JSON")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="RoBERTa model name")
    parser.add_argument("--task", choices=["A", "B"], default="A", help="Which task: A (soft) or B (perspectivist)")
    parser.add_argument("--output_tsv", type=str, default=None, help="Optional output filename")
    parser.add_argument("--num_bins", type=int, default=7, help="Number of bins to output for task A.")
    parser.add_argument("--dem_dim", type=int, default=8, help="Dimension of each demographic embedding")
    parser.add_argument("--sbert_dim", type=int, default=384, help="Dimension of SBERT embeddings")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes (Likert scale -5 to 5)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    main(args) 