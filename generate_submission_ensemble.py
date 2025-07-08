import argparse
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


class CrossAttentionLayer(torch.nn.Module):
    """Cross-attention between text and demographic embeddings."""
    
    def __init__(self, text_dim, dem_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert self.head_dim * num_heads == text_dim
        
        self.q_proj = torch.nn.Linear(text_dim, text_dim)
        self.k_proj = torch.nn.Linear(dem_dim, text_dim)
        self.v_proj = torch.nn.Linear(dem_dim, text_dim)
        self.out_proj = torch.nn.Linear(text_dim, text_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(text_dim)
        self.norm2 = torch.nn.LayerNorm(text_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(text_dim, text_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(text_dim * 4, text_dim)
        )
        
    def forward(self, text_emb, dem_emb):
        B, L, D = text_emb.shape
        
        q = self.q_proj(text_emb).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dem_emb).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dem_emb).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        text_emb = self.norm1(text_emb + attn_output)
        ffn_output = self.ffn(text_emb)
        text_emb = self.norm2(text_emb + ffn_output)
        
        return text_emb


class CrossAttentionParModel(torch.nn.Module):
    """Advanced model with cross-attention."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.text_model = SentenceTransformer(base_name)
        self.num_classes = num_classes
        
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        emb_dim = self.text_model.get_sentence_embedding_dimension()
        emb_dim = int(emb_dim)
        
        self.text_proj = torch.nn.Linear(emb_dim, 768)
        
        self.cross_attention_layers = torch.nn.ModuleList([
            CrossAttentionLayer(768, dem_dim, num_heads=12, dropout=dropout_rate)
            for _ in range(2)
        ])
        
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, *, texts, **demographic_inputs):
        text_embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        text_embeddings = self.text_proj(text_embeddings)
        
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)
        
        if demographic_vectors:
            dem_emb = torch.cat(demographic_vectors, dim=-1)
            
            for cross_attn in self.cross_attention_layers:
                text_embeddings = cross_attn(text_embeddings, dem_emb)
        
        pooled = torch.mean(text_embeddings, dim=1)
        logits = self.classifier(pooled)
        return logits


class SimpleConcatParModel(torch.nn.Module):
    """Simple concatenation model."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.text_model = SentenceTransformer(base_name)
        self.num_classes = num_classes
        
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        emb_dim = self.text_model.get_sentence_embedding_dimension()
        emb_dim = int(emb_dim)
        
        num_demog_fields = len(vocab_sizes)
        total_dim = emb_dim + num_demog_fields * dem_dim
        
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, *, texts, **demographic_inputs):
        text_embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        if demographic_vectors:
            concat = torch.cat([text_embeddings] + demographic_vectors, dim=-1)
        else:
            concat = text_embeddings
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class DemographicOnlyModel(torch.nn.Module):
    """Model that focuses heavily on demographic information."""
    
    def __init__(self, base_name: str, vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.text_model = SentenceTransformer(base_name)
        self.num_classes = num_classes
        
        self.demographic_embeddings = torch.nn.ModuleDict()
        for field, vocab_size in vocab_sizes.items():
            self.demographic_embeddings[field] = torch.nn.Embedding(vocab_size, dem_dim, padding_idx=0)

        emb_dim = self.text_model.get_sentence_embedding_dimension()
        emb_dim = int(emb_dim)
        
        self.text_proj = torch.nn.Linear(emb_dim, 256)
        num_demog_fields = len(vocab_sizes)
        dem_dim_total = num_demog_fields * dem_dim
        
        total_dim = 256 + dem_dim_total
        
        self.demographic_attention = torch.nn.MultiheadAttention(dem_dim_total, num_heads=8, dropout=dropout_rate)
        self.norm = torch.nn.LayerNorm(total_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, *, texts, **demographic_inputs):
        text_embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        text_embeddings = self.text_proj(text_embeddings)
        
        demographic_vectors = []
        for field, emb_layer in self.demographic_embeddings.items():
            field_key = f"{field}_ids"
            if field_key in demographic_inputs:
                demographic_vec = emb_layer(demographic_inputs[field_key])
                demographic_vectors.append(demographic_vec)

        if demographic_vectors:
            dem_emb = torch.cat(demographic_vectors, dim=-1)
            dem_emb = dem_emb.unsqueeze(1)
            dem_emb, _ = self.demographic_attention(dem_emb, dem_emb, dem_emb)
            dem_emb = dem_emb.squeeze(1)
            
            concat = torch.cat([text_embeddings, dem_emb], dim=-1)
        else:
            concat = text_embeddings
            
        concat = self.norm(concat)
        concat = self.dropout(concat)
        logits = self.classifier(concat)
        return logits


class EnsembleParDemogModel(torch.nn.Module):
    """Ensemble of multiple models with different architectures."""
    
    def __init__(self, model_configs: List[Dict], vocab_sizes: dict, dem_dim: int = 32, 
                 dropout_rate: float = 0.1, num_classes: int = 11):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.num_classes = num_classes
        
        for config in model_configs:
            model = self._create_single_model(config, vocab_sizes, dem_dim, dropout_rate, num_classes)
            self.models.append(model)
    
    def _create_single_model(self, config: Dict, vocab_sizes: dict, dem_dim: int, 
                           dropout_rate: float, num_classes: int):
        base_name = config['base_name']
        model_type = config.get('model_type', 'cross_attention')
        
        if model_type == 'cross_attention':
            return CrossAttentionParModel(base_name, vocab_sizes, dem_dim, dropout_rate, num_classes)
        elif model_type == 'simple_concat':
            return SimpleConcatParModel(base_name, vocab_sizes, dem_dim, dropout_rate, num_classes)
        elif model_type == 'demographic_only':
            return DemographicOnlyModel(base_name, vocab_sizes, dem_dim, dropout_rate, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, *, texts, **demographic_inputs):
        all_logits = []
        
        for model in self.models:
            logits = model(texts=texts, **demographic_inputs)
            all_logits.append(logits)
        
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        return ensemble_logits


class EnsembleDataset:
    """Dataset class for loading test data with demographic processing."""
    
    PAD_IDX = 0
    UNK_IDX = 1

    FIELD_KEYS = {
        "age": "Age",
        "gender": "Gender",
        "ethnicity": "Ethnicity simplified",
        "country_birth": "Country of birth",
        "country_residence": "Country of residence",
        "nationality": "Nationality",
        "student": "Student status",
        "employment": "Employment status",
    }

    PROMPT_TEMPLATES = [
        "Question 1: {q1}\nQuestion 2: {q2}\nAre these questions asking about the same thing?",
        "Q1: {q1}\nQ2: {q2}\nRate how similar these questions are:",
        "Compare these two questions:\n{q1}\n{q2}\nHow similar are they?",
        "Question A: {q1}\nQuestion B: {q2}\nAssess their similarity:",
        "Text 1: {q1}\nText 2: {q2}\nEvaluate similarity:",
    ]

    @staticmethod
    def get_age_bin(age):
        if age is None or str(age).strip() == "" or str(age) == "DATA_EXPIRED":
            return "<UNK>"
        try:
            age = float(age)
            if age < 20: return "18-19"
            elif age < 25: return "20-24"
            elif age < 30: return "25-29"
            elif age < 35: return "30-34"
            elif age < 40: return "35-39"
            elif age < 50: return "40-49"
            else: return "50+"
        except (ValueError, TypeError):
            return "<UNK>"

    def __init__(self, test_file: str, annot_meta_path: str, use_prompts: bool = True):
        self.active_field_keys = self.FIELD_KEYS
        self.use_prompts = use_prompts
        
        with open(annot_meta_path, "r", encoding="utf-8") as f:
            self.annot_meta = json.load(f)

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
        
        with open(test_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_demographic_ids(self, annotator_id):
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

    def get_text_formats(self, question1, question2):
        text_formats = []
        
        if self.use_prompts:
            for template in self.PROMPT_TEMPLATES:
                text_formats.append(template.format(q1=question1, q2=question2))
        
        text_formats.extend([
            f"{question1} [SEP] {question2}",
            f"Question 1: {question1} Question 2: {question2}",
            f"Q1: {question1} Q2: {question2}",
            f"{question1} ||| {question2}",
            f"Text A: {question1} Text B: {question2}",
        ])
        
        return text_formats


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = EnsembleDataset(args.test_file, args.annot_meta, use_prompts=args.use_prompts)
    print(f"Demographic vocabulary sizes: {dataset.vocab_sizes}")

    # Model configurations (must match training)
    model_configs = [
        {'base_name': 'all-mpnet-base-v2', 'model_type': 'cross_attention'},
        {'base_name': 'all-MiniLM-L12-v2', 'model_type': 'simple_concat'},
        {'base_name': 'multi-qa-mpnet-base-dot-v1', 'model_type': 'demographic_only'},
    ]

    # Load ensemble model
    model = EnsembleParDemogModel(
        model_configs=model_configs,
        vocab_sizes=dataset.vocab_sizes,
        dem_dim=args.dem_dim,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
    )
    
    model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Build output filename
    if args.output_tsv is None:
        dataset_name = Path(args.test_file).stem.replace("_clear", "")
        suffix = "_pe.tsv" if args.task == "B" else "_soft.tsv"
        args.output_tsv = f"{dataset_name}{suffix}"

    with open(args.output_tsv, "w", encoding="utf-8") as out_f:
        for ex_id, ex in tqdm(dataset.data.items(), desc="Predicting"):
            question1 = ex["text"].get("Question1", "")
            question2 = ex["text"].get("Question2", "")
            
            # Get multiple text formats for ensemble prediction
            text_formats = dataset.get_text_formats(question1, question2)
            
            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()] if ann_str else []
            
            if args.task == "A":
                # Task A: Generate soft label distribution
                if ann_list:
                    ann_num = ann_list[0][3:] if ann_list[0].startswith("Ann") else ann_list[0]
                    demog_ids = dataset.get_demographic_ids(ann_num)
                else:
                    demog_ids = {field: dataset.UNK_IDX for field in dataset.active_field_keys}
                
                demographic_inputs = {f"{field}_ids": torch.tensor([demog_ids[field]], dtype=torch.long).to(device) 
                                    for field in dataset.active_field_keys}
                
                # Ensemble prediction across text formats
                all_probs = []
                for text_format in text_formats:
                    with torch.no_grad():
                        logits = model(texts=[text_format], **demographic_inputs)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                        all_probs.append(probs)
                
                # Average predictions across text formats
                ensemble_probs = torch.stack(all_probs).mean(dim=0)
                
                # Convert 11-class to 7-class if needed
                if args.num_bins == 7:
                    mapping = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
                    mapped_probs = torch.zeros(7)
                    for i, prob in enumerate(ensemble_probs):
                        mapped_probs[mapping[i]] += prob
                    ensemble_probs = mapped_probs
                
                # Ensure correct number of bins
                out_probs = ensemble_probs.tolist()
                if len(out_probs) < args.num_bins:
                    pad = [0.0] * (args.num_bins - len(out_probs))
                    out_probs = pad + out_probs
                
                # Round and fix drift
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
                    
                    demographic_inputs = {f"{field}_ids": torch.tensor([demog_ids[field]], dtype=torch.long).to(device) 
                                        for field in dataset.active_field_keys}
                    
                    # Ensemble prediction across text formats
                    all_probs = []
                    for text_format in text_formats:
                        with torch.no_grad():
                            logits = model(texts=[text_format], **demographic_inputs)
                            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                            all_probs.append(probs)
                    
                    ensemble_probs = torch.stack(all_probs).mean(dim=0)
                    rating_idx = torch.argmax(ensemble_probs).item()
                    
                    if args.num_bins == 7:
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
                    
                    all_probs = []
                    for text_format in text_formats:
                        with torch.no_grad():
                            logits = model(texts=[text_format], **demographic_inputs)
                            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                            all_probs.append(probs)
                    
                    ensemble_probs = torch.stack(all_probs).mean(dim=0)
                    rating_idx = torch.argmax(ensemble_probs).item()
                    
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
    parser = argparse.ArgumentParser(description="Generate ensemble submission for Paraphrase dataset.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to ensemble model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test file")
    parser.add_argument("--annot_meta", type=str, default="dataset/Paraphrase/Paraphrase_annotators_meta.json")
    parser.add_argument("--task", choices=["A", "B"], default="A", help="Task A (soft) or B (perspectivist)")
    parser.add_argument("--output_tsv", type=str, default=None, help="Output filename")
    parser.add_argument("--num_bins", type=int, default=7, help="Number of bins for task A")
    parser.add_argument("--dem_dim", type=int, default=32, help="Demographic embedding dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes")
    parser.add_argument("--use_prompts", action="store_true", help="Use LLM-style prompts")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    main(args) 