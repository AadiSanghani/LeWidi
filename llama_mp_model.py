import json
import torch
import re
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPDataset(Dataset):
    """Dataset class for MP irony detection data"""
    
    def __init__(self, data_path: str, annotators_meta_path: str, include_labels: bool = True):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        with open(annotators_meta_path, 'r') as f:
            self.annotators_meta = json.load(f)
            
        self.include_labels = include_labels
        self.samples = list(self.data.values())
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        item = {
            'id': list(self.data.keys())[idx],
            'post': sample['text']['post'],
            'reply': sample['text']['reply'],
            'annotators': sample['annotators'].split(',') if sample['annotators'] else [],
        }
        
        if self.include_labels and sample.get('annotations'):
            item['annotations'] = sample['annotations']
            item['soft_label'] = sample['soft_label']
            
        return item


class LlamaIronyDetector:
    """Llama-based irony detection model using prompting"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
    def create_prompt(self, post: str, reply: str, task_type: str = "general", 
                      annotator_context: Optional[Dict] = None) -> str:
        """Create prompt for irony detection"""
        
        # Clean text
        post = post.replace('@USER', '[USER]').strip()
        reply = reply.replace('@USER', '[USER]').strip()
        
        if task_type == "general":
            prompt = f"""<s>[INST] Analyze this social media conversation for irony:

Post: "{post}"
Reply: "{reply}"

Is the reply ironic? Consider:
- Does it say something positive about a negative situation?
- Does it use obvious exaggeration or contradiction?
- Does it mean the opposite of what it literally says?

Answer with ONLY a number:
0 = Not ironic/sarcastic
1 = Ironic/sarcastic

Answer: [/INST]"""

        elif task_type == "perspectivist" and annotator_context:
            context = self._get_annotator_context(annotator_context)
            prompt = f"""<s>[INST] From the perspective of a {context}, analyze this conversation for irony:

Post: "{post}"
Reply: "{reply}"

Is the reply ironic?

Answer with ONLY a number:
0 = Not ironic/sarcastic  
1 = Ironic/sarcastic

Answer: [/INST]"""
        
        return prompt
    
    def _get_annotator_context(self, annotator_meta: Dict) -> str:
        """Generate demographic context for perspectivist prediction"""
        age = annotator_meta.get('Age', 'unknown age')
        gender = annotator_meta.get('Gender', 'unknown gender').lower()
        ethnicity = annotator_meta.get('Ethnicity simplified', 'unknown ethnicity').lower()
        country = annotator_meta.get('Country of residence', 'unknown country')
        
        context_parts = []
        
        if age != 'unknown age':
            age_group = "young adult" if age < 30 else "middle-aged adult" if age < 50 else "older adult"
            context_parts.append(age_group)
            
        if gender != 'unknown gender':
            context_parts.append(gender)
            
        if ethnicity not in ['unknown ethnicity', 'data_expired']:
            context_parts.append(f"person of {ethnicity} background")
            
        if country != 'unknown country':
            context_parts.append(f"from {country}")
            
        return ", ".join(context_parts) if context_parts else "person"
    
    def predict_single(self, post: str, reply: str, task_type: str = "general", 
                      annotator_context: Optional[Dict] = None, temperature: float = 0.1) -> float:
        """Predict irony probability for a single sample"""
        
        prompt = self.create_prompt(post, reply, task_type, annotator_context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self._parse_prediction(generated)
    
    def _parse_prediction(self, text: str) -> float:
        """Parse model output to get probability"""
        text = text.strip().lower()
        
        if text.startswith('0') or text == '0':
            return 0.0
        elif text.startswith('1') or text == '1':
            return 1.0
        
        return 0.0
    
    def predict_soft_labels(self, dataset: List[Dict], batch_size: int = 1) -> List[Dict]:
        """Predict soft labels for Task A"""
        results = []
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting soft labels"):
            batch = dataset[i:i+batch_size]
            
            for sample in batch:
                predictions = []
                temperatures = [0.1, 0.3, 0.5]
                
                for temp in temperatures:
                    pred = self.predict_single(sample['post'], sample['reply'], temperature=temp)
                    predictions.append(pred)
                
                irony_prob = sum(predictions) / len(predictions)
                
                prob_0 = round(1.0 - irony_prob, 6)
                prob_1 = round(irony_prob, 6)
                
                if abs((prob_0 + prob_1) - 1.0) > 1e-6:
                    prob_1 = round(1.0 - prob_0, 6)
                
                soft_label = {"0.0": prob_0, "1.0": prob_1}
                
                results.append({
                    'id': sample['id'],
                    'soft_label': soft_label
                })
                
        return results
    
    def predict_annotator_labels(self, dataset: List[Dict], annotators_meta: Dict, 
                                batch_size: int = 1) -> List[Dict]:
        """Predict individual annotator labels for Task B"""
        results = []
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting annotator labels"):
            batch = dataset[i:i+batch_size]
            
            for sample in batch:
                annotator_predictions = {}
                
                for annotator_id in sample['annotators']:
                    if annotator_id in annotators_meta:
                        annotator_context = annotators_meta[annotator_id]
                        pred = self.predict_single(
                            sample['post'], 
                            sample['reply'], 
                            task_type="perspectivist",
                            annotator_context=annotator_context
                        )
                        annotator_predictions[annotator_id] = str(int(pred > 0.5))
                    else:
                        pred = self.predict_single(sample['post'], sample['reply'])
                        annotator_predictions[annotator_id] = str(int(pred > 0.5))
                
                results.append({
                    'id': sample['id'],
                    'annotations': annotator_predictions,
                    'annotator_order': sample['annotators']
                })
                
        return results

def save_soft_labels_tsv(predictions: List[Dict], output_path: str):
    """Save soft label predictions in TSV format for Task A"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            item_id = pred['id']
            soft_label = pred['soft_label']
            
            out_probs = [soft_label['0.0'], soft_label['1.0']]
            out_probs = [round(p, 10) for p in out_probs]
            
            drift = 1.0 - sum(out_probs)
            if abs(drift) > 1e-10:
                idx_max = max(range(len(out_probs)), key=out_probs.__getitem__)
                out_probs[idx_max] = round(out_probs[idx_max] + drift, 10)
            
            prob_str = ",".join(f"{p:.10f}" for p in out_probs)
            f.write(f"{item_id}\t[{prob_str}]\n")


def save_perspectivist_tsv(predictions: List[Dict], output_path: str):
    """Save perspectivist predictions in TSV format for Task B"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            item_id = pred['id']
            annotations = pred['annotations']
            annotator_order = pred.get('annotator_order', list(annotations.keys()))
            
            pred_list = [int(annotations[ann_id]) for ann_id in annotator_order]
            preds = ", ".join(str(rating) for rating in pred_list)
            f.write(f"{item_id}\t[{preds}]\n")



def generate_predictions(model: LlamaIronyDetector, test_data: str, annotators_meta: str, 
                        output_dir: str, batch_size: int = 1):
    """Generate predictions for test set"""
    logger.info("Generating test set predictions...")
    
    test_dataset = MPDataset(test_data, annotators_meta, include_labels=False)
    test_samples = [test_dataset[i] for i in range(len(test_dataset))]
    
    soft_predictions = model.predict_soft_labels(test_samples, batch_size)
    save_soft_labels_tsv(soft_predictions, f"{output_dir}/MP_test_soft.tsv")
    
    annotator_predictions = model.predict_annotator_labels(test_samples, test_dataset.annotators_meta, batch_size)
    save_perspectivist_tsv(annotator_predictions, f"{output_dir}/MP_test_pe.tsv")
    
    logger.info(f"Predictions saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Llama Irony Detection for MP Dataset")
    parser.add_argument("--dev_data", default="dataset/MP/MP_dev.json", help="Development data path") 
    parser.add_argument("--test_data", default="dataset/MP/MP_test_clear.json", help="Test data path")
    parser.add_argument("--annotators_meta", default="dataset/MP/MP_annotators_meta.json", help="Annotators metadata path")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf", help="Hugging Face model name")
    parser.add_argument("--output_dir", default="results", help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    logger.info(f"Initializing model: {args.model_name}")
    model = LlamaIronyDetector(args.model_name)

    generate_predictions(model, args.test_data, args.annotators_meta, args.output_dir, args.batch_size)
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 