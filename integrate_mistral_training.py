#!/usr/bin/env python3
"""
Integrated Mistral-7B-Instruct-v0.2 Training for Paraphrase Detection
This script combines enhanced transformer training with optimized Mistral prompting.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import the enhanced training components
from train_par_enhanced import EnhancedParDemogModel, EnhancedParDataset, enhanced_collate_fn, enhanced_evaluate
from optimize_mistral_prompts import OptimizedMistralPrompter, load_optimized_few_shot_examples


class IntegratedMistralTrainer:
    """Integrated trainer that combines transformer models with optimized Mistral prompting."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all training components."""
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        
        # Load datasets
        self.train_ds = EnhancedParDataset(
            self.args.train_file, self.args.annot_meta, self.tokenizer, self.args.max_length
        )
        self.val_ds = EnhancedParDataset(
            self.args.val_file, self.args.annot_meta, self.tokenizer, self.args.max_length
        ) if self.args.val_file else None

        print(f"Training samples: {len(self.train_ds)}")
        if self.val_ds:
            print(f"Validation samples: {len(self.val_ds)}")
        print(f"Using demographic fields: {list(self.train_ds.active_field_keys.keys())}")
        print(f"Demographic vocabulary sizes: {self.train_ds.vocab_sizes}")

        # Initialize optimized Mistral prompter
        if self.args.use_mistral:
            try:
                self.mistral_prompter = OptimizedMistralPrompter(
                    model_name=self.args.mistral_model_name,
                    device=self.args.device,
                    use_auth_token=self.args.auth_token
                )
                print("Optimized Mistral prompter initialized successfully!")
                
                # Load few-shot examples
                if self.args.few_shot_file:
                    self.few_shot_examples = load_optimized_few_shot_examples(
                        self.args.few_shot_file, self.args.num_shots
                    )
                    print(f"Loaded {len(self.few_shot_examples)} few-shot examples")
                else:
                    self.few_shot_examples = None
                    
            except Exception as e:
                print(f"Warning: Failed to initialize Mistral prompter: {e}")
                self.args.use_mistral = False
                self.mistral_prompter = None
                self.few_shot_examples = None

        # Create data loaders
        from torch.utils.data import DataLoader, WeightedRandomSampler
        
        def build_sampler(labels):
            counts = Counter(labels)
            total = float(sum(counts.values()))
            num_classes = len(counts)
            class_weights = {c: total / (num_classes * cnt) for c, cnt in counts.items()}
            weights = [class_weights[l] for l in labels]
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        sampler = build_sampler(self.train_ds.labels) if self.args.balance else None
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=enhanced_collate_fn,
        )
        self.val_loader = (
            DataLoader(self.val_ds, batch_size=self.args.batch_size, shuffle=False, collate_fn=enhanced_collate_fn) 
            if self.val_ds else None
        )

        # Create enhanced model
        self.model = EnhancedParDemogModel(
            base_name=self.args.model_name,
            vocab_sizes=self.train_ds.vocab_sizes,
            dem_dim=self.args.dem_dim,
            sbert_dim=self.args.sbert_dim,
            dropout_rate=self.args.dropout_rate,
            num_classes=self.args.num_classes,
            use_mistral=self.args.use_mistral,
            mistral_model_name=self.args.mistral_model_name,
        )
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {"params": [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(grouped_params, lr=self.args.lr)

        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
                           self.optimizer,
                           num_warmup_steps=warmup_steps,
                           num_training_steps=total_steps)

        print(f"Integrated training setup complete!")
        print(f"  - Mistral integration: {'Enabled' if self.args.use_mistral else 'Disabled'}")
        print(f"  - Prompt style: {self.args.prompt_style}")
        print(f"  - Total training steps: {total_steps}")
        print(f"  - Warmup steps: {warmup_steps}")

    def get_mistral_predictions_batch(self, texts: List[str], 
                                     demographic_contexts: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Mistral predictions for a batch of texts."""
        if not self.args.use_mistral or self.mistral_prompter is None:
            return None, None
        
        predictions = []
        confidence_scores = []
        
        for i, text in enumerate(texts):
            # Extract questions from text
            if "[SEP]" in text:
                question1, question2 = text.split("[SEP]", 1)
                question1 = question1.strip()
                question2 = question2.strip()
            else:
                question1 = text
                question2 = ""
            
            # Get demographic context
            demo_context = demographic_contexts[i] if demographic_contexts and i < len(demographic_contexts) else None
            
            # Get optimized Mistral prediction
            rating, confidence = self.mistral_prompter.evaluate_paraphrase_optimized(
                question1, question2, demo_context, self.few_shot_examples, self.args.prompt_style
            )
            
            # Convert rating to class index (-5 to 5 -> 0 to 10)
            class_idx = rating + 5
            
            # Create one-hot encoding
            one_hot = torch.zeros(self.args.num_classes)
            one_hot[class_idx] = 1.0
            
            predictions.append(one_hot)
            confidence_scores.append(confidence)
        
        if predictions:
            predictions_tensor = torch.stack(predictions)
            confidence_tensor = torch.tensor(confidence_scores, dtype=torch.float32)
            return predictions_tensor, confidence_tensor
        
        return None, None

    def train_epoch(self, epoch: int):
        """Train for one epoch with integrated Mistral prompting."""
        self.model.train()
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}"), 1):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get Mistral predictions if enabled
            mistral_preds = None
            mistral_conf = None
            if self.args.use_mistral and self.mistral_prompter is not None:
                mistral_preds, mistral_conf = self.get_mistral_predictions_batch(
                    batch["texts"], 
                    batch.get("demographic_contexts", None)
                )
                if mistral_conf is not None:
                    self.model.mistral_confidence = mistral_conf.to(self.device)
            
            # Prepare demographic inputs
            demographic_inputs = {k: v for k, v in batch.items() 
                                if k.endswith("_ids") and k not in ["input_ids"]}
            
            # Forward pass
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                texts=batch["texts"],
                **demographic_inputs
            )
            
            # Combine with Mistral predictions if available
            if mistral_preds is not None:
                transformer_probs = torch.softmax(logits, dim=-1)
                mistral_probs = mistral_preds.to(self.device)
                
                # Adaptive weighting based on confidence
                alpha = 0.7  # Base weight for transformer
                if mistral_conf is not None:
                    # Adjust weight based on average confidence
                    avg_confidence = torch.mean(mistral_conf).item()
                    alpha = 0.7 + 0.2 * avg_confidence  # Higher confidence = more weight to Mistral
                    alpha = min(0.9, max(0.5, alpha))  # Clamp between 0.5 and 0.9
                
                combined_probs = alpha * transformer_probs + (1 - alpha) * mistral_probs
                logits = torch.log(combined_probs + 1e-8)

            # Loss calculation
            import torch.nn.functional as F
            loss = F.cross_entropy(logits, batch["labels"])

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            step_count += 1
            
            if step % 50 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step_count
                tqdm.write(f"Epoch {epoch} step {step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                if mistral_conf is not None:
                    avg_conf = torch.mean(mistral_conf).item()
                    tqdm.write(f"  Mistral avg confidence: {avg_conf:.3f}, alpha: {alpha:.3f}")

        return epoch_loss / step_count

    def validate_epoch(self, epoch: int):
        """Validate for one epoch with integrated Mistral prompting."""
        if not self.val_loader:
            return None, None, None, None, None
        
        val_dist, predictions, targets, embeddings, labels = enhanced_evaluate(
            self.model, self.val_loader, self.device, self.args.use_mistral
        )
        
        print(f"Validation Manhattan distance after epoch {epoch}: {val_dist:.4f}")
        
        # Generate PCA plot
        self.generate_pca_plot(
            embeddings, labels, self.args.output_dir, 
            model_name="integrated_paraphrase_model", epoch=epoch
        )
        
        return val_dist, predictions, targets, embeddings, labels

    def generate_pca_plot(self, embeddings, labels, output_dir, model_name="model", epoch=None):
        """Generate PCA plot for integrated model."""
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"Generating PCA plot for {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
        
        # Perform PCA
        pca = PCA(n_components=min(50, embeddings.shape[1], len(embeddings)))
        pca_embeddings = pca.fit_transform(embeddings)
        
        # Print variance explained
        variance_explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        
        print(f"PCA Variance Analysis:")
        print(f"  PC1 explains {variance_explained[0]:.3f} of variance")
        print(f"  PC2 explains {variance_explained[1]:.3f} of variance")
        print(f"  First 2 components explain {cumulative_variance[1]:.3f} of total variance")
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: PCA scatter plot
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            axes[0, 0].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                              c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        
        axes[0, 0].set_xlabel(f'PC1 ({variance_explained[0]:.3f} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({variance_explained[1]:.3f} variance)')
        axes[0, 0].set_title(f'Integrated PCA Visualization - {model_name}' + (f' (Epoch {epoch})' if epoch else ''))
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
        plot_filename = f"integrated_pca_plot_{model_name}{epoch_suffix}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved integrated PCA plot → {plot_path}")

    def train(self):
        """Main training loop with integrated Mistral prompting."""
        best_metric = float("inf")
        epochs_no_improve = 0
        best_epoch = 0
        os.makedirs(self.args.output_dir, exist_ok=True)

        train_loss_history = []
        val_dist_history = []
        lr_history = []

        print(f"Starting integrated training with Mistral prompting...")
        print(f"  - Mistral integration: {'Enabled' if self.args.use_mistral else 'Disabled'}")
        print(f"  - Prompt style: {self.args.prompt_style}")
        print(f"  - Few-shot examples: {len(self.few_shot_examples) if self.few_shot_examples else 0}")

        for epoch in range(1, self.args.epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            train_loss_history.append(train_loss)
            lr_history.append(self.scheduler.get_last_lr()[0])
            
            # Validation
            val_results = self.validate_epoch(epoch)
            if val_results[0] is not None:
                val_dist, predictions, targets, embeddings, labels = val_results
                
                if val_dist < best_metric:
                    best_metric = val_dist
                    best_epoch = epoch
                    epochs_no_improve = 0
                    
                    # Save best model
                    save_path = os.path.join(self.args.output_dir, "best_integrated_model")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                    
                    # Save metadata
                    metadata = {
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "use_mistral": self.args.use_mistral,
                        "mistral_model_name": self.args.mistral_model_name,
                        "prompt_style": self.args.prompt_style,
                        "few_shot_examples": len(self.few_shot_examples) if self.few_shot_examples else 0,
                        "training_config": {
                            "lr": self.args.lr,
                            "epochs": self.args.epochs,
                            "batch_size": self.args.batch_size,
                            "model_name": self.args.model_name,
                            "patience": self.args.patience,
                            "dem_dim": self.args.dem_dim,
                            "sbert_dim": self.args.sbert_dim,
                            "dropout_rate": self.args.dropout_rate
                        },
                        "vocab_sizes": self.train_ds.vocab_sizes,
                        "active_fields": list(self.train_ds.active_field_keys.keys())
                    }
                    
                    with open(os.path.join(save_path, "training_metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"Saved best integrated model to {save_path}")
                    
                else:
                    epochs_no_improve += 1
                    
                val_dist_history.append(val_dist)
                
                if epochs_no_improve >= self.args.patience:
                    print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                    break

        # Save final model
        final_path = os.path.join(self.args.output_dir, "final_integrated_model")
        os.makedirs(final_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))

        print(f"Integrated training completed!")
        print(f"Best validation distance: {best_metric:.4f} at epoch {best_epoch}")
        print(f"Mistral integration: {'Enabled' if self.args.use_mistral else 'Disabled'}")
        print(f"Prompt style used: {self.args.prompt_style}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Mistral-7B-Instruct-v0.2 training for paraphrase detection")
    parser.add_argument("--train_file", type=str, default="dataset/Paraphrase/Paraphrase_train.json", help="Path to Paraphrase_train.json")
    parser.add_argument("--val_file", type=str, default="dataset/Paraphrase/Paraphrase_dev.json", help="Path to Paraphrase_dev.json")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="RoBERTa-Large model name")
    parser.add_argument("--output_dir", type=str, default="runs/integrated_outputs_par")
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
    parser.add_argument("--use_mistral", action="store_true", help="Enable Mistral-7B-Instruct-v0.2 integration")
    parser.add_argument("--mistral_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Mistral model name")
    parser.add_argument("--prompt_style", choices=["comprehensive", "concise", "chain_of_thought", "demographic_aware"], 
                       default="comprehensive", help="Prompt style to use")
    parser.add_argument("--few_shot_file", type=str, help="Training file for few-shot examples")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token for gated models")

    args = parser.parse_args()
    
    # Create and run integrated trainer
    trainer = IntegratedMistralTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main() 