# CSC Sarcasm – Quick Start

This repository contains **two small Python utilities** to train a RoBERTa model on the CSC dataset and to turn the resulting model into Codabench submission files.

---
## 1.  `train_roberta_csc.py`
Fine-tunes a HuggingFace RoBERTa encoder on the train/dev JSON files using **Wasserstein loss**.

### Usage
```bash
python train_roberta_csc.py \
  --train_file dataset/CSC/CSC_train.json \
  --val_file   dataset/CSC/CSC_dev.json   # optional but recommended \
  --model_name roberta-base              # any HF text model \
  --output_dir runs/roberta_csc          # where checkpoints are written \
  --balance                              # optional – class-imbalance sampler
```

Checkpoints:
* `runs/roberta_csc/best_model/`  – best validation Wasserstein distance
* `runs/roberta_csc/last_model/`  – final epoch

---
## 2.  `generate_submission.py`
Creates TSV files that the competition expects.

### Task A (soft-label)
```bash
python generate_submission.py \
  --model_dir runs/roberta_csc/best_model \
  --test_file dataset/CSC/CSC_test_clear.json \
  --task A                # default \
  --num_bins 7            # outputs 7 probs (ratings 0-6) 
```
Produces `CSC_test_soft.tsv`.

### Task B (perspectivist / per-annotator)
```bash
python generate_submission.py \
  --model_dir runs/roberta_csc/best_model \
  --test_file dataset/CSC/CSC_test_clear.json \
  --task B
```
Produces `CSC_test_pe.tsv`.

Zip and submit:
```bash
zip -j res.zip CSC_test_soft.tsv  # add CSC_test_pe.tsv if you generated it
```

---
### Requirements
Install once with:
```bash
pip install -r requirements.txt
```
