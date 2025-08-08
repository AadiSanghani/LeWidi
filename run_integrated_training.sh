#!/bin/bash

# Integrated Mistral-7B-Instruct-v0.2 Training for Paraphrase Detection
# This script demonstrates different configurations for maximum efficiency

echo "=== Integrated Mistral Training for Paraphrase Detection ==="

# Create output directories
mkdir -p runs/integrated_outputs_par
mkdir -p runs/enhanced_outputs_par
mkdir -p runs/optimized_prompts

# Configuration 1: Comprehensive prompting with few-shot learning
echo "Running Configuration 1: Comprehensive prompting with few-shot learning..."
python integrate_mistral_training.py \
    --train_file dataset/Paraphrase/Paraphrase_train.json \
    --val_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_dir runs/integrated_outputs_par/comprehensive_fewshot \
    --use_mistral \
    --prompt_style comprehensive \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 5 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-5 \
    --balance

# Configuration 2: Chain-of-thought prompting for detailed reasoning
echo "Running Configuration 2: Chain-of-thought prompting..."
python integrate_mistral_training.py \
    --train_file dataset/Paraphrase/Paraphrase_train.json \
    --val_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_dir runs/integrated_outputs_par/chain_of_thought \
    --use_mistral \
    --prompt_style chain_of_thought \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 3 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-5 \
    --balance

# Configuration 3: Demographic-aware prompting
echo "Running Configuration 3: Demographic-aware prompting..."
python integrate_mistral_training.py \
    --train_file dataset/Paraphrase/Paraphrase_train.json \
    --val_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_dir runs/integrated_outputs_par/demographic_aware \
    --use_mistral \
    --prompt_style demographic_aware \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 4 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-5 \
    --balance

# Configuration 4: Concise prompting for faster inference
echo "Running Configuration 4: Concise prompting for speed..."
python integrate_mistral_training.py \
    --train_file dataset/Paraphrase/Paraphrase_train.json \
    --val_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_dir runs/integrated_outputs_par/concise \
    --use_mistral \
    --prompt_style concise \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-5 \
    --balance

# Configuration 5: Enhanced training without Mistral (baseline)
echo "Running Configuration 5: Enhanced training baseline (no Mistral)..."
python integrate_mistral_training.py \
    --train_file dataset/Paraphrase/Paraphrase_train.json \
    --val_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_dir runs/integrated_outputs_par/baseline \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-5 \
    --balance

# Run optimized prompt evaluation
echo "Running optimized prompt evaluation..."
python optimize_mistral_prompts.py \
    --data_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_file runs/optimized_prompts/comprehensive_results.json \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 5 \
    --prompt_style comprehensive

python optimize_mistral_prompts.py \
    --data_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_file runs/optimized_prompts/chain_of_thought_results.json \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 3 \
    --prompt_style chain_of_thought

python optimize_mistral_prompts.py \
    --data_file dataset/Paraphrase/Paraphrase_dev.json \
    --output_file runs/optimized_prompts/demographic_aware_results.json \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 4 \
    --prompt_style demographic_aware

# Generate comparison report
echo "Generating comparison report..."
python generate_comparison_report.py \
    --results_dir runs/integrated_outputs_par \
    --output_file runs/comparison_report.json

echo "=== Training Complete ==="
echo "Results saved in:"
echo "  - runs/integrated_outputs_par/"
echo "  - runs/optimized_prompts/"
echo ""
echo "Best configurations to try:"
echo "  1. Comprehensive prompting with 5 few-shot examples"
echo "  2. Chain-of-thought prompting for detailed reasoning"
echo "  3. Demographic-aware prompting for contextual understanding"
echo "  4. Concise prompting for faster inference"
echo "  5. Enhanced baseline without Mistral integration" 