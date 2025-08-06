#!/bin/bash

# Run LLM evaluation for VariErrNLI and Paraphrase detection
# Updated to use alternative models if Mistral is not accessible

echo "=== LLM Evaluation for VariErrNLI and Paraphrase Detection ==="

# Create output directory
mkdir -p llm_results

# Choose model (modify as needed)
MODEL_NAME="EleutherAI/gpt-j-6b"  # Open alternative to Mistral
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"  # Uncomment if you have access

echo "Using model: $MODEL_NAME"

# 1. Zero-shot evaluation on VariErrNLI development set
echo "Running zero-shot NLI evaluation..."
python mistral_llm_eval.py \
    --data_file dataset/VariErrNLI/VariErrNLI_dev.json \
    --task_type nli \
    --output_file llm_results/varierrnli_dev_zeroshot.json \
    --model_name "$MODEL_NAME"

# 2. Few-shot evaluation on VariErrNLI development set
echo "Running few-shot NLI evaluation..."
python mistral_llm_eval.py \
    --data_file dataset/VariErrNLI/VariErrNLI_dev.json \
    --task_type nli \
    --output_file llm_results/varierrnli_dev_fewshot.json \
    --few_shot \
    --few_shot_file dataset/VariErrNLI/VariErrNLI_train.json \
    --num_shots 3 \
    --model_name "$MODEL_NAME"

# 3. Zero-shot evaluation on Paraphrase development set
echo "Running zero-shot paraphrase evaluation..."
python mistral_llm_eval.py \
    --data_file dataset/Paraphrase/Paraphrase_dev.json \
    --task_type paraphrase \
    --output_file llm_results/paraphrase_dev_zeroshot.json \
    --model_name "$MODEL_NAME"

# 4. Few-shot evaluation on Paraphrase development set
echo "Running few-shot paraphrase evaluation..."
python mistral_llm_eval.py \
    --data_file dataset/Paraphrase/Paraphrase_dev.json \
    --task_type paraphrase \
    --output_file llm_results/paraphrase_dev_fewshot.json \
    --few_shot \
    --few_shot_file dataset/Paraphrase/Paraphrase_train.json \
    --num_shots 5 \
    --model_name "$MODEL_NAME"

echo "=== Evaluation Complete ==="
echo "Results saved in llm_results/"
echo ""

# Generate submission files
echo "Generating submission files..."

python generate_llm_submissions.py \
    --results_file llm_results/varierrnli_dev_fewshot.json \
    --task_type nli \
    --submission_task A \
    --output_file llm_results/VariErrNLI_test_soft.tsv

python generate_llm_submissions.py \
    --results_file llm_results/paraphrase_dev_fewshot.json \
    --task_type paraphrase \
    --submission_task A \
    --output_file llm_results/Paraphrase_test_soft.tsv

echo "Submission files created!"
echo ""
echo "Alternative models to try if current model fails:"
echo "  - microsoft/DialoGPT-large"
echo "  - facebook/opt-6.7b"  
echo "  - microsoft/DialoGPT-medium (smaller/faster)"
echo ""
echo "For test sets (replace dev with test_clear in filenames):"
echo "python mistral_llm_eval.py --data_file dataset/VariErrNLI/VariErrNLI_test_clear.json --task_type nli --output_file llm_results/varierrnli_test_fewshot.json --few_shot --few_shot_file dataset/VariErrNLI/VariErrNLI_train.json --model_name $MODEL_NAME"