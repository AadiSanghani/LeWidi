#!/bin/bash
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=3  
#SBATCH --mem=64000M       
#SBATCH --time=2-03:00     


SOURCEDIR=~/aadi-projects/sarcasm-model

source ~/aadi-projects/sarcasm-model/.venv/bin/activate
pip install -r $SOURCEDIR/requirements.txt

python $SOURCEDIR/sarcasm_transformer.py