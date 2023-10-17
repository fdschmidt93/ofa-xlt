#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate tx

# task
# model
# checkpoint
# k
# topk
python ./cumulative_averaging.py $1 $2 $3 $4 $5 $6
