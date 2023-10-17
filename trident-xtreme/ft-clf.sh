#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate tx # make sure to install environment from environment.yaml

# Fine-tuning a classification head per (Schmidt et. al, 2023) https://arxiv.org/abs/2305.16834
# Initial fine-tuning is on seed 0, other runs on grids are ran on random_sample(1..1e6)
# 1 task: one of "nli" "tydiqa" "ner"
# 2 model: "xlm-roberta-large"
# 3 batch_size: 32
# 4 lr: 0.00002
env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m trident.run \
    seed=0 \
    experiment=ofa-${1} \
    logger.wandb.project=ofa-${1}-head \
    module.model.pretrained_model_name_or_path=${2} \
    datamodule.dataloader_cfg.train.batch_size=${3} \
    module.optimizer.lr=${4} \
    'hydra.run.dir="logs/ofa-head/${task}/${rm_upto_backslash:${module.model.pretrained_model_name_or_path}}/batch_size=${datamodule.dataloader_cfg.train.batch_size}/lr=${module.optimizer.lr}/scheduler=${module.scheduler.num_warmup_steps}/seed=${seed}"' \
    'module.clf_path=null' # no classification head instantiated
