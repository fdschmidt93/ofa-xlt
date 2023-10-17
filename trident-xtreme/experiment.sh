#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate tx

# 1 task
# 2 model
# 3 batch_size
# 4 lr
env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m trident.run \
    'seed=${rnd:1,1000000}' \
    experiment=ofa-${1} \
    module.model.pretrained_model_name_or_path=${2} \
    datamodule.dataloader_cfg.train.batch_size=${3} \
    module.optimizer.lr=${4} \
    'hydra.run.dir="logs/ofa/${task}/${rm_upto_backslash:${module.model.pretrained_model_name_or_path}}/batch_size=${datamodule.dataloader_cfg.train.batch_size}/lr=${module.optimizer.lr}/scheduler=${module.scheduler.num_warmup_steps}/seed=${seed}"' \
    'module.clf_path="logs/ofa-head/${task}/${rm_upto_backslash:${module.model.pretrained_model_name_or_path}}/batch_size=32/lr=2e-05/scheduler=0.1/seed=0/checkpoints/last.ckpt"'
