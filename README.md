# One For All & All For One: Bypassing Hyperparameter Tuning with Model Averaging For Cross-Lingual Transfer

This is the code for our experiments as part of our paper [One For All & All For One: Bypassing Hyperparameter Tuning with Model Averaging For Cross-Lingual Transfer](https://arxiv.org/abs/2310.10532) accepted to findings of EMNLP 2023.

## Brief Description

Multilingual language models like XLM-R or mT5 enable zero-shot cross-lingual transfer (ZS-XLT), where models fine-tuned in one language perform tasks in another without labeled data. However, current methods for model selection, based on source-language validation, often yield suboptimal performance in target languages. This work introduces an unsupervised evaluation protocol for ZS-XLT that separates performance optimization from hyperparameter tuning. Instead of extensive hyperparameter tuning, we suggest to accumulatively average model snapshots periodically stored in training from different runs. Through experiments on NLI, NER, and TyDiQA-GoldP, we demonstrate that accumulative model averaging enhances ZS-XLT over model selection on source-language validation data and aligns closely with ideal ZS-XLT, as if models were selected on target-language validation instances.

## Preliminaries

You can install the required dependencies in two steps:

1. CD to `trident-xtreme`
2. `conda env create -f environment.yaml`
3. Activate the conda environment `conda env activate tx`
4. Change your working directory to `trident`
5. `pip install -e ./`

Then switch to `trident-xtreme` and

## Experiments

The below presumes you are running a compute infrastructure managed by SLURM.

### Fine-tune classification head for NLI and TyDiQA

1. `conda activate tx`
2. `bash ft-clf.sh $TASK "xlm-roberta-large" 32 0.00002` (where `$TASK` is one of "nli" or "tydiqa" cf. appendix)

### Train all models for all hyperparameters

This requires having fine-tuned classification heads for NLI and TyDiQA.

1. CD to `trident-xtreme`
2. `conda activate tx && python create_clf_symlink.py`
3. `bash batch-experiment.sh` for your `$TASk` of choice (see bash script)


### Accumulative averaging

Requires previous two steps.

1. CD to `trident-xtreme`
2. `bash batch-cumulative_averaging.sh` for your `$TASK` of choice (see bash script)

### Analysis

trident-xtreme logs to `wandb`. The results are then fetched with the `wandb` API and reshaped to clean pandas dataframes.
`wandb_avg.py` can be used to output the Tables 3 and 4 in the appendix (see bottom of file).
`analyse_cumulative_runs.py` can be used to analyse cumulative runs (Table 1 & 2).
The results are loaded from `./analysis/pickles/`.
