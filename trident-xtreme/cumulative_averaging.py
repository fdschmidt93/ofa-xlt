import argparse
import pickle
import subprocess
from pathlib import Path

import pandas as pd
import wandb

ENTITY = "YOUR_WANDB_ENTITY"
CWD = Path.cwd()
DATASET = {
    "nli": "validation_xnli_en",
    "ner": "validation_wikiann_en",
    "tydiqa": "test_tydiqa_en",
}
METRIC = {"nli": "acc", "ner": "f1", "tydiqa": "f1"}
KIND = {"avg-ckpt": "ca", "last": "last", "oracle": "srcdev"}


def read_result(name) -> pd.DataFrame:
    with open(f"./results/{name}.pkl", "rb") as file:
        return pickle.load(file)


def sample_runs(entity, project, model, rnd_seed, k):
    import random

    random.seed(rnd_seed)
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    run_params = {}
    for run in runs:
        name = run.name
        id_ = run.id
        kind, m, lr, _, bs, seed, scheduler = name.split("_")
        m = m.split("=")[-1]
        kind = kind.split("=")[-1]
        if m == model and kind == "single":
            lr = lr.split("=")[-1]
            bs = int(bs.split("=")[-1])
            seed = int(seed.split("=")[-1])
            scheduler = float(scheduler.split("=")[-1])
            cfg = (lr, bs)
            if not cfg in run_params:
                run_params[cfg] = []
            run_params[cfg].append((id_, seed))
    sampled = []
    params = list(run_params.keys())
    random.shuffle(params)
    if k > len(params):
        raise ValueError(f"Only {len(params)} runs available")
    for i in range(k):
        sampled_run = random.choice(run_params[params[i]])
        sampled.append(
            {
                "id": sampled_run[0],
                "seed": sampled_run[1],
                "lr": params[i][0],
                "bs": params[i][1],
            }
        )
    return sampled


def get_run(entity, task, model, seed, kind):
    api = wandb.Api()
    runs = api.runs(f"{entity}/ofa-{task}-cum")
    run = [
        r
        for r in runs
        if (kind in r.name)
        and (f"seed={seed}" in r.name)
        and (f"model={model}" in r.name)
    ]
    assert len(run) == 1, f"{model} {seed} {kind} do not uniquely identify run"
    run = run[0]
    return run


def params_to_paths(params, task, model) -> list[str]:
    base_path = Path.cwd() / "logs" / "ofa" / f"{task}" / model
    paths = []
    for p in params:
        lr, bs = p["lr"], p["bs"]
        path = (
            base_path
            / f"batch_size={bs}"
            / f"lr={lr}"
            / "scheduler=0.1"
            / f"seed={p['seed']}"
            / "checkpoints"
        )
        assert path.exists(), f"{path} does not exist"
        paths.append(str(path))
    return paths


def cast(x: list[str]) -> str:
    return repr(x).replace(" ", "")


def run_experiment(
    task: str, model: str, checkpoint: str, sample_seed: int, k: int, topk: int
):
    if topk is not None and topk != "": # cast topk to int
        topk = int(topk)
    kind = KIND[checkpoint]
    if kind == "ca" and isinstance(topk, int) and topk > 1:
        kind = f"top{topk}"
    model_name = model.split("/", maxsplit=1)[-1]
    params = sample_runs(ENTITY, f"ofa-{task}", model_name, sample_seed, k)
    paths = params_to_paths(params, task, model_name)
    paths = {i + 1: v for i, v in enumerate(paths)}
    ids = [v["id"] for v in params]
    for i in range(1, k + 1):
        kind_ = f"kind=ra{kind}"
        run_paths = [paths[j] for j in range(1, i + 1) if j in paths]
        cmd = (
            f"source $HOME/.bashrc && "
            f"conda activate tx && "
            f"env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m trident.run seed={sample_seed} "
            f"experiment=ofa-{task} "
            f"""\"module.avg_ckpts={run_paths}\" """  # passing the folder paths where checkpoints by runs are stored to hydra
            f"""\"+module.avg_wandb_ids={ids[:i]}\" """  # tracking the ids of runs to compare with `max. src-dev` as per paper
            f"logger.wandb.project=ofa-{task}-cum "
            f"{kind_} "
            f"module.model.pretrained_model_name_or_path={model} "
            f"module.clf_path=null "
            f"+train=false "
            f"datamodule.dataloader_cfg.train.batch_size=32 "
            f"test_after_training=true "
        )
        if isinstance(topk, int):
            cmd += f"+module.avg_topk={topk} "  # soup
        if checkpoint in ["oracle", "src-dev"]:
            cmd += f"+module.avg_best_val_ckpt=true "
        if checkpoint in ["last"]:
            cmd += f"+module.avg_last=true "
        cmd = cmd.rstrip()
        subprocess.run(cmd, executable="/usr/bin/bash", shell=True, check=True)


def parse_k(value: str) -> int | list[int]:
    if "," in value:
        return [int(x) for x in value.split(",")]
    else:
        return int(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run an experiment with the given parameters."
    )
    parser.add_argument("task", type=str, help="Task name")
    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("checkpoint", type=str, help="Checkpoint name")
    parser.add_argument(
        "sample_seed",
        type=int,
        help="A single integer or a comma-separated list of integers",
    )
    parser.add_argument(
        "k", type=parse_k, help="A single integer or a comma-separated list of integers"
    )
    parser.add_argument("topk", type=int, help="topk models for SOUP evaluation")

    args = parser.parse_args()
    run_experiment(
        args.task, args.model, args.checkpoint, args.sample_seed, args.k, args.topk
    )


if __name__ == "__main__":
    main()
