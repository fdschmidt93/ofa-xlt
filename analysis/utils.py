import pickle
import random
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import thread_map

import wandb

api = wandb.Api()

ENTITY = "YOUR_WANDB_ENTITY"


def get_run_data(run, params, datasets, metrics, metric_stage, source_dataset):
    outputs = {}
    # run name MUST follow this format: lr=2e-5_epochs=10
    # where `_` splits parameters and `=` splits name and value
    run_name = run.name
    run_params = run_name.split("_")
    for p in run_params:
        name, val = p.split("=")
        # if param found, cast to type and add to outputs dictionary
        if t := params.get(name):
            outputs[name] = t(val)
    print(f"Processing {outputs}")
    for key in params.keys():
        assert key in outputs, f"{key} missing from parameter names: {run.name}"
    outputs["id"] = run.id
    # if not outputs["shots"] in [50]:
    #     continue

    # flat list[str] of [val_xvnli_en/val/acc, test_xvnli_en/val/acc, ...] required to
    # query wandb
    run_columns = set(run.history(samples=1))

    project_metrics_flat = []
    # build a dictionary for easier look up primarily of source_dataset
    #  { str[dataset_name] : { str[metric] : { str[split] : str[full_metric_name] } } }
    project_metrics = {}
    for dataset in datasets:
        if not dataset in project_metrics:
            project_metrics[dataset] = {}
        for metric in metrics:
            if not metric in project_metrics[dataset]:
                project_metrics[dataset][metric] = {}
            for split in ["validation", "test"]:
                # check if validation split exists
                project_metrics[dataset][metric][split] = {}
                for ms in metric_stage:
                    name = f"{split}_{dataset}/{ms}/{metric}"
                    if name not in run_columns:
                        continue
                    project_metrics[dataset][metric][split][ms] = {}
                    project_metrics[dataset][metric][split][ms]["name"] = name
                    project_metrics_flat.append(name)
    if len(metric_stage) == 1:
        run_history = list(run.scan_history(project_metrics_flat))
    else:
        run_history = []
        for m in metric_stage:
            stage_metrics = [p for p in project_metrics_flat if f"/{m}/" in p]
            rh = list(run.scan_history(stage_metrics))
            run_history.extend(rh)
    if run_history:
        # First collect all metrics, then compute oracle, last, source
        #  { str[dataset_name] : { str[metric] : { str[split] : { name: str[full_metric_name], value: float[metric_value] } } }
        # example
        #   {'xgqa_en':
        #     {'acc':
        #       {'validation':
        #          {'name': 'validation_xgqa_en/val/acc','value': array([0.5377, 0.6154, 0.6491, 0.6588, 0.6627]),}
        #        'test':
        #          {'name': 'test_xgqa_en/val/acc', 'value': array([0.4635, 0.5228, 0.5451, 0.5556, 0.5574 }}}}}
        for dataset, dataset_metric in project_metrics.items():
            for _, metric_dico in dataset_metric.items():
                for _, split_dico in metric_dico.items():
                    for _, stage in split_dico.items():
                        run_name = stage["name"]
                        if "/val/" in run_name:
                            stage["value"] = np.array(
                                [
                                    epoch[run_name]
                                    for epoch in run_history
                                    if run_name in epoch
                                ]
                            )
                        else:
                            stage["value"] = run_history[-1][run_name]
        outputs["validation_oracle"] = []
        for dataset, dataset_metric in project_metrics.items():
            for metric_name, metric_dico in dataset_metric.items():
                test = metric_dico["test"]
                if "val" in test and "value" in test["val"]:
                    outputs[f"{dataset}_{metric_name}_last"] = np.round(
                        test["val"]["value"][-1], 4
                    )
                    # outputs[f"{dataset}_{metric_name}_first"] = np.round(
                    #     test["val"]["value"][0], 4
                    # )
                    if validation := metric_dico.get("validation"):
                        outputs[f"{dataset}_{metric_name}_oracle"] = np.round(
                            test["val"]["value"][validation["val"]["value"].argmax()],
                            4,
                        )
                        outputs["validation_oracle"].append(
                            np.round(validation["val"]["value"].max(), 4)
                        )
                    if source_dataset is not None:
                        if source_val := project_metrics[source_dataset][
                            metric_name
                        ].get("validation" if not "tydiqa" in dataset else "test"):
                            if "val" in source_val and "value" in source_val["val"]:
                                source_val_arr = source_val["val"]["value"]
                                outputs[
                                    f"{dataset}_{metric_name}_oracle-source"
                                ] = np.round(
                                    test["val"]["value"][source_val_arr.argmax()],
                                    4,
                                )
                        if dataset == source_dataset:
                            if source_val := project_metrics[source_dataset][
                                metric_name
                            ].get("validation" if not "tydiqa" in dataset else "test"):
                                outputs[f"validation_last"] = source_val["val"][
                                    "value"
                                ][-1]
                                outputs[f"validation_oracle-source"] = source_val[
                                    "val"
                                ]["value"][source_val["val"]["value"].argmax()]
                                outputs[f"validation_avg-ckpt"] = source_val["test"][
                                    "value"
                                ]

                if "test" in test:
                    outputs[f"{dataset}_{metric_name}_avg-ckpt"] = test["test"]["value"]
        outputs["validation_oracle"] = np.array(outputs["validation_oracle"]).mean()
    return outputs


def get_project_metrics(
    project: str,
    datasets: list[str],
    # assumes all datasets share same list[str] of metrics
    metrics: list[str],
    entity: str = ENTITY,
    # list of parameter name and type to be cast to
    params: dict[str, type] = {"seed": str, "lr": float},
    metric_stage: Union[str, list[str]] = "val",
    source_dataset: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Opinionated parser for wandb runs.

    CONVENTION:
        - metrics in wandb are composed of f"{split}_{dataset_name}/{metric_stage}/{metric}", e.g. test_xvnli_en/val/acc
          where split = test, dataset_name = xvnli_en, metric_stage = val, metric = val
        - the names of each run are composed of the most important varying hyperparameters, e.g. "lr=4e-05_seed=42_epochs=10" from which params are extracted

    IMPORTANT:
        - The script assumes that all datasets are homogenous over metrics


    Args:
        project: str, name of wandb project
        datasets: list[str], list[dataset_name] per CONVENTION
        metrics: list[str], e.g. ["acc"], assumed to be homogeneous over datasets
        entity: str, name of wandb entity (user or team)
        params: dict[str, type] dict mapping parameter name to parameter type for parsing
        metric_stage: [TODO:description]
        source_dataset: [TODO:description]

    Returns:
        pd.DataFrame: DataFrame where rows list parsed outputs of runs across columns
    """
    runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
    if run_id:
        runs = [r for r in runs if r.id == run_id]
    if isinstance(metric_stage, str):
        metric_stage = [metric_stage]

    def fn(r):
        return get_run_data(r, params, datasets, metrics, metric_stage, source_dataset)

    # for run in runs:
    #         project_outputs.append(outputs)
    outputs = thread_map(fn, runs, max_workers=40, desc="Processing runs")
    df = pd.DataFrame(outputs)
    return df


def average_performance(df, model, ckpt):
    # Filter the DataFrame to keep only the rows with the specified model
    filtered_df = df[df["model"] == model]

    # Group by the required columns
    grouped_df = filtered_df.groupby(["kind", "lr", "bs"])

    # Extract the dataset names and language codes from the column names
    datasetnames = set()
    langs = set()
    metrics = set()
    pattern = re.compile(r"(\w+)_(\w+)_(\w+)_")

    for column_name in df.columns:
        match = pattern.match(column_name)
        if match:
            datasetnames.add(match.group(1))
            langs.add(match.group(2))
            metrics.add(match.group(3))

    assert len(metrics) == 1
    metric = list(metrics)[0]

    # Calculate the mean performance for each group and performance column
    mean_performance = {}
    for dataset in datasetnames:
        for lang in langs:
            column_name = f"{dataset}_{lang}_{metric}_{ckpt}"
            if column_name in df.columns:
                mean_performance[column_name] = grouped_df[column_name].mean()

    return pd.DataFrame(mean_performance, index=grouped_df.groups.keys())


def average_performance_by_ckpt(df, model):
    ckpts = ["last", "oracle-source", "avg-ckpt"]
    dfs = [average_performance(df, model, ckpt).mean(1) for ckpt in ckpts]
    df_ = pd.concat(dfs, axis=1, keys=ckpts)
    df_.columns = ckpts
    for c in ckpts:
        df_[c] = (100 * df_[c]).round(2)
    return df_


def average_performance_by_dataset(df, model, ckpt):
    # Filter the DataFrame to keep only the rows with the specified model
    filtered_df = df[df["model"] == model]

    # Group by the required columns
    grouped_df = filtered_df.groupby(["kind", "lr", "bs"])

    # Extract the dataset names and language codes from the column names
    datasetnames = set()
    langs = set()
    metrics = set()
    pattern = re.compile(r"(\w+)_(\w{2,3})_(\w+)_")

    for column_name in df.columns:
        match = pattern.match(column_name)
        if match:
            datasetnames.add(match.group(1))
            langs.add(match.group(2))
            metrics.add(match.group(3))
    assert len(metrics) == 1
    metric = list(metrics)[0]
    # Calculate the mean performance for each group and performance column
    mean_performance = {}
    for dataset in datasetnames:
        performance_sum = 0
        lang_count = 0
        for lang in langs:
            column_name = f"{dataset}_{lang}_{metric}_{ckpt}"
            if column_name in df.columns:
                performance_sum += grouped_df[column_name].mean()
                lang_count += 1

        if lang_count > 0:
            mean_performance[dataset] = performance_sum / lang_count

    return (100 * pd.DataFrame(mean_performance, index=grouped_df.groups.keys())).round(
        2
    )


def average_performance_by_ckpt_dataset(df):
    models = ["xlm-roberta-large"]
    ckpts = ["last", "oracle", "oracle-source", "avg-ckpt"]
    dfs = []
    for model in models:
        for ckpt in ckpts:
            df_ = average_performance_by_dataset(df, model, ckpt)
            df_.columns = pd.MultiIndex.from_tuples(
                ((model, ckpt, c) for c in df_.columns)
            )
            dfs.append(df_)
    return pd.concat(dfs, axis=1)


def average_performance_by_model(df):
    models = ["xlm-roberta-large"]
    dfs = [
        average_performance_by_ckpt(df, model)
        for model in models
        if model in df.model.unique()
    ]
    for m, df_ in zip(models, dfs):
        df_.columns = pd.MultiIndex.from_tuples(((m, c) for c in df_.columns))
    return pd.concat(dfs, axis=1)


def pickle_df(df: pd.DataFrame, name: str):
    with open(f"./pickles/{name}.pkl", "wb") as file:
        pickle.dump(df, file)


def read(name: str):
    with open(f"./pickles/{name}.pkl", "rb") as file:
        return pickle.load(file)


def exists(name):
    return Path.cwd().joinpath("pickles", f"{name}.pkl").exists()


def update_batch_size():
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/ofa-tydiqa")
    for run in runs:
        kind, model, lr, epochs, _, seed, scheduler = run.name.split("_")
        accumulate_grad_batches = run.config["trainer/accumulate_grad_batches"]
        batch_size = run.config["datamodule/dataloader_cfg/train/batch_size"]
        bs = accumulate_grad_batches * batch_size
        run.name = f"{kind}_{model}_{lr}_{epochs}_bs={bs}_{seed}_{scheduler}"
        for i, tag in enumerate(run.tags):
            if tag.startswith("bs="):
                run.tags[i] = f"bs={bs}"
                break
        run.update()
