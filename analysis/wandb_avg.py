import re

import pandas as pd

import data_defs
from utils import *

ENTITY = "YOUR_WANDB_ENTITY"

# Model variants as per paper:
# avg-ckpt -> CA (checkpoint averaging)
# oracle-source -> SRC-DEV
# oracle -> TRG-DEV

def read_or_get_write(**kwargs):
    name = kwargs.pop("name")
    assert name is not None
    if exists(name):
        return read(name)
    else:
        df = get_project_metrics(**kwargs)
        pickle_df(df, name)
        return df


# ZERO-SHOT
zs_params = {
    "model": str,
    "kind": str,
    "seed": int,
    "epochs": int,
    "lr": float,
    "bs": int,
    "scheduler": float,
}

nli_zs = read_or_get_write(
    name="nli_zs",
    entity=ENTITY,
    project="ofa-nli",
    datasets=data_defs.xnli + data_defs.indicxnli,
    metrics=["acc"],
    metric_stage=["val", "test"],
    params=zs_params,
    source_dataset="xnli_en",
)

ner_zs = read_or_get_write(
    name="ner_zs",
    entity=ENTITY,
    project="ofa-ner",
    datasets=data_defs.wikiann + data_defs.masakhaner + data_defs.masakhaner2,
    metrics=["f1"],
    metric_stage=["val", "test"],
    params=zs_params,
    source_dataset="wikiann_en",
)

tyqa_zs = read_or_get_write(
    name="tydiqa_zs",
    entity=ENTITY,
    project="ofa-tydiqa",
    datasets=data_defs.tydiqa,
    metrics=["f1"],
    metric_stage=["val", "test"],
    params=zs_params,
    source_dataset="tydiqa_en",
)


def average_performance_by_dataset_ckpt(df, model):
    # Filter the DataFrame to keep only the rows with the specified model
    filtered_df = df[df["model"] == model]

    # Group by the required columns
    grouped_df = filtered_df.groupby(["kind", "lr", "bs"])

    # Extract the dataset names and language codes from the column names
    datasetnames = set()
    langs = set()
    metrics = set()
    ckpts = [
        "last",
        "oracle-source",
        "avg-ckpt",
        "oracle",
    ]
    pattern = re.compile(r"(\w+)_(\w{2,3})_(\w+)_")

    for column_name in df.columns:
        match = pattern.match(column_name)
        if match:
            datasetnames.add(match.group(1))
            langs.add(match.group(2))
            metrics.add(match.group(3))
    assert len(metrics) == 1
    metric = list(metrics)[0]

    # Calculate the mean performance for each group, dataset, and checkpoint
    mean_performance = {}
    for dataset in datasetnames:
        mean_performance[dataset] = {}
        for ckpt in ckpts:
            performance_sum = 0
            lang_count = 0
            for lang in langs:
                column_name = f"{dataset}_{lang}_{metric}_{ckpt}"
                if column_name in df.columns:
                    performance_sum += grouped_df[column_name].mean()
                    lang_count += 1

            if lang_count > 0:
                mean_performance[dataset][ckpt] = performance_sum / lang_count

    # Convert the nested dictionary to a multi-level DataFrame
    mean_performance_df = pd.DataFrame.from_dict(
        {
            (dataset, ckpt): mean_performance[dataset][ckpt]
            for dataset in mean_performance
            for ckpt in mean_performance[dataset]
        },
        orient="index",
    ).T
    mean_performance_df.index = grouped_df.groups.keys()
    mean_performance_df = (100 * mean_performance_df).round(2)

    return mean_performance_df


def average_performance_by_model_dataset_ckpt(df):
    models = ["xlm-roberta-large"]
    dfs = []
    for model in models:
        if model in df.model.unique():
            df_ = average_performance_by_dataset_ckpt(df, model)
            df_.columns = pd.MultiIndex.from_tuples(
                ((model, c1, c2) for c1, c2 in df_.columns)
            )
            dfs.append(df_)
    return pd.concat(dfs, axis=1)


def average_performance_by_ckpt(df, model):
    """Print nicely formatted dataframe of results for performance by model variant"""
    filtered_df = df.loc[df.model == model]
    ckpts = ["last", "oracle-source", "avg-ckpt", "oracle"]
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
    for ckpt in ckpts:
        columns = [
            c
            for c in df.columns
            if any(c.startswith(d) for d in datasetnames) and c.endswith(ckpt)
        ]
        filtered_df[ckpt] = filtered_df.loc[:, columns].mean(1)
    out = filtered_df.groupby(["kind", "lr", "bs"]).agg(["mean", "std"])[
        ["last", "oracle-source", "avg-ckpt", "oracle"]
    ]
    return (100 * out).round(1)


def average_val_by_ckpt(df, model):
    """Print nicely formatted dataframe of results for validation performance by kind"""
    filtered_df = df.loc[df.model == model]
    validation_columns = [
        "validation_last",
        "validation_oracle-source",
        "validation_avg-ckpt",
        "validation_oracle",
    ]
    return (
        100
        * filtered_df.groupby(["kind", "lr", "bs"])[validation_columns].agg(
            ["mean", "std"]
        )
    ).round(1)


nli_zs_excl_en = nli_zs.loc[:, (c for c in nli_zs.columns if not "xnli_en" in c)]
average_performance_by_ckpt(nli_zs_excl_en, "xlm-roberta-large")

tyqa_zs_excl_en = tyqa_zs.loc[:, (c for c in tyqa_zs.columns if not "tydiqa_en" in c)]
(average_performance_by_ckpt(tyqa_zs_excl_en, "xlm-roberta-large") / 100).round(1)

ner_zs_excl_en = ner_zs.loc[:, (c for c in ner_zs.columns if not "wikiann_en" in c)]
average_performance_by_ckpt(ner_zs_excl_en, "xlm-roberta-large")
