import re

import pandas as pd
import wandb
from tqdm.contrib.concurrent import thread_map

import data_defs
from utils import *
from wandb_avg import read_or_get_write

ENTITY = "YOUR_WANDB_ENTITY"

def get_run_data(run):
    params = run.name.split("_")
    kind = params[0].split("=")[1]
    kind_split = kind.split("-")
    weights = kind_split[0]
    approach = kind_split[1]
    num_models = kind_split[2]
    model = params[1].split("=")[1]
    lr = params[2].split("=")[1]
    epochs = params[3].split("=")[1]
    bs = params[4].split("=")[1]
    seed = params[5].split("=")[1]
    scheduler = params[6].split("=")[1]
    ids = run.config["module/avg_wandb_ids"]
    dico = {
        "weights": weights,
        "approach": approach,
        "num_models": num_models,
        "model": model,
        "lr": lr,
        "epochs": epochs,
        "bs": bs,
        "seed": seed,
        "scheduler": scheduler,
        "ids": ids,
    }
    run_columns = set(run.history(samples=1))
    columns = [c for c in run_columns if c.startswith("test_")]
    history = list(run.scan_history(columns))[0]
    for k, v in history.items():
        splitted = k.split("/")
        name_ = f"{splitted[0].lstrip('test_')}_{splitted[2]}"
        dico[name_] = v
    return dico


def get_runs(entity, project, *args, **kwargs) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", {"state": "finished"})
    outputs = thread_map(get_run_data, runs, max_workers=40, desc="Processing runs")
    return pd.DataFrame(outputs)


def read_or_get_write_ra(**kwargs):
    name = kwargs.pop("name")
    assert name is not None
    if exists(name):
        return read(name)
    else:
        df = get_runs(**kwargs)
        pickle_df(df, name)
        return df


def compute_averages(df: pd.DataFrame):
    datasetnames = set()
    pattern = re.compile(r"(\w+)_(\w+)_(\w+)_")
    df_ = df.copy()
    for column_name in df_.columns:
        match = pattern.match(column_name)
        if match:
            datasetnames.add(match.group(1))
    for ckpt in ["last", "oracle-source", "avg-ckpt", "oracle"]:
        cols = [
            c
            for c in df_.columns
            if c.endswith(ckpt) and c.split("_")[0] in datasetnames
        ]
        df_[ckpt] = df_.loc[:, cols].mean(axis=1)
        df_.drop(columns=cols, inplace=True)
    keep_cols = ["kind", "model", "lr", "epochs", "bs", "seed", "scheduler", "id"]
    df_ = df_.loc[
        :,
        keep_cols
        + [
            "last",
            "oracle-source",
            "avg-ckpt",
            "oracle",
            "validation_last",
            "validation_oracle-source",
            "validation_avg-ckpt",
            "validation_oracle",
        ],
    ]
    return df_


def ra_average(df: pd.DataFrame):
    cols = {
        "weights",
        "approach",
        "num_models",
        "model",
        "lr",
        "epochs",
        "bs",
        "seed",
        "scheduler",
        "ids",
    }
    mean_cols = [c for c in df.columns if c not in cols]
    df_ = df.copy()
    df_["mean"] = df_.loc[:, mean_cols].mean(axis=1)
    df_.drop(columns=mean_cols, inplace=True)
    return df_


def get_max_val(df: pd.DataFrame, ids: list[str], ckpt: str) -> float:
    val = -1
    test = -1
    for id_ in ids:
        val_ = df.loc[df["id"] == id_, f"validation_{ckpt}"]
        assert len(val_) == 1
        val_ = val_.iloc[0]
        if val_ > val:
            val = val_
            test = df.loc[df["id"] == id_, ckpt]
            assert len(test) == 1
            test = test.iloc[0]
    return test


def merge(
    zs: pd.DataFrame,
    ra: pd.DataFrame,
    model: str,
    datasets: Optional[list[str]] = None,
    agg=["mean", "std"],
):

    if datasets is None:
        datasetnames = set()
        pattern = re.compile(r"(\w+)_(\w+)_(\w+)")
        for column_name in ra.columns:
            match_ = pattern.match(column_name)
            if match_:
                datasetnames.add(match_.group(1))
    else:
        datasetnames = datasets

    metric = "f1" if any("f1" in c for c in zs.columns) else "acc"
    zs_ = zs.loc[zs.model == model].copy()
    zs_base_cols = [c for c in zs_.columns if not metric in c]
    zs_val_cols = [
        c
        for c in zs_.columns
        if any(c.startswith(d) for d in datasetnames) and metric in c
    ]
    zs_ = zs_.loc[:, zs_base_cols + zs_val_cols]
    zs_avg = compute_averages(zs_)
    # zs_avg = zs_avg.rename(
    #     {}, axis=1
    # )
    # account for rounding
    ra_ = ra.loc[ra.model == model].copy()
    ra_base_cols = [c for c in ra_.columns if not metric in c]
    ra_val_cols = [
        c
        for c in ra_.columns
        if any(c.startswith(d) for d in datasetnames) and metric in c
    ]
    ra_ = ra_.loc[:, ra_base_cols + ra_val_cols]
    for c in ra_.columns:
        if any(c.endswith(c_) for c_ in ("f1", "acc")):
            ra_[c] = ra_.loc[:, c].astype(float).round(4)
    ra_avg = ra_average(ra_)
    # ra_ = ra_.rename(
    #     {
    #     },
    #     axis=1,
    # )
    for c in ("seed", "num_models"):
        ra_avg[c] = ra_avg.loc[:, c].astype(int)

    lines = {}
    for s in ra_avg.seed.unique():
        # if random.random() < 0.5:
        #     continue
        if s == 1:
            continue
        lines[s] = {}
        for j in range(1, 11):
            ids_series = ra_avg.loc[
                (ra_avg["seed"] == s) & (ra_avg["num_models"] == j), "ids"
            ]
            ids_ = ids_series.iloc[0]
            # assert all(ids_series.apply(lambda k: k == ids_))
            assert len(ids_) == j
            lines[s][j] = {"ids": ids_, "num_models": j}
            for ckpt in ["last", "oracle-source", "avg-ckpt", "oracle"]:
                test = get_max_val(zs_avg, ids_, ckpt)
                lines[s][j][ckpt] = test
    for _, line in ra_avg.iterrows():
        if line["seed"] == 1:
            continue
        if line["seed"] in lines:
            d = lines[line["seed"]][line["num_models"]]
            # assert d["ids"] == line["ids"]
            d[line["weights"] + "-" + line["approach"]] = line["mean"]
    lines_ = []
    for s in lines:
        for j in lines[s]:
            lines_.append(lines[s][j])
    df = pd.DataFrame(lines_)
    df = df.rename(
        {
            "last": "S-L",
            "oracle-source": "S-DEV",
            "avg-ckpt": "S-CA",
            "oracle": "S-TDEV",
            "ralast-all": "R-L-all",
            "rasrcdev-all": "R-DEV-all",
            "raca-all": "R-CA-all",
            "ratop5-all": "Soup5",
            "ratop10-all": "Soup10",
        },
        axis=1,
    )
    df = df.reindex(
        columns=[
            "num_models",
            "S-L",
            "S-DEV",
            "S-CA",
            "S-TDEV",
            "R-L-all",
            "R-DEV-all",
            "R-CA-all",
            "Soup5",
        ]
    )
    df = df.loc[
        :,
        [
            "num_models",
            "S-L",
            "S-DEV",
            "S-CA",
            "S-TDEV",
            "R-L-all",
            "R-DEV-all",
            "R-CA-all",
            "Soup5",
        ],
    ]
    (100 * df.groupby("num_models").agg(agg)).round(2)
    stats = df.groupby("num_models").agg(agg)
    if df["S-L"].max() < 1.0001:
        stats = stats * 100
    stats = stats.round(1)

    return stats

nli_ra = read_or_get_write_ra(name="nli_ra", entity=ENTITY, project="ofa-nli-cum")
ner_ra = read_or_get_write_ra(
    name="ner_ra", entity=ENTITY, project="ofa-ner-cum"
)
tyqa_ra = read_or_get_write_ra(
    name="tyqa_ra", entity=ENTITY, project="ofa-tydiqa-cum"
)
# not sure what was wrong here but it just works
tyqa_ra.columns = [
    c.replace("ydiqa", "tydiqa") if "diqa" in c else c for c in tyqa_ra.columns
]
tyqa_ra = tyqa_ra.drop([c for c in tyqa_ra.columns if "exact_match" in c], axis=1)

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
nli_zs = nli_zs.loc[:, (c for c in nli_zs.columns if not "jampatois" in c)]

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

# filter english
nli_ra = nli_ra.loc[:, (c for c in nli_ra.columns if not "xnli_en" in c)]
ner_ra = ner_ra.loc[:, (c for c in ner_ra.columns if not "wikiann_en" in c)]
tyqa_ra = tyqa_ra.loc[:, (c for c in tyqa_ra.columns if not "tydiqa_en" in c)]
nli_zs = nli_zs.loc[:, (c for c in nli_zs.columns if not "xnli_en" in c)]
ner_zs = ner_zs.loc[:, (c for c in ner_zs.columns if not "wikiann_en" in c)]
tyqa_zs = tyqa_zs.loc[:, (c for c in tyqa_zs.columns if not "tydiqa_en" in c)]

# paper outputs
merge(
    nli_zs,
    nli_ra,
    model="xlm-roberta-large",
    datasets=["xnli", "indicxnli"],
    agg="mean",
)
merge(tyqa_zs, tyqa_ra, model="xlm-roberta-large")
merge(ner_zs, ner_ra, model="xlm-roberta-large")
