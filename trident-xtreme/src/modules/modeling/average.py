from pathlib import Path
from typing import Optional, Union

import re
import torch
import gc
import random
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.seed import isolate_rng
from trident import TridentModule
from trident.utils.logging import get_logger
from peft.utils.config import PeftConfig
from peft import LoraConfig
from peft import get_peft_model
from functools import cmp_to_key

log = get_logger(__name__)

original_model = None

def cmp_val_perf(path1: Union[str, Path], path2: Union[str, Path]) -> int:
    path1_ = path1 if isinstance(path1, str) else str(path1)
    path2_ = path2 if isinstance(path2, str) else str(path2)
    
    match1 = re.search(r"epoch=(\d+)_val=([\d.]+).ckpt", path1_)
    match2 = re.search(r"epoch=(\d+)_val=([\d.]+).ckpt", path2_)
    assert match1 is not None
    assert match2 is not None
    val1 = float(match1.group(2))
    val2 = float(match2.group(2))
    return 1 if val1 >= val2 else -1

def get_topk_paths(paths: list[Path], k: int = 1) -> list[Path]:
    sorted_paths = paths.copy()
    sorted_paths.sort(key=cmp_to_key(cmp_val_perf))
    return sorted_paths[-k:]


def get_max_val_path(paths: list[Path]):
    max_val = float("-inf")
    max_val_paths = []

    for path in paths:
        path_str = str(path)
        match = re.search(r"epoch=(\d+)_val=([\d.]+).ckpt", path_str)

        if match:
            _ = int(match.group(1))
            val = float(match.group(2))

            if val > max_val:
                max_val = val
                max_val_paths = [path]
            elif val == max_val:
                max_val_paths.append(path)

    if not max_val_paths:
        raise ValueError("No valid paths found.")
    
    # in case more than one path with same 4 decimal performance
    return random.choice(max_val_paths)


def get_max_epoch_path(paths):
    max_epoch = float("-inf")
    max_epoch_paths = []

    for path in paths:
        path_str = str(path)
        match = re.search(r"epoch=(\d+)_val=([\d.]+).ckpt", path_str)

        if match:
            epoch = int(match.group(1))
            _ = float(match.group(2))

            if epoch > max_epoch:
                max_epoch = epoch
                max_epoch_paths = [path]
            elif epoch == max_epoch:
                max_epoch_paths.append(path)

    if not max_epoch_paths:
        raise ValueError("No valid paths found.")

    assert len(max_epoch_paths) == 1, "Something is wrong with {max_epoch_paths}"
    return max_epoch_paths[0]

def harmonize_state_dict(model: nn.Module, ckpt_state_dict: dict[str, torch.Tensor]):
    """
    Accounts for `._orig_mod` prefix created at torch.compile.
    Since the model is potentially re-instantiated prior to evaluation it get's messy to track whether
    `model` retrained `._orig_mod` or not.
    This function just accounts cleanly for that.
    Very hacky stuff by PyTorch.
    """
    model_state_dict = model.state_dict()

    harmonized_ckpt_state_dict = {}

    for ckpt_key, ckpt_weight in ckpt_state_dict.items():
        new_key = None

        if ckpt_key.startswith("model._orig_mod."):
            stripped_key = ckpt_key[len("model._orig_mod."):]
        elif ckpt_key.startswith("model."):
            stripped_key = ckpt_key[len("model."):]
        else:
            stripped_key = ckpt_key

        if "model." + stripped_key in model_state_dict:
            new_key = "model." + stripped_key
        elif "model._orig_mod." + stripped_key in model_state_dict:
            new_key = "model._orig_mod." + stripped_key
        if new_key is not None:
            harmonized_ckpt_state_dict[new_key] = ckpt_weight
        else:
            print(f"Warning: {ckpt_key} not found in model's state_dict after harmonization")
    return harmonized_ckpt_state_dict

def harmonize_and_load_state_dict(model: nn.Module, ckpt_state_dict: dict[str, torch.Tensor]):
    """
    Accounts for `._orig_mod` prefix created at torch.compile.
    Since the model is potentially re-instantiated prior to evaluation it get's messy to track whether
    `model` retrained `._orig_mod` or not.
    This function just accounts cleanly for that.
    Very hacky stuff by PyTorch.
    """
    model_state_dict = model.state_dict()

    harmonized_ckpt_state_dict = {}

    for ckpt_key, ckpt_weight in ckpt_state_dict.items():
        new_key = None

        if ckpt_key.startswith("model._orig_mod."):
            stripped_key = ckpt_key[len("model._orig_mod."):]
        elif ckpt_key.startswith("model."):
            stripped_key = ckpt_key[len("model."):]
        else:
            stripped_key = ckpt_key

        if "model." + stripped_key in model_state_dict:
            new_key = "model." + stripped_key
        elif "model._orig_mod." + stripped_key in model_state_dict:
            new_key = "model._orig_mod." + stripped_key
        if new_key is not None:
            harmonized_ckpt_state_dict[new_key] = ckpt_weight
        else:
            print(f"Warning: {ckpt_key} not found in model's state_dict after harmonization")
    model.load_state_dict(harmonized_ckpt_state_dict, strict=True)

def avg_ckpts(checkpoints: list[Path]) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoints[0], map_location="cpu")["state_dict"]
    if len(checkpoints) == 1:
        return ckpt
    else:
        denom = 1 / len(checkpoints)
        for key in ckpt.keys():
            if not "int" in str(ckpt[key].dtype):
                ckpt[key] *= denom
            else:
                print(key)
        for path in checkpoints[1:]:
            ckpt_ = torch.load(path, map_location="cpu")["state_dict"]
            for key in ckpt_.keys():
                if not "int" in str(ckpt_[key].dtype):
                    ckpt[key] += denom * ckpt_[key]
            del ckpt_
            gc.collect()
    return ckpt

def average_checkpoints(
    module: LightningModule,
    # ckpt_dir: str,
    ckpt_paths: list[Path],
):
    # ckpt_paths = list(Path(ckpt_dir).glob("*.ckpt"))
    try:
        log.info([x.name for x in ckpt_paths])
    except:
        pass
    denom = 1 / len(ckpt_paths)
    ckpt = torch.load(ckpt_paths.pop(), map_location="cpu")["state_dict"]
    for key in ckpt.keys():
        if not "int" in str(ckpt[key].dtype):
            ckpt[key] *= denom
        else:
            print(key)
    for path in ckpt_paths:
        ckpt_ = torch.load(path, map_location="cpu")["state_dict"]
        for key in ckpt_.keys():
            if not "int" in str(ckpt_[key].dtype):
                ckpt[key] += denom * ckpt_[key]
        del ckpt_
        gc.collect()
    # ckpt = avg_ckpts(ckpt_paths)
    harmonize_and_load_state_dict(module, ckpt)
    # log.info(f"Successfully averaged weights from {ckpt_dir}")

class AverageTridentModule(TridentModule):
    def __init__(
        self,
        avg_ckpts: Optional[str | list[str]] = None,
        avg_best_val_ckpt: bool = False,
        avg_topk: Optional[int] = None,
        avg_last: bool = False,
        avg_wandb_ids: Optional[list[str]] = None,
        clf_seed: int = 42,
        clf_path: Optional[str] = None,
        compile: bool = True,
        lora_cfg: Optional[PeftConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.avg_ckpts = avg_ckpts
        self.avg_topk = avg_topk
        self.clf_seed = clf_seed
        self.compile = compile
        self.lora_cfg = lora_cfg
        self.avg_best_val_ckpt = avg_best_val_ckpt
        self.avg_last = avg_last
        self.avg_wandb_ids = avg_wandb_ids

        if isinstance(clf_path, str) and Path(clf_path).exists():
            self.clf_path = clf_path
        else:
            self.clf_path = None


    def get_diff_vector(
        self,
        base: dict[str, torch.Tensor],
        ckpt: dict[str, torch.Tensor],
        filter_layers: Optional[list[str]] = ["embeddings", "qa_outputs", "classifier"],
    ) -> torch.Tensor:
        diff = []
        for layer, weights in base.items():
            if (filter_layers is None) or (not any(f in layer for f in filter_layers)):
                if (ckpt_weights := ckpt.get(layer)) is not None:
                    diff_ = (ckpt_weights - weights).ravel().abs()
                    self.log(f"diff/{layer}/abs_sum", diff_.sum())
                    self.log(f"diff/{layer}/abs_mean", diff_.mean())
                    diff.append(diff_)
                else:
                    print(layer)
        diff = torch.cat(diff)
        return diff

    def on_train_epoch_end(self):
        if original_model is not None:
            state_dict = {k.replace("_orig_mod.", ""):v.detach().cpu() for k,v in self.model.state_dict().items()}
            diff = self.get_diff_vector(original_model, state_dict)
            self.log("diff/model/abs_sum", diff.sum())
            self.log("diff/model/abs_mean", diff.mean())
            del state_dict

    def on_test_epoch_start(self):
        if self.avg_ckpts is not None:
            if isinstance(self.avg_ckpts, str):
                ckpt_paths = list(Path(self.avg_ckpts).glob("*.ckpt"))
                average_checkpoints(
                    self,
                    ckpt_paths,
                )
            else:
                from itertools import chain
                ckpt_dirs = [list(Path(d).glob("*.ckpt")) for d in self.avg_ckpts]
                if isinstance(self.avg_topk, int):
                    ckpt_dirs = list(chain.from_iterable(ckpt_dirs))
                    ckpt_dirs = get_topk_paths(ckpt_dirs, self.avg_topk)
                    ckpt_dirs = [[p] for p in ckpt_dirs]
                else:
                    if self.avg_best_val_ckpt and not self.avg_last:
                        ckpt_dirs = [[get_max_val_path(d)] for d in ckpt_dirs]
                    if self.avg_last and not self.avg_best_val_ckpt:
                        ckpt_dirs = [[get_max_epoch_path(d)] for d in ckpt_dirs]
                ckpt_paths = list(chain.from_iterable(ckpt_dirs))
                log.info(f"Averaging {ckpt_paths}!")
                # ckpt = avg_ckpts(ckpt_paths)
                # harmonize_and_load_state_dict(self, ckpt)
                average_checkpoints(self, ckpt_paths)
                log.info(f"Successfully averaged weights!")

        if original_model is not None:
            state_dict = {k.replace("_orig_mod.", ""):v.detach().cpu() for k,v in self.model.state_dict().items()}
            diff = self.get_diff_vector(original_model, state_dict)
            self.log("diff/model/abs_sum", diff.sum())
            self.log("diff/model/abs_mean", diff.mean())
            del state_dict

    def setup(self, stage: str):
        super().setup(stage)
        global original_model
        original_model = {k:v.detach().cpu() for k,v in self.model.state_dict().items()}
        # from transformers import AutoModelForSequenceClassification
        # model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=3, return_dict=True)
        # self.model = get_peft_model(model, peft_config)
        # self.model.print_trainable_parameters()

        # self.model
        if stage == "fit":
            if self.lora_cfg is not None:
                peft_config = LoraConfig(**self.lora_cfg)
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
            # head = ["classifier", "qa_outputs"]

            if isinstance(self.clf_path, str):
                state_dict = self.state_dict()
                loaded_state_dict = torch.load(self.clf_path, map_location="cpu")["state_dict"]
                loaded_state_dict = harmonize_state_dict(self, loaded_state_dict)
                loaded_state_dict = {
                    k: v for k, v in loaded_state_dict.items() if any(unfrozen in k for unfrozen in ["classifier", "qa_outputs"])
                }
                state_dict.update(loaded_state_dict)
                print(loaded_state_dict.keys())
                self.load_state_dict(state_dict, strict=True)
                print("Loaded classifier successfully!")
                for name, weights in self.named_parameters():
                    if name in loaded_state_dict:
                        weights.requires_grad = False
                        print(f"Freezing {name}")
            # else:
            #     for h in head:
            #         if (head := getattr(self.model, h, None)) is not None:
            #             log.info(f"Setting {h} seed to {self.clf_seed}")
            #             with isolate_rng():
            #                 torch.manual_seed(self.clf_seed)
            #                 # if hasattr(head, "original_module"):
            #                 #     pass
            #                 #     # in_features, out_features = head.original_module.out_proj.in_features, head.original_module.out_proj.out_features
            #                 #     # new_head = nn.Linear(in_features, out_features).to(self.device)
            #                 #     # self.model.classifier.original_module.out_proj = new_head
            #                 #     # self.model.classifier.modules_to_save.out_proj = new_head
            #                 # else:
            #                 if hasattr(head, "in_features"):
            #                     in_features, out_features = head.in_features, head.out_features
            #                     new_head = nn.Linear(in_features, out_features).to(self.device)
            #                     self.model.classifier = new_head
            #                     setattr(self.model, h, new_head)
            #                 elif hasattr(head, "out_proj"):
            #                     in_features, out_features = head.out_proj.in_features, head.out_proj.out_features
            #                     new_head = nn.Linear(in_features, out_features).to(self.device)
            #                     self.model.classifier.out_proj = new_head
            #                 else:
            #                     raise NotImplementedError("Need to add more classifier types")
            #                 log.info(f"First 20 neurons: {new_head.weight[0, :20].cpu()}")
            #                 global original_model
            #             original_model = {k:v.detach().cpu() for k,v in self.model.state_dict().items()}
            #             break
            #     if self.compile:
            #         self.model = torch.compile(self.model)
