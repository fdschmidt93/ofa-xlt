import importlib.util
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import random

random_num = None

def random_number(min_val: int, max_val: int) -> int:
    global random_num
    if random_num is None:
        random_num = random.randint(min_val, max_val)
    return random_num

def remove_text_before_backslash(input_string: str) -> str:
    """Removes the text up to and including the last forward slash ('/') character in the input string.
    
    Args:
        input_string (str): The input string to process.
    
    Returns:
        str: The substring of the input string that follows the last forward slash ('/') character, or the entire
            input string if no forward slashes are present.
    """
    parts = input_string.split("/")
    if len(parts) == 1:
        return input_string
    else:
        return parts[-1]

def batch_accumulate(total_batch_size, batch_size):
    return total_batch_size // batch_size

OmegaConf.register_new_resolver("rm_upto_backslash", remove_text_before_backslash)
OmegaConf.register_new_resolver("rnd", random_number)
OmegaConf.register_new_resolver("batch_accumulate", batch_accumulate)

cwd = Path.cwd()


@hydra.main(
    version_base="1.3",
    config_path=str(cwd.joinpath("configs/")),
    config_name="config.yaml",
)
def main(cfg: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    cwd_train_path = cwd.joinpath("src", "train.py")
    if cwd_train_path.exists():
        spec = importlib.util.spec_from_file_location("src.train", cwd_train_path)
        assert spec is not None
        train_mod = importlib.util.module_from_spec(spec)
        sys.modules["src.train"] = train_mod
        if exec_module := getattr(spec.loader, "exec_module"):
            exec_module(train_mod)
        train = train_mod.train
    else:
        from trident.train import train
    from trident.utils.runner import extras, print_config

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    extras(cfg)
    # Init lightning datamodule

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # Train model
    return train(cfg)


if __name__ == "__main__":
    main()
