import os
from importlib import import_module
from pathlib import Path

import torch

from .config import get_config


MODULE_PIPELINES = "pipelines"


def get_pipeline(src: str | Path, use_scratch: bool = False):
    src = Path(src)
    assert src.exists(), f"Source '{src}' does not exist."

    match src.suffix:
        case ".toml":
            config = get_config(src.stem)
            params = None
        case ".pth" | ".pt":
            state_dict = torch.load(src, map_location="cpu")
            config = get_config(state_dict["config"])
            params = state_dict["model"]
        case _:
            raise ValueError(f"Supported source types are .toml, .pth(pt).")

    if use_scratch:
        os.environ["DEEP_LEARNING_SCRATCH_BASE_TYPE"] = "scratch"
    else:
        os.environ["DEEP_LEARNING_SCRATCH_BASE_TYPE"] = "fabric"

    task = config.task.name
    cls = getattr(import_module(f"{MODULE_PIPELINES}"), f"{task}Pipeline")
    pipeline = cls(config)
    return pipeline
