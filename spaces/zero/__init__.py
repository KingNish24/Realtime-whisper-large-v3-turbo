"""
"""

from pathlib import Path

from ..config import Config


if Config.zero_gpu:

    from . import gradio
    from . import torch

    if torch.is_in_bad_fork():
        raise RuntimeError(
            "CUDA has been initialized before importing the `spaces` package"
        )

    torch.patch()
    gradio.one_launch(torch.pack)
    Path(Config.zerogpu_offload_dir).mkdir(parents=True, exist_ok=True)
