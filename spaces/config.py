"""
"""
from __future__ import annotations

import os
from pathlib import Path

from .utils import boolean


ZEROGPU_OFFLOAD_DIR_DEFAULT = str(Path.home() / '.zerogpu' / 'tensors')


class Settings:
    def __init__(self):
        self.zero_gpu = boolean(
            os.getenv('SPACES_ZERO_GPU'))
        self.zero_device_api_url = (
            os.getenv('SPACES_ZERO_DEVICE_API_URL'))
        self.gradio_auto_wrap = boolean(
            os.getenv('SPACES_GRADIO_AUTO_WRAP'))
        self.zero_patch_torch_device = boolean(
            os.getenv('ZERO_GPU_PATCH_TORCH_DEVICE'))
        self.zero_gpu_v2 = boolean(
            os.getenv('ZEROGPU_V2'))
        self.zerogpu_offload_dir = (
            os.getenv('ZEROGPU_OFFLOAD_DIR', ZEROGPU_OFFLOAD_DIR_DEFAULT))


Config = Settings()


if Config.zero_gpu:
    assert Config.zero_device_api_url is not None, (
        'SPACES_ZERO_DEVICE_API_URL env must be set '
        'on ZeroGPU Spaces (identified by SPACES_ZERO_GPU=true)'
    )
