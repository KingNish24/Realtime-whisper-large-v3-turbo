"""
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class AliasId(NamedTuple):
    data_ptr: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(
            tensor.data_ptr(),
            tensor.dtype,
            tensor.shape,
            tensor.stride(),
        )
