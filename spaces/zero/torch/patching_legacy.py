"""
"""
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from functools import partial
from types import SimpleNamespace
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import torch
from torch.utils.weak import WeakTensorKeyDictionary

from ...config import Config
from . import bitsandbytes


# Nvidia A100.80G MIG (drivers 535) / Torch 2.2.0
CUDA_DEVICE_NAME = 'NVIDIA A100-SXM4-80GB MIG 3g.40gb'
CUDA_TOTAL_MEMORY = 42144366592
CUDA_MEM_GET_INFO = (41911451648, CUDA_TOTAL_MEMORY)
CUDA_DEVICE_CAPABILITY = (8, 0)
CUDA_DEVICE_PROPERTIES = SimpleNamespace(name=CUDA_DEVICE_NAME, major=8, minor=0, total_memory=CUDA_TOTAL_MEMORY, multi_processor_count=42)

GENERIC_METHOD_NAMES = [
    'arange',
    'as_tensor',
    'asarray',
    'bartlett_window',
    'blackman_window',
    'empty',
    'empty_like',
    'empty_strided',
    'eye',
    'full',
    'full_like',
    'hamming_window',
    'hann_window',
    'kaiser_window',
    'linspace',
    'logspace',
    'ones',
    'ones_like',
    'rand',
    'rand_like',
    'randint',
    'randint_like',
    'randn',
    'randn_like',
    'randperm',
    'range',
    'sparse_bsc_tensor',
    'sparse_bsr_tensor',
    'sparse_compressed_tensor',
    'sparse_coo_tensor',
    'sparse_csc_tensor',
    'sparse_csr_tensor',
    'tensor',
    'tril_indices',
    'triu_indices',
    'zeros',
    'zeros_like',
]


TO_CUDA = (torch.device('cuda'), None, False, None)

_tensor__deepcopy__ = torch.Tensor.__deepcopy__
_tensor_to         = torch.Tensor.to
_tensor_cuda       = torch.Tensor.cuda
_tensor_cpu        = torch.Tensor.cpu
_torch_generics    = {name: getattr(torch, name) for name in GENERIC_METHOD_NAMES}
_cuda_init         = torch._C._cuda_init
_cuda_available      = torch.cuda.is_available
_cuda_device_count   = torch.cuda.device_count
_cuda_current_device = torch.cuda.current_device
_cuda_mem_get_info   = torch.cuda.mem_get_info
_cuda_get_device_capability   = torch.cuda.get_device_capability
_cuda_get_device_properties   = torch.cuda.get_device_properties
_cuda_get_device_name         = torch.cuda.get_device_name

TensorToArgs = Tuple[Optional[torch.device], Optional[torch.dtype], bool, Optional[torch.memory_format]]

to_ops: dict[torch.Tensor, TensorToArgs] = WeakTensorKeyDictionary() # type: ignore

def _tensor_new_register(*args, **kwargs):
    new_tensor: torch.Tensor = torch._C._TensorBase.__new__(*args, **kwargs)
    if (base_tensor := new_tensor._base) is not None:
        if base_tensor in to_ops:
            to_ops[new_tensor] = to_ops[base_tensor]
    return new_tensor

def _tensor_deepcopy_register(self: torch.Tensor, memo):
    new_tensor = _tensor__deepcopy__(self, memo)
    if isinstance(new_tensor, torch.Tensor):
        if self in to_ops:
            to_ops[new_tensor] = to_ops[self]
    return new_tensor

@property
def _tensor_device_property(self: torch.Tensor):
    if self in to_ops:
        return torch.device(type='cuda', index=0)
    del torch.Tensor.device
    try:
        return self.device
    finally:
        torch.Tensor.device = _tensor_device_property # type: ignore

@property
def _tensor_dtype_property(self: torch.Tensor):
    if self in to_ops:
        if (to_dtype := to_ops[self][1]) is not None:
            return to_dtype
    del torch.Tensor.dtype
    try:
        return self.dtype
    finally:
        torch.Tensor.dtype = _tensor_dtype_property # type: ignore

def _to_op_register(self: torch.Tensor, *args, **kwargs):
    parsed = torch._C._nn._parse_to(*args, **kwargs)
    device, dtype, *_ = parsed
    try:
        to_args = to_ops.pop(self)
    except KeyError:
        to_args = None
    if device is None: # pyright: ignore [reportUnnecessaryComparison]
        if to_args is not None:
            to_ops[self] = (to_args[0], dtype, *to_args[2:])
            return self
        return _tensor_to(self, *args, **kwargs)
    if device.type != 'cuda':
        if to_args is not None:
            if (to_dtype := to_args[1]) is not None:
                kwargs = {'dtype': to_dtype, **kwargs}
        return _tensor_to(self, *args, **kwargs)
    to_ops[self] = parsed
    return self

def _cuda_op_arg_check(device: torch.device | int | str | None) -> bool:
    if device is None:
        return True
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        device = torch.device(device)
    return device.type == 'cuda'

def _cuda_op_register(self: torch.Tensor, device: torch.device | int | str | None = None, **kwargs):
    if not _cuda_op_arg_check(device):
        # Let PyTorch handle the fail
        return _tensor_cuda(self, device, **kwargs)
    to_ops[self] = TO_CUDA
    return self

def _cpu_op_remove(self: torch.Tensor, **kwargs):
    try:
        to_args = to_ops.pop(self)
    except KeyError:
        to_args = None
    if to_args is not None:
        if (to_dtype := to_args[1]) is not None:
            return _tensor_to(self, 'cpu', **{'dtype': to_dtype, **kwargs})
    return _tensor_cpu(self, **kwargs)

def _cuda_init_raise():
    raise RuntimeError(
        "CUDA must not be initialized in the main process "
        "on Spaces with Stateless GPU environment.\n"
        "You can look at this Stacktrace to find out "
        "which part of your code triggered a CUDA init"
    )

def _generic_method_register(name: str, *args: Any, **kwargs: Any):
    try:
        device = torch.device(kwargs.get('device', "cpu"))
    except Exception:
        return _torch_generics[name](*args, **kwargs)
    if device.type != 'cuda':
        return _torch_generics[name](*args, **kwargs)
    tensor = _torch_generics[name](*args, **{**kwargs, 'device': "cpu"})
    to_ops[tensor] = TO_CUDA
    return tensor

def patch():
    torch.Tensor.__deepcopy__ = _tensor_deepcopy_register
    torch.Tensor.__new__      = _tensor_new_register # pyright: ignore [reportAttributeAccessIssue]
    torch.Tensor.to         = _to_op_register   # type: ignore
    torch.Tensor.cuda       = _cuda_op_register # type: ignore
    torch.Tensor.cpu        = _cpu_op_remove # type: ignore
    if Config.zero_patch_torch_device:
        torch.Tensor.device = _tensor_device_property # type: ignore
        torch.Tensor.dtype  = _tensor_dtype_property # pyright: ignore [reportAttributeAccessIssue]
    for name in GENERIC_METHOD_NAMES:
        setattr(torch, name, partial(_generic_method_register, name))
    torch._C._cuda_init     = _cuda_init_raise
    torch.cuda.is_available   = lambda: True
    torch.cuda.device_count   = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.mem_get_info   = lambda *args, **kwargs: CUDA_MEM_GET_INFO
    torch.cuda.get_device_capability = lambda *args, **kwargs: CUDA_DEVICE_CAPABILITY
    torch.cuda.get_device_properties = lambda *args, **kwargs: CUDA_DEVICE_PROPERTIES
    torch.cuda.get_device_name       = lambda *args, **kwargs: CUDA_DEVICE_NAME
    bitsandbytes.patch()

def unpatch():
    torch.Tensor.__deepcopy__ = _tensor__deepcopy__
    with suppress(AttributeError):
        del torch.Tensor.__new__
    torch.Tensor.to         = _tensor_to
    torch.Tensor.cuda       = _tensor_cuda
    torch.Tensor.cpu        = _tensor_cpu
    with suppress(AttributeError):
        del torch.Tensor.device
    with suppress(AttributeError):
        del torch.Tensor.dtype
    for name in GENERIC_METHOD_NAMES:
        setattr(torch, name, _torch_generics[name])
    torch._C._cuda_init     = _cuda_init
    torch.cuda.is_available   = _cuda_available
    torch.cuda.device_count   = _cuda_device_count
    torch.cuda.current_device = _cuda_current_device
    torch.cuda.mem_get_info   = _cuda_mem_get_info
    torch.cuda.get_device_capability = _cuda_get_device_capability
    torch.cuda.get_device_properties = _cuda_get_device_properties
    torch.cuda.get_device_name       = _cuda_get_device_name
    bitsandbytes.unpatch()

def pack():
    pass

def init(nvidia_uuid: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = nvidia_uuid
    torch.Tensor([0]).cuda() # CUDA init

def size():
    return 0

def move(callback: Callable[[int]] | None = None):
    for op in to_ops.items():
        tensor, parsed_args = op
        _, dtype, _, memory_format = parsed_args
        tensor.data = _tensor_to(tensor,
            device='cuda',
            dtype=dtype,
            memory_format=memory_format,
        ) # type: ignore
    bitsandbytes.move()
    torch.cuda.synchronize()

def is_in_bad_fork():
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context('fork')) as e:
        f = e.submit(torch.cuda._is_in_bad_fork)
        return f.result()

def disable_cuda_intercept():
    torch.Tensor.to   = _tensor_to
    torch.Tensor.cuda = _tensor_cuda
