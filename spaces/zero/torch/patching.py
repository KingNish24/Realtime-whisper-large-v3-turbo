"""
"""
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import gc
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from types import SimpleNamespace
from typing import Any
from typing import Callable

import torch
from torch.overrides import TorchFunctionMode
from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.weak import WeakTensorKeyDictionary

from ...config import Config
from ...utils import malloc_trim
from ..tqdm import tqdm
from . import bitsandbytes
from .packing import ZeroGPUTensorPack
from .packing import pack_tensors
from .packing import pack_to_cuda
from .types import AliasId


# Nvidia A100.80G MIG (drivers 535) / Torch 2.2.0
CUDA_DEVICE_NAME = 'NVIDIA A100-SXM4-80GB MIG 3g.40gb'
CUDA_TOTAL_MEMORY = 42144366592
CUDA_MEM_GET_INFO = (41911451648, CUDA_TOTAL_MEMORY)
CUDA_DEVICE_CAPABILITY = (8, 0)
CUDA_DEVICE_PROPERTIES = SimpleNamespace(name=CUDA_DEVICE_NAME, major=8, minor=0, total_memory=CUDA_TOTAL_MEMORY, multi_processor_count=42)

OPS_INPUTS_CHECK_NO_RETURN = (
    torch.Tensor.equal,
)

OPS_INPUT_CHECK_SELF_RETURN = (
    torch.Tensor.set_, # probably never dispatched
    torch.ops.aten.set_.source_Tensor, # pyright: ignore [reportAttributeAccessIssue]
)

OFFLOADED_ERROR_MESSAGE = "Cannot apply function {} on disk-offloaded Tensor {}"

_tensor_make_subclass = torch.Tensor._make_subclass
_asarray           = torch.asarray
_cuda_init         = torch._C._cuda_init
_cuda_exchange_device = torch.cuda._exchange_device
_cuda_available      = torch.cuda.is_available
_cuda_device_count   = torch.cuda.device_count
_cuda_current_device = torch.cuda.current_device
_cuda_mem_get_info   = torch.cuda.mem_get_info
_cuda_get_device_capability   = torch.cuda.get_device_capability
_cuda_get_device_properties   = torch.cuda.get_device_properties
_cuda_get_device_name         = torch.cuda.get_device_name

# PyTorch 2.3
_cuda_maybe_exchange_device = getattr(torch.cuda, '_maybe_exchange_device', None)


cuda_aliases: dict[torch.Tensor, torch.Tensor | None] = WeakTensorKeyDictionary() # pyright: ignore [reportAssignmentType]

tensor_packs: list[ZeroGPUTensorPack] = []

class ZeroGPUTensor(torch.Tensor):
    pass

def empty_fake(tensor: torch.Tensor):
    fake = torch.empty_like(tensor, requires_grad=tensor.requires_grad)
    if fake.__class__ != tensor.__class__:
        fake = _tensor_make_subclass(tensor.__class__, fake, require_grad=tensor.requires_grad) # pyright: ignore [reportArgumentType]
    return fake

class ZeroGPUFunctionMode(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs: dict[str, Any] | None = None):

        kwargs = {} if kwargs is None else kwargs

        if func == torch._C._nn._parse_to:
            return func(*args, **kwargs)

        # Redispatch: tensor.cuda() -> tensor.to(device='cuda')
        if func == torch.Tensor.cuda or func == torch.Tensor.cpu:
            memory_format = kwargs.get('memory_format')
            return self.__torch_function__(torch.Tensor.to, types, (args[0],), {
                'device': 'cuda' if func == torch.Tensor.cuda else 'cpu',
                **({'memory_format': memory_format} if memory_format is not None else {}),
            })

        # Redispatch: tensor.to('cuda') -> tensor.to(device='cuda')
        if func == torch.Tensor.to and len(args) > 1:
            device, dtype, _, memory_format = torch._C._nn._parse_to(*args[1:], **kwargs)
            return self.__torch_function__(torch.Tensor.to, types, (args[0],), {
                'device': device,
                'dtype': dtype,
                'memory_format': memory_format,
            })

        if func == torch.Tensor.data.__set__: # pyright: ignore [reportAttributeAccessIssue]
            self, target = args
            if target in cuda_aliases:
                if (target_original := cuda_aliases[target]) is None:
                    raise Exception(OFFLOADED_ERROR_MESSAGE.format(resolve_name(func), target))
                original = empty_fake(self)
                original.data = target_original
                cuda_aliases[self] = original
            elif self in cuda_aliases:
                del cuda_aliases[self]
            self.data = target
            return

        if func == torch.Tensor.device.__get__:
            tensor, = args
            if tensor in cuda_aliases:
                return torch.device('cuda', index=0)

        elif func == torch.Tensor.__repr__:
            tensor, = args
            if tensor in cuda_aliases:
                if (original := cuda_aliases[tensor]) is None:
                    original = tensor.to('meta')
                original_class = original.__class__
                original.__class__ = ZeroGPUTensor
                try:
                    return func(original, **kwargs)
                finally:
                    original.__class__ = original_class

        elif func == torch.Tensor.untyped_storage:
            tensor, = args
            if tensor in cuda_aliases:
                if (original := cuda_aliases[tensor]) is None:
                    raise Exception(OFFLOADED_ERROR_MESSAGE.format(resolve_name(func), tensor))
                res = func(original, **kwargs)
                res._zerogpu = True
                return res

        cuda: bool | None = None

        # Handle device kwarg
        if (device := kwargs.get('device')) is not None:
            device = torch.device(device)
            if device.type == 'cuda':
                kwargs['device'] = torch.device('cpu')
                cuda = True
            else:
                cuda = False

        # Swap fake inputs with original data
        swapped = {}
        inputs_are_cuda = set()
        def swap(tensor: torch.Tensor):
            nonlocal inputs_are_cuda
            if tensor not in cuda_aliases:
                inputs_are_cuda |= {False}
                return tensor
            if (original := cuda_aliases[tensor]) is None:
                raise Exception(OFFLOADED_ERROR_MESSAGE.format(resolve_name(func), tensor))
            swapped[original] = tensor
            inputs_are_cuda |= {True}
            return original
        args_ = tree_map_only(torch.Tensor, swap, args)
        kwargs_ = tree_map_only(torch.Tensor, swap, kwargs)
        if inputs_are_cuda == {True}:
            if cuda is not False:
                cuda = True

        res = func(*args_, **kwargs_)

        # Re-generate swapped fakes in case of mutation
        for original, fake in swapped.items():
            fake.data = empty_fake(original)

        # Special case for Tensor indexing where only 'self' matters
        if func in {
            torch.ops.aten.index.Tensor, # pyright: ignore [reportAttributeAccessIssue]
            torch.Tensor.__getitem__, # PyTorch 2.4+
        }:
            self = args[0]
            cuda = self in cuda_aliases
            inputs_are_cuda = {cuda}

        # Emulate device check
        if isinstance(res, torch.Tensor) or func in OPS_INPUTS_CHECK_NO_RETURN:
            self = None
            if len(args_) >= 1 and isinstance(args_[0], torch.Tensor):
                self = args_[0]
            # Only raise if func does not return its first input (Tensor.copy_)
            if res is not self or func in OPS_INPUT_CHECK_SELF_RETURN:
                if inputs_are_cuda == {True, False}:
                    raise RuntimeError(
                        "Expected all tensors to be on the same device, "
                        "but found at least two devices, cuda:0 (ZeroGPU) and cpu!"
                    )

        # Register output
        def register(tensor: torch.Tensor):
            if tensor in swapped and cuda is not False:
                return swapped[tensor]
            if cuda is not True:
                return tensor
            fake = empty_fake(tensor)
            cuda_aliases[fake] = tensor
            return fake

        return tree_map_only(torch.Tensor, register, res)

# When enabling DispatchMode, some aten ops are dispatched to FunctionMode
# We are using it for aten.alias.default and aten.set_.source_Tensor
class DefaultDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs: dict[str, Any] | None = None):
        return func(*args, **(kwargs or {}))


function_mode = ZeroGPUFunctionMode()
dispatch_mode = DefaultDispatchMode()


def _untyped_storage_new_register(*args, **kwargs):
    cuda = False
    if (device := kwargs.get('device')) is not None and device.type == 'cuda':
        cuda = True
        del kwargs['device']
    storage = torch._C.StorageBase.__new__(*args, **kwargs)
    if cuda:
        storage._zerogpu = True
    return storage

@property
def _untyped_storage_device(self):
    if hasattr(self, '_zerogpu'):
        return torch.device('cuda', index=0)
    return torch._C.StorageBase.device.__get__(self) # pyright: ignore [reportAttributeAccessIssue]

# Force dispatch
def _tensor_make_subclass_function_mode(*args, **kwargs):
    with torch._C.DisableTorchFunction():
        return function_mode.__torch_function__(_tensor_make_subclass, (), args=args, kwargs=kwargs)
def _asarray_function_mode(*args, **kwargs):
    with torch._C.DisableTorchFunction():
        return function_mode.__torch_function__(_asarray, (), args=args, kwargs=kwargs)

def _cuda_init_raise():
    raise RuntimeError(
        "CUDA must not be initialized in the main process "
        "on Spaces with Stateless GPU environment.\n"
        "You can look at this Stacktrace to find out "
        "which part of your code triggered a CUDA init"
    )

def _cuda_dummy_exchange_device(device):
    assert device in {-1, 0}
    return device

def patch():
    function_mode.__enter__()
    dispatch_mode.__enter__()
    # TODO: only patch bellow methods on current Thread to be consistent with TorchModes
    # (or hijack threading.Thread.__init__ to force Modes on all threads)
    torch.Tensor._make_subclass = _tensor_make_subclass_function_mode # pyright: ignore [reportAttributeAccessIssue]
    torch.UntypedStorage.__new__ = _untyped_storage_new_register
    torch.UntypedStorage.device  = _untyped_storage_device # pyright: ignore [reportAttributeAccessIssue]
    torch.asarray           = _asarray_function_mode
    torch._C._cuda_init     = _cuda_init_raise
    torch.cuda._exchange_device = _cuda_dummy_exchange_device
    torch.cuda.is_available   = lambda: True
    torch.cuda.device_count   = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.mem_get_info   = lambda *args, **kwargs: CUDA_MEM_GET_INFO
    torch.cuda.get_device_capability = lambda *args, **kwargs: CUDA_DEVICE_CAPABILITY
    torch.cuda.get_device_properties = lambda *args, **kwargs: CUDA_DEVICE_PROPERTIES
    torch.cuda.get_device_name       = lambda *args, **kwargs: CUDA_DEVICE_NAME
    # PyTorch 2.3
    if _cuda_maybe_exchange_device is not None: # pragma: no cover
        setattr(torch.cuda, '_maybe_exchange_device', _cuda_dummy_exchange_device)
    bitsandbytes.patch()

def unpatch():
    try:
        dispatch_mode.__exit__(None, None, None)
        function_mode.__exit__(None, None, None)
    except RuntimeError:
        pass # patch() and unpatch() called from != threads
    torch.Tensor._make_subclass = _tensor_make_subclass
    torch.UntypedStorage.__new__ = torch._C.StorageBase.__new__
    torch.UntypedStorage.device  = torch._C.StorageBase.device # pyright: ignore [reportAttributeAccessIssue]
    torch.asarray           = _asarray
    torch._C._cuda_init     = _cuda_init
    torch.cuda._exchange_device = _cuda_exchange_device
    torch.cuda.is_available   = _cuda_available
    torch.cuda.device_count   = _cuda_device_count
    torch.cuda.current_device = _cuda_current_device
    torch.cuda.mem_get_info   = _cuda_mem_get_info
    torch.cuda.get_device_capability = _cuda_get_device_capability
    torch.cuda.get_device_properties = _cuda_get_device_properties
    torch.cuda.get_device_name       = _cuda_get_device_name
    # PyTorch 2.3
    if _cuda_maybe_exchange_device is not None: # pragma: no cover
        setattr(torch.cuda, '_maybe_exchange_device', _cuda_exchange_device)
    bitsandbytes.unpatch()


def _total_unpacked_size():
    tensors = [tensor for tensor in cuda_aliases.values() if tensor is not None]
    deduped = {AliasId.from_tensor(tensor): tensor for tensor in tensors}
    return sum([tensor.numel() * tensor.element_size() for tensor in deduped.values()])


def _pack(offload_dir: str):
    # Pack to disk
    originals: set[torch.Tensor] = set()
    originals_dedup: dict[AliasId, torch.Tensor] = {}
    fakes: dict[torch.Tensor, list[torch.Tensor]] = defaultdict(list)
    for fake, original in cuda_aliases.items():
        # TODO filter-out sparse Tensors
        if original is not None:
            original_id = AliasId.from_tensor(original)
            if original_id not in originals_dedup:
                originals_dedup[original_id] = original
                originals |= {original}
            fakes[originals_dedup[original_id]] += [fake]
    progress = tqdm(
        total=_total_unpacked_size(),
        unit='B',
        unit_scale=True,
        desc="ZeroGPU tensors packing",
    ) if tqdm is not None else nullcontext()
    with progress as progress:
        update = progress.update if progress is not None else lambda _: None
        pack = pack_tensors(originals, fakes, offload_dir, callback=update)
    tensor_packs.append(pack)
    # Free memory
    for fake_list in fakes.values():
        for fake in fake_list:
            cuda_aliases[fake] = None

def pack():
    _pack(Config.zerogpu_offload_dir)
    gc.collect()
    malloc_trim()

def init(nvidia_uuid: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = nvidia_uuid
    torch.Tensor([0]).cuda()

def size():
    return _total_unpacked_size() + sum([pack.total_size for pack in tensor_packs])

def _move(callback: Callable[[int]] | None = None):
    callback = callback if callback is not None else lambda _: None
    # CPU -> CUDA
    moved: dict[AliasId, torch.Tensor] = {}
    for fake, original in cuda_aliases.items():
        if original is not None:
            original_id = AliasId.from_tensor(original)
            if original_id not in moved:
                moved[original_id] = original.cuda()
                callback(fake.numel() * fake.element_size())
    for fake, original in cuda_aliases.items():
        if original is not None:
            fake.data = moved[AliasId.from_tensor(original)]
    # Disk -> CUDA
    for tensor_pack in tensor_packs:
        pack_to_cuda(tensor_pack, callback=callback)
    bitsandbytes.move()

def move(callback: Callable[[int]] | None = None):
    callback = callback if callback is not None else lambda _: None
    with ThreadPoolExecutor(1) as e:
        e.submit(copy_context().run, _move, callback=callback).result()
    torch.cuda.synchronize()

def is_in_bad_fork():
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context('fork')) as e:
        f = e.submit(torch.cuda._is_in_bad_fork)
        return f.result()
