"""
"""
from __future__ import annotations

import time

import ctypes
import os
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from queue import Queue
from typing import Callable

from ...utils import debug

import torch
from typing_extensions import TypeAlias


PAGE_SIZE = 4096
TOTAL_MEMORY = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
VM_MAX_SIZE = min(2**38, TOTAL_MEMORY // 2)

BUFFER_SIZE = 64 * 2**20
BUFFER_COUNT = 2


TensorWithSizes: TypeAlias = 'tuple[torch.Tensor, int, int]'

@dataclass
class ZeroGPUTensorPack:
    base_dir: str
    batches: list[list[TensorWithSizes]]
    big_tensors: list[TensorWithSizes]
    fakes: dict[torch.Tensor, list[torch.Tensor]]
    total_size: int
    def path(self):
        return f'{self.base_dir}/{id(self)}'
    def __del__(self):
        try:
            os.remove(self.path())
        except FileNotFoundError: # pragma: no cover
            pass


def write(fd: int, tensor: torch.Tensor):
    clone = torch.empty_like(tensor)
    size = clone.untyped_storage().size() # pyright: ignore [reportAttributeAccessIssue]
    buffer = torch.UntypedStorage(VM_MAX_SIZE)
    buffer_ptr = buffer.data_ptr()
    offset = -buffer_ptr % PAGE_SIZE
    padding = -size % PAGE_SIZE
    clone.set_(buffer[offset:offset+size], 0, clone.shape, clone.stride()) # pyright: ignore [reportArgumentType]
    clone.copy_(tensor)
    mv = memoryview((ctypes.c_char * (size+padding)).from_address(buffer_ptr+offset))
    written_bytes = 0
    while written_bytes < size:
        written_bytes += os.write(fd, mv[written_bytes:])


def pack_tensors(
    tensors: set[torch.Tensor],
    fakes: dict[torch.Tensor, list[torch.Tensor]],
    offload_dir: str,
    callback: Callable[[int]] | None = None,
):

    callback = (lambda bytes: None) if callback is None else callback

    batches: list[list[TensorWithSizes]] = []
    big_tensors: list[TensorWithSizes] = []

    tensors_with_sizes: list[tuple[torch.Tensor, int, int]] = []
    for tensor in tensors:
        size = tensor.numel() * tensor.element_size()
        aligned_size = size + (-size % PAGE_SIZE)
        tensors_with_sizes += [(tensor, size, aligned_size)]

    current_batch, current_size = [], 0
    for (tensor, size, aligned_size) in sorted(tensors_with_sizes, key=lambda item: item[2]):
        if aligned_size > BUFFER_SIZE:
            big_tensors += [(tensor, size, aligned_size)]
            continue
        current_size += aligned_size
        if current_size > BUFFER_SIZE:
            batches += [current_batch]
            current_batch, current_size = [(tensor, size, aligned_size)], aligned_size
        else:
            current_batch += [(tensor, size, aligned_size)]

    if current_batch:
        batches += [current_batch]

    get_meta = {tensor: torch.empty_like(tensor) for tensor in tensors}
    batches_meta = [[(get_meta[tensor], size, asize) for tensor, size, asize in batch] for batch in batches]
    big_tensors_meta = [(get_meta[tensor], size, asize) for tensor, size, asize in big_tensors]
    fakes_meta = {get_meta[tensor]: fake_list for tensor, fake_list in fakes.items()}

    pack = ZeroGPUTensorPack(
        base_dir=offload_dir,
        batches=batches_meta,
        big_tensors=big_tensors_meta,
        fakes=fakes_meta,
        total_size=sum([size for _, size, _ in tensors_with_sizes]),
    )

    fd = os.open(pack.path(), os.O_CREAT | os.O_WRONLY | os.O_DIRECT)
    try:
        total_asize = sum([aligned_size for batch in batches for *_, aligned_size in batch])
        total_asize += sum([aligned_size for *_, aligned_size in big_tensors])
        if total_asize > 0:
            os.posix_fallocate(fd, 0, total_asize)
            for batch in batches:
                for tensor, size, _ in batch:
                    write(fd, tensor)
                    callback(size)
            for tensor, size, _ in big_tensors:
                write(fd, tensor)
                callback(size)
        return pack
    finally:
        os.close(fd)


def pack_to_cuda(pack: ZeroGPUTensorPack, callback: Callable[[int]] | None = None):

    callback = (lambda bytes: None) if callback is None else callback

    free_buffers: Queue[torch.Tensor] = Queue()
    read_buffers: Queue[torch.Tensor] = Queue()

    for _ in range(BUFFER_COUNT):
        free_buffers.put(torch.ByteTensor(BUFFER_SIZE).pin_memory())

    def read(fd: int, buffer: torch.Tensor, size: int):
        mv = memoryview((ctypes.c_char * size).from_address(buffer.data_ptr()))
        read_bytes = 0
        while read_bytes < size:
            read_bytes += os.readv(fd, [mv[read_bytes:]])

    def disk_to_pin(fd: int):
        for batch in pack.batches:
            buffer = free_buffers.get()
            batch_size = sum([aligned_size for *_, aligned_size in batch])
            read(fd, buffer, batch_size)
            read_buffers.put(buffer)
        for *_, aligned_size in pack.big_tensors:
            read_bytes = 0
            while read_bytes < aligned_size:
                buffer = free_buffers.get()
                read_size = min(BUFFER_SIZE, aligned_size - read_bytes)
                read(fd, buffer, read_size)
                read_buffers.put(buffer)
                read_bytes += read_size

    def pin_to_cuda():
        total_duration_in_callback = 0
        for batch in pack.batches:
            buffer = read_buffers.get()
            offset = 0
            cuda_storages = []
            for tensor, size, aligned_size in batch:
                cuda_storages += [buffer[offset:offset+size].cuda(non_blocking=True)]
                offset += aligned_size
            torch.cuda.synchronize()
            free_buffers.put(buffer)
            batch_total_size = 0
            for (tensor, size, _), cuda_storage in zip(batch, cuda_storages):
                cuda_tensor = torch.tensor([], dtype=tensor.dtype, device='cuda')
                cuda_tensor = cuda_tensor.set_(cuda_storage.untyped_storage(), 0, tensor.shape, tensor.stride())
                for fake in pack.fakes[tensor]:
                    fake.data = cuda_tensor
                batch_total_size += size
            t0 = time.perf_counter()
            callback(batch_total_size)
            total_duration_in_callback += time.perf_counter() - t0
        for tensor, size, _ in pack.big_tensors:
            cuda_storage = torch.empty(size, dtype=torch.uint8, device='cuda')
            offset = 0
            while offset < size:
                buffer = read_buffers.get()
                read_size = min(BUFFER_SIZE, size - offset)
                cuda_storage[offset:offset+read_size] = buffer[:read_size]
                offset += read_size
                torch.cuda.synchronize() # Probably not needed
                free_buffers.put(buffer)
                t0 = time.perf_counter()
                callback(read_size)
                total_duration_in_callback += time.perf_counter() - t0
            cuda_tensor = torch.tensor([], dtype=tensor.dtype, device='cuda')
            cuda_tensor = cuda_tensor.set_(cuda_storage.untyped_storage(), 0, tensor.shape, tensor.stride())
            for fake in pack.fakes[tensor]:
                fake.data = cuda_tensor

        debug(f"{total_duration_in_callback=}")

    with ThreadPoolExecutor(2) as e:
        fd = os.open(pack.path(), os.O_RDONLY | os.O_DIRECT)
        try:
            futures = [
                e.submit(copy_context().run, disk_to_pin, fd),
                e.submit(copy_context().run, pin_to_cuda),
            ]
            for future in as_completed(futures):
                future.result()
        finally:
            os.close(fd)
