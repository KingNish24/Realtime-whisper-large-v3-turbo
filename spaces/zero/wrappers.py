"""
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from datetime import timedelta
from functools import partial
from functools import wraps
from multiprocessing.context import ForkProcess
from pickle import PicklingError
from queue import Empty
from queue import Queue as ThreadQueue
from threading import Thread
from typing import TYPE_CHECKING
from typing import Callable
from typing import Generator
from typing import Generic
from typing_extensions import assert_never

import psutil

from ..config import Config
from ..utils import debug
from ..utils import drop_params
from ..utils import gradio_request_var
from ..utils import SimpleQueue as Queue
from . import client
from . import torch
from .api import AllowToken
from .api import NvidiaIndex
from .api import NvidiaUUID
from .gradio import GradioPartialContext
from .gradio import get_server_port
from .gradio import patch_gradio_queue
from .gradio import try_process_queue_event
from .tqdm import remove_tqdm_multiprocessing_lock
from .tqdm import tqdm
from .types import * # TODO: Please don't do that


GENERATOR_GLOBAL_TIMEOUT = 20 * 60

SPAWN_PROGRESS_CLEANUP = 0.1
SPAWN_PROGRESS_INIT = 0.1


Process = multiprocessing.get_context('fork').Process
forked = False


class Worker(Generic[Res]):
    process: ForkProcess
    arg_queue: Queue[tuple[Params, GradioPartialContext]]
    res_queue: Queue[Res | None]
    _sentinel: Thread

    def __init__(
        self,
        target: Callable[[
            Queue[tuple[Params, GradioPartialContext]],
            Queue[Res | None],
            AllowToken,
            NvidiaUUID,
            list[int],
        ], None],
        allow_token: str,
        nvidia_uuid: str,
    ):
        self._sentinel = Thread(target=self._close_on_exit, daemon=True)
        self.arg_queue = Queue()
        self.res_queue = Queue()
        debug(f"{self.arg_queue._writer.fileno()=}") # pyright: ignore [reportAttributeAccessIssue]
        debug(f"{self.res_queue._writer.fileno()=}") # pyright: ignore [reportAttributeAccessIssue]
        if (server_port := get_server_port()) is not None:
            fds = [c.fd for c in psutil.Process().connections() if c.laddr.port == server_port]
            debug(f"{fds=}")
        else:
            warnings.warn("Using a ZeroGPU function outside of Gradio caching or request might block the app")
            fds = []
        args = self.arg_queue, self.res_queue, allow_token, nvidia_uuid, fds
        if TYPE_CHECKING:
            target(*args)
        self.process = Process(
            target=target,
            args=args,
            daemon=True,
        )
        self.process.start()
        self._sentinel.start()

    def _close_on_exit(self):
        self.process.join()
        self.arg_queue.close()
        self.res_queue.wlock_release()
        self.res_queue.put(None)


def worker_init(
    res_queue: Queue[RegularResQueueResult | None] | Queue[GeneratorResQueueResult | None],
    allow_token: str,
    nvidia_uuid: str,
    fds: list[int],
) -> None | ExceptionResult:
    # Immediately close file descriptors
    for fd in fds:
        try:
            os.close(fd)
        except Exception as e: # pragma: no cover
            if isinstance(e, OSError) and e.errno == 9:
                continue
            traceback.print_exc()
            return ExceptionResult(e)
    progress = nullcontext()
    if tqdm is not None and Config.zero_gpu_v2:
        progress = tqdm(total=100, desc="ZeroGPU init", file=open(os.devnull, 'w'))
    try: # Unrecoverable init part
        patch_gradio_queue(res_queue)
        with progress as progress:
            current_progress = 0 # Gradio does not support float progress updates
            def update(n: float):
                nonlocal current_progress
                current_progress += n
                if progress is not None:
                    progress.update(round(current_progress * 100) - progress.n)
            client.allow(allow_token)
            update(SPAWN_PROGRESS_CLEANUP)
            torch.unpatch()
            torch.init(nvidia_uuid)
            update(SPAWN_PROGRESS_INIT)
            callback = None
            if (transfer_size := torch.size()) > 0:
                remaining = 1 - (SPAWN_PROGRESS_CLEANUP + SPAWN_PROGRESS_INIT)
                callback = lambda n: update(n * remaining / transfer_size)
            torch.move(callback=callback)
    except Exception as e: # pragma: no cover
        traceback.print_exc()
        return ExceptionResult(e)
    try:
        remove_tqdm_multiprocessing_lock()
    except Exception: # pragma: no cover
        print("Error while trying to remove tqdm mp_lock:")
        traceback.print_exc()


def process_duration(duration: Duration | None):
    if duration is None or isinstance(duration, timedelta):
        return duration
    return timedelta(seconds=duration)


def static_duration(duration: DynamicDuration[Param], *args: Param.args, **kwargs: Param.kwargs):
    if not callable(duration):
        return duration
    return duration(*args, **kwargs)


def regular_function_wrapper(
    task: Callable[Param, Res],
    duration: DynamicDuration[Param],
) -> Callable[Param, Res]:

    import gradio as gr

    request_var = gradio_request_var()
    workers: dict[NvidiaIndex, Worker[RegularResQueueResult[Res]]] = {}
    task_id = id(task)

    @wraps(task)
    def gradio_handler(*args: Param.args, **kwargs: Param.kwargs) -> Res:

        if forked:
            return task(*args, **kwargs)

        request = request_var.get()
        duration_ = static_duration(duration, *args, **kwargs)
        duration_ = process_duration(duration_)
        schedule_response = client.schedule(task_id=task_id, request=request, duration=duration_)
        allow_token = schedule_response.allowToken
        nvidia_index = schedule_response.nvidiaIndex
        nvidia_uuid = schedule_response.nvidiaUUID
        release = partial(client.release, allow_token)

        try:
            worker = workers.pop(nvidia_index)
        except KeyError:
            worker = None

        if worker is not None and worker.process.is_alive() and schedule_response.idle:
            assert worker.arg_queue.empty()
            assert worker.res_queue.empty()
        else:
            worker = Worker(thread_wrapper, allow_token, nvidia_uuid)

        try:
            worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))
        except PicklingError: # TODO: detailed serialization diagnostic
            release(fail=True)
            raise

        while True:
            res = worker.res_queue.get()
            if res is None:
                release(fail=True, allow_404=True)
                raise gr.Error("GPU task aborted")
            if isinstance(res, ExceptionResult):
                release(fail=True)
                raise res.value
            if isinstance(res, OkResult):
                release()
                workers[nvidia_index] = worker
                return res.value
            if isinstance(res, GradioQueueEvent):
                try_process_queue_event(res.method_name, *res.args, **res.kwargs)
                continue
            assert_never(res)


    def thread_wrapper(
        arg_queue: Queue[tuple[Params, GradioPartialContext]],
        res_queue: Queue[RegularResQueueResult[Res] | None],
        allow_token: str,
        nvidia_uuid: str,
        fds: list[int],
    ):
        global forked
        forked = True
        signal.signal(signal.SIGTERM, drop_params(arg_queue.close))
        initialized = False
        while True:
            try:
                (args, kwargs), gradio_context = arg_queue.get()
            except OSError:
                break
            if not initialized:
                if (res := worker_init(
                    res_queue=res_queue,
                    allow_token=allow_token,
                    nvidia_uuid=nvidia_uuid,
                    fds=fds,
                )) is not None:
                    res_queue.put(res)
                    return
                initialized = True
            GradioPartialContext.apply(gradio_context)
            context = copy_context()
            with ThreadPoolExecutor() as executor:
                future = executor.submit(context.run, task, *args, **kwargs) # type: ignore
            try:
                res = future.result()
            except Exception as e:
                traceback.print_exc()
                res = ExceptionResult(e)
            else:
                res = OkResult(res)
            try:
                res_queue.put(res)
            except PicklingError as e:
                res_queue.put(ExceptionResult(e))

    # https://github.com/python/cpython/issues/91002
    if not hasattr(task, '__annotations__'):
        gradio_handler.__annotations__ = {}

    return gradio_handler


def generator_function_wrapper(
    task: Callable[Param, Generator[Res, None, None]],
    duration: DynamicDuration[Param],
) -> Callable[Param, Generator[Res, None, None]]:

    import gradio as gr

    request_var = gradio_request_var()
    workers: dict[NvidiaIndex, Worker[GeneratorResQueueResult[Res]]] = {}
    task_id = id(task)

    @wraps(task)
    def gradio_handler(*args: Param.args, **kwargs: Param.kwargs) -> Generator[Res, None, None]:

        if forked:
            yield from task(*args, **kwargs)
            return

        request = request_var.get()
        duration_ = static_duration(duration, *args, **kwargs)
        duration_ = process_duration(duration_)
        schedule_response = client.schedule(task_id=task_id, request=request, duration=duration_)
        allow_token = schedule_response.allowToken
        nvidia_index = schedule_response.nvidiaIndex
        nvidia_uuid = schedule_response.nvidiaUUID
        release = partial(client.release, allow_token)

        try:
            worker = workers.pop(nvidia_index)
        except KeyError:
            worker = None

        if worker is not None and worker.process.is_alive() and schedule_response.idle:
            assert worker.arg_queue.empty()
            assert worker.res_queue.empty()
        else:
            worker = Worker(thread_wrapper, allow_token, nvidia_uuid)

        try:
            worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))
        except PicklingError: # TODO: detailed serialization diagnostic
            release(fail=True)
            raise

        yield_queue: ThreadQueue[YieldQueueResult[Res]] = ThreadQueue()
        def fill_yield_queue(worker: Worker[GeneratorResQueueResult[Res]]):
            while True:
                res = worker.res_queue.get()
                if res is None:
                    release(fail=True, allow_404=True)
                    yield_queue.put(AbortedResult())
                    return
                if isinstance(res, ExceptionResult):
                    release(fail=True)
                    yield_queue.put(ExceptionResult(res.value))
                    return
                if isinstance(res, EndResult):
                    release()
                    workers[nvidia_index] = worker
                    yield_queue.put(EndResult())
                    return
                if isinstance(res, OkResult):
                    yield_queue.put(OkResult(res.value))
                    continue
                if isinstance(res, GradioQueueEvent): # pragma: no cover (not working properly on Gradio side)
                    try_process_queue_event(res.method_name, *res.args, **res.kwargs)
                    continue
                debug(f"fill_yield_queue: assert_never({res=})")
                assert_never(res)
        from typing_extensions import assert_never
        with ThreadPoolExecutor() as e:
            f = e.submit(copy_context().run, fill_yield_queue, worker)
            f.add_done_callback(lambda _: debug("fill_yield_queue DONE"))
            while True:
                try:
                    res = yield_queue.get(timeout=GENERATOR_GLOBAL_TIMEOUT)
                except Empty: # pragma: no cover
                    debug(f"yield_queue TIMEOUT ({GENERATOR_GLOBAL_TIMEOUT=})")
                    raise
                if isinstance(res, AbortedResult):
                    raise gr.Error("GPU task aborted")
                if isinstance(res, ExceptionResult):
                    raise res.value
                if isinstance(res, EndResult):
                    break
                if isinstance(res, OkResult):
                    yield res.value
                    continue
                debug(f"gradio_handler: assert_never({res=})")
                assert_never(res)


    def thread_wrapper(
        arg_queue: Queue[tuple[Params, GradioPartialContext]],
        res_queue: Queue[GeneratorResQueueResult[Res] | None],
        allow_token: str,
        nvidia_uuid: str,
        fds: list[int],
    ):
        global forked
        forked = True
        signal.signal(signal.SIGTERM, drop_params(arg_queue.close))
        initialized = False
        while True:
            try:
                (args, kwargs), gradio_context = arg_queue.get()
            except OSError:
                break
            if not initialized:
                if (res := worker_init(
                    res_queue=res_queue,
                    allow_token=allow_token,
                    nvidia_uuid=nvidia_uuid,
                    fds=fds,
                )) is not None:
                    res_queue.put(res)
                    return
                initialized = True
            def iterate():
                gen = task(*args, **kwargs) # type: ignore
                while True:
                    try:
                        res = next(gen)
                    except StopIteration:
                        break
                    except Exception as e:
                        res_queue.put(ExceptionResult(e))
                        break
                    try:
                        res_queue.put(OkResult(res))
                    except PicklingError as e:
                        res_queue.put(ExceptionResult(e))
                        break
                    else:
                        continue
            GradioPartialContext.apply(gradio_context)
            with ThreadPoolExecutor() as executor:
                executor.submit(copy_context().run, iterate)
            res_queue.put(EndResult())

    # https://github.com/python/cpython/issues/91002
    if not hasattr(task, '__annotations__'):
        gradio_handler.__annotations__ = {}

    return gradio_handler
