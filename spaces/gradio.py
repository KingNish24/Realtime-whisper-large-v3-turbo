"""
"""
from __future__ import annotations

from typing import Callable
from typing import Generator
from typing import TypeVar
from typing import overload
from typing_extensions import ParamSpec

from .config import Config
from .zero.decorator import GPU


Param = ParamSpec('Param')
Res = TypeVar('Res')


gradio_auto_wrap_enabled = Config.gradio_auto_wrap


def disable_gradio_auto_wrap():
    global gradio_auto_wrap_enabled
    gradio_auto_wrap_enabled = False

def enable_gradio_auto_wrap():
    global gradio_auto_wrap_enabled
    gradio_auto_wrap_enabled = True


@overload
def gradio_auto_wrap(
    task:
     Callable[Param, Res],
) -> Callable[Param, Res]:
    ...
@overload
def gradio_auto_wrap(
    task:
     None,
) -> None:
    ...
def gradio_auto_wrap(
    task:
      Callable[Param, Res]
    | None,
) -> (Callable[Param, Res]
    | None):
    """
    """
    if not gradio_auto_wrap_enabled:
        return task
    if not callable(task):
        return task
    return GPU(task) # type: ignore
