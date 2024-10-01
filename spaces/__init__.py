"""
"""

import sys


if sys.version_info.minor < 8: # pragma: no cover
    raise RuntimeError("Importing PySpaces requires Python 3.8+")


# Prevent gradio from importing spaces
if (gr := sys.modules.get('gradio')) is not None: # pragma: no cover
    try:
        gr.Blocks
    except AttributeError:
        raise ImportError


from .zero.decorator import GPU
from .gradio import gradio_auto_wrap
from .gradio import disable_gradio_auto_wrap
from .gradio import enable_gradio_auto_wrap


__all__ = [
    'GPU',
    'gradio_auto_wrap',
    'disable_gradio_auto_wrap',
    'enable_gradio_auto_wrap',
]
