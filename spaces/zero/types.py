"""
"""
from __future__ import annotations


from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TypedDict
from typing_extensions import Callable
from typing_extensions import Generic
from typing_extensions import ParamSpec
from typing_extensions import TypeAlias
from typing_extensions import TypeVar


Params = Tuple[Tuple[object, ...], Dict[str, Any]]
Res = TypeVar('Res')
Param = ParamSpec('Param')

class EmptyKwargs(TypedDict):
    pass

@dataclass
class OkResult(Generic[Res]):
    value: Res
@dataclass
class ExceptionResult:
    value: Exception
@dataclass
class AbortedResult:
    pass
@dataclass
class EndResult:
    pass
@dataclass
class GradioQueueEvent:
    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

RegularResQueueResult:   TypeAlias = "OkResult[Res] | ExceptionResult | GradioQueueEvent"
GeneratorResQueueResult: TypeAlias = "OkResult[Res] | ExceptionResult | EndResult | GradioQueueEvent"
YieldQueueResult:        TypeAlias = "OkResult[Res] | ExceptionResult | EndResult | AbortedResult"

Duration:        TypeAlias = "int | timedelta"
DynamicDuration: TypeAlias = "Duration | Callable[Param, Duration] | None"
