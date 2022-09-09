import typing
from dataclasses import dataclass


@dataclass
class TensorShape:
    dtype: str
    shape: typing.Sequence[int]


NamedTensorShapes = typing.Dict[str, TensorShape]
