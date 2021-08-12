from enum import Enum

from dataclasses import dataclass
from typing import *

__all__ = ['FLOAT_X', 'ArrayInfo', 'SplitInfo', 'ChannelFormat']

FLOAT_X = 'float32'
TInfoClass = TypeVar('TInfoClass')


@dataclass
class BaseInfo(object):

    def copy(self: TInfoClass, **override) -> TInfoClass:
        key_values = {}
        key_values.update(self.__dict__)
        key_values.update(override)
        return self.__class__(**key_values)


@dataclass
class ArrayInfo(BaseInfo):
    """
    Information of an array, typically used to describe a slot of a dataset.
    """

    dtype: str = FLOAT_X
    """Datatype of the array."""

    shape: Optional[List[Optional[int]]] = None
    """Shape of each data in the array (i.e., shape excluding the batch dimension)."""

    is_discrete: bool = False
    """Whether or not the values are discrete?"""

    min_val: Optional[Union[int, float]] = None
    """Minimum value of the data."""

    max_val: Optional[Union[int, float]] = None
    """Maximum value of the data."""

    n_discrete_vals: Optional[int] = None
    """Number of the possible values of the discrete data."""

    bit_depth: Optional[int] = None
    """2 ** (bit_depth - 1) < n_disrete_vals <= 2 ** bit_depth."""

    @property
    def is_numeric(self) -> bool:
        return self.dtype in (
            'int', 'int32', 'int64', 'uint8',
            'float', 'double', 'float16', 'float32', 'float64',
        )

    @property
    def n_categories(self) -> Optional[int]:
        return self.n_discrete_vals

    def require_shape(self, deterministic: bool):
        if self.shape is None:
            raise ValueError('Shape is required to be not None.')
        if deterministic and any(s is None for s in self.shape):
            raise ValueError(f'Shape is required to be deterministic: '
                             f'got {self.shape!r}')

    def require_discrete(self):
        if not self.is_discrete:
            raise ValueError('Slot is required to be discrete.')

    def require_min_max_val(self):
        if self.min_val is None or self.max_val is None:
            raise ValueError('Either `min_val` or `max_val` of the slot is '
                             'None, which is not supported.')


@dataclass
class SplitInfo(BaseInfo):

    data_count: int
    """Number of data in the split."""


class ChannelFormat(str, Enum):

    CHANNEL_LAST = 'channel_last'
    CHANNEL_FIRST = 'channel_first'
