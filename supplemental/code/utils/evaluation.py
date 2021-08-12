from typing import *

import numpy as np
from tensorkit import tensor as T

from .data import ArrayInfo

__all__ = ['dequantized_bpd']


def dequantized_bpd(log_likelihood: Union[np.ndarray, T.Tensor],
                    value_info: ArrayInfo):
    value_info.require_shape(deterministic=True)
    value_info.require_min_max_val()
    if value_info.is_discrete or value_info.n_discrete_vals is None:
        raise ValueError(f'Only dequantized discrete values are supported.')

    value_size = 1
    for s in value_info.shape:
        value_size *= s
    bpd = value_size * (
            np.log(value_info.n_discrete_vals) -
            np.log(value_info.max_val - value_info.min_val)) - log_likelihood
    bpd = bpd / (np.log(2) * value_size)

    return bpd

