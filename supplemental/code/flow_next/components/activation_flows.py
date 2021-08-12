import math
from typing import *

import tensorkit as tk
from tensorkit.tensor import (Tensor, jit, jit_method, shape, get_device,
                              float_scalar_like, greater_equal, where,
                              reduce_sum, int_range)
from tensorkit.tensor.nn import leaky_relu, LEAKY_RELU_DEFAULT_SLOPE

__all__ = [
    'ActivationFlow', 'LeakyReLUFlow',
]


@jit
def prod_shape(shape: List[int]) -> int:
    r = 1
    for s in shape:
        r *= s
    return r


class ActivationFlow(tk.flows.Flow):

    def __init__(self, event_ndims: int):
        super().__init__(
            x_event_ndims=event_ndims,
            y_event_ndims=event_ndims,
            explicitly_invertible=True,
        )


class LeakyReLUFlow(ActivationFlow):

    __constants__ = ActivationFlow.__constants__ + (
        'negative_slope', '_log_negative_slope',
        '_inv_negative_slope',
    )

    negative_slope: float
    _log_negative_slope: float
    _inv_negative_slope: float

    def __init__(self,
                 event_ndims: int,
                 negative_slope: float = LEAKY_RELU_DEFAULT_SLOPE):
        super().__init__(event_ndims=event_ndims)
        self.negative_slope = negative_slope
        self._log_negative_slope = math.log(negative_slope)
        self._inv_negative_slope = 1. / negative_slope

    @jit_method
    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            slope = self._inv_negative_slope
            log_det = -self._log_negative_slope
        else:
            slope = self.negative_slope
            log_det = self._log_negative_slope

        input = leaky_relu(input, slope)

        if compute_log_det:
            log_det = where(
                greater_equal(input, float_scalar_like(0., input)),
                float_scalar_like(0., input),
                float_scalar_like(log_det, input),
            )
            log_det = reduce_sum(log_det, int_range(-self.x_event_ndims, 0))
            if input_log_det is not None:
                input_log_det = input_log_det + log_det
            else:
                input_log_det = log_det

        return input, input_log_det
