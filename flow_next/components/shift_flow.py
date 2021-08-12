from typing import Optional, Tuple

import tensorkit as tk
from tensorkit.backend import Tensor
from tensorkit.tensor import jit_method, float_scalar_like

__all__ = ['ShiftFlow']


class ShiftFlow(tk.flows.Flow):

    __constants__ = tk.flows.Flow.__constants__ + ('shift',)

    shift: float

    def __init__(self, shift: float, event_ndims: int):
        super().__init__(x_event_ndims=event_ndims, y_event_ndims=event_ndims,
                         explicitly_invertible=True)
        self.shift = shift

    @jit_method
    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        shift = self.shift
        if inverse:
            shift = -shift

        input = input + shift
        if compute_log_det and input_log_det is None:
            input_log_det = float_scalar_like(0., input)

        return input, input_log_det
