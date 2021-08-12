import os

import numpy as np
import pytest
from scipy import sparse as sp

from tensorkit import tensor as T
from tensorkit import *

__all__ = [
    'assert_allclose', 'assert_not_allclose', 'assert_equal',  'assert_not_equal',
    'slow_test',
]


def wrap_numpy_testing_assertion_fn(fn):
    def f(t):
        if T.sparse.is_sparse_tensor(t):
            t = T.sparse.to_numpy(t)
        if isinstance(t, (T.Tensor, StochasticTensor)):
            t = T.to_numpy(T.as_tensor(t))
        if isinstance(t, sp.spmatrix):
            t = t.toarray()
        return t

    def wrapper(x, y, **kwargs):
        return fn(f(x), f(y), **kwargs)
    return wrapper


assert_allclose = wrap_numpy_testing_assertion_fn(np.testing.assert_allclose)


@wrap_numpy_testing_assertion_fn
def assert_not_allclose(x, y, err_msg='', **kwargs):
    if np.all(np.allclose(x, y, **kwargs)):
        msg = f'`not allclose(x, y)` not hold'
        if err_msg:
            msg += f': {err_msg}'
        msg += f'\nx = {x}\ny = {y}'
        raise AssertionError(msg)


assert_equal = wrap_numpy_testing_assertion_fn(np.testing.assert_equal)


@wrap_numpy_testing_assertion_fn
def assert_not_equal(x, y, err_msg=''):
    if np.all(np.equal(x, y)):
        msg = f'`x != y` not hold'
        if err_msg:
            msg += f': {err_msg}'
        msg += f'\nx = {x}\ny = {y}'
        raise AssertionError(msg)


# decorate a test that is slow
def slow_test(fn):
    fn = pytest.mark.skipif(
        os.environ.get('FAST_TEST', '0').lower() in ('1', 'on', 'yes', 'true'),
        reason=f'slow test: {fn}'
    )(fn)
    return fn
