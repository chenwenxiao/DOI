from unittest import TestCase

import numpy as np
import pytest
from tensorkit import tensor as T

from tests._helper import assert_allclose
from utils.data import ArrayInfo
from utils.data.mappers import *
from utils.evaluation import dequantized_bpd


class EvaluationTestCase(TestCase):

    def test_dequantized_bpd(self):
        ll = np.random.normal(size=[11, 12])

        ##
        info = ArrayInfo(shape=[2, 3, 4], min_val=-1., max_val=3.,
                         is_discrete=False, n_discrete_vals=128, bit_depth=7)
        expected = (24 * np.log(128 / 4) - ll) / (np.log(2) * 24)
        assert_allclose(dequantized_bpd(ll, info), expected)
        assert_allclose(dequantized_bpd(T.as_tensor(ll), info), expected)

        ##
        info = ArrayInfo(shape=[2, 3, 4], min_val=0, max_val=255,
                         is_discrete=True, n_discrete_vals=256, bit_depth=8)
        m = ArrayMapperList([
            Dequantize(),
            ScaleToRange(-0.5, 0.5),
        ])
        info = m.fit(info)

        expected = (24 * np.log(256) - ll) / (np.log(2) * 24)
        assert_allclose(dequantized_bpd(ll, info), expected)
        assert_allclose(dequantized_bpd(T.as_tensor(ll), info), expected)

        ##
        with pytest.raises(ValueError,
                           match='Only dequantized discrete values are '
                                 'supported'):
            _ = dequantized_bpd(ll, ArrayInfo(
                is_discrete=True, n_discrete_vals=256, shape=[2, 3, 4],
                min_val=0, max_val=255
            ))

        with pytest.raises(ValueError,
                           match='Only dequantized discrete values are '
                                 'supported'):
            _ = dequantized_bpd(ll, ArrayInfo(
                is_discrete=False, shape=[2, 3, 4], min_val=0, max_val=255))
