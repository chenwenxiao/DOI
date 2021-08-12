import itertools
from functools import partial
from unittest import TestCase

import numpy as np
import pytest
from skimage import transform
from tensorkit import tensor as T

from utils.data import *
from utils.data.mappers import *
from tests._helper import slow_test


def standard_image_mapper_check(ctx, image_shape, cls, fn, is_invertible=False,
                                is_supported=None, comparer=None,
                                check_out_info=None):
    if is_supported is None:
        is_supported = lambda *args: True
    if comparer is None:
        comparer = lambda a, b, max_val, dtype: \
            np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-6)

    def f(batch_shape, channel_last, the_channel_last, n_channels, max_val, dtype):
        value_shape = list(image_shape)

        # generate the input image to the mapper
        if n_channels is not None:
            if the_channel_last:
                value_shape = value_shape + [n_channels]
            else:
                value_shape = [n_channels] + value_shape

        x = np.random.randint(0, 255, dtype=np.int32, size=batch_shape + value_shape)
        if max_val != 255:
            x = (x / float(max_val))
        x = x.astype(dtype)

        # generate the input to the `fn`
        if n_channels is not None:
            x0 = x if the_channel_last else np.transpose(
                x, list(range(len(x.shape) - 3)) + [-2, -1, -3])
        else:
            x0 = x.reshape(x.shape + (1,))

        # get the output and the answer
        m = cls(channel_last=channel_last)
        info = ArrayInfo(shape=value_shape, dtype=dtype, min_val=0,
                         max_val=max_val, is_discrete=True, n_discrete_vals=255)
        out_info = m.fit(info)
        ctx.assertTrue(m.fitted)
        ctx.assertEqual(m.input_info, info)
        ctx.assertAlmostEqual(out_info.max_val, max_val)
        ctx.assertEqual(out_info.dtype, dtype)
        if check_out_info is not None:
            check_out_info(out_info, channel_last, the_channel_last, n_channels, max_val, dtype)

        y = m.transform(x)
        ctx.assertEqual(y.dtype, np.dtype(dtype))

        y0 = fn(x0, n_channels, max_val, dtype)
        if n_channels is not None:
            y0 = y0 if the_channel_last else np.transpose(
                y0, list(range(len(x.shape) - 3)) + [-1, -3, -2])
        else:
            y0 = y0.reshape(y0.shape[:-1])

        comparer(y, y0, max_val, dtype)

        # compare the inverted result
        if is_invertible:
            x2 = m.inverse_transform(y)
            comparer(x2, x, max_val, dtype)

    for batch_shape, channel_last, n_channels, max_val, dtype in itertools.product(
            ([], [11], [11, 12]),
            (None, True, False),
            (1, 3),
            (1, 255),
            ('float32', 'int32', 'uint8')):
        if n_channels is None and channel_last is not None:
            continue
        if not is_supported(channel_last, n_channels, max_val, dtype):
            continue

        if n_channels is None:
            f(batch_shape, None, None, n_channels, max_val, dtype)
        elif channel_last is None:
            f(batch_shape, channel_last, True, n_channels, max_val, dtype)
            f(batch_shape, channel_last, False, n_channels, max_val, dtype)
        else:
            f(batch_shape, channel_last, channel_last, n_channels, max_val, dtype)

    for channel_last in (None, True, False):
        with pytest.raises(ValueError, match='Invalid shape'):
            m = cls(channel_last=channel_last)
            m.fit(ArrayInfo(shape=image_shape, is_discrete=True, min_val=0,
                            max_val=255, dtype='uint8'))

    with pytest.raises(ValueError, match='`channel_last` cannot be determined '
                                         'automatically'):
        m = cls(channel_last=None)
        m.fit(ArrayInfo(shape=[1, 25, 3], is_discrete=True, min_val=0,
                        max_val=255, dtype='uint8'))

    with pytest.raises(ValueError, match='Invalid shape .* for `channel_last`'):
        m = cls(channel_last=False)
        m.fit(ArrayInfo(shape=[21, 25, 3], is_discrete=True, min_val=0,
                        max_val=255, dtype='uint8'))

    with pytest.raises(ValueError, match='Invalid shape .* for `channel_last`'):
        m = cls(channel_last=True)
        m.fit(ArrayInfo(shape=[1, 25, 21], is_discrete=True, min_val=0,
                        max_val=255, dtype='uint8'))


class MappersTestCase(TestCase):

    def test_ArrayMapperList(self):
        def f(mappers_factory):
            mapper_list_items = mappers_factory()
            mapper_list = ArrayMapperList(mapper_list_items)
            self.assertEqual(len(mapper_list), len(mapper_list_items))
            self.assertListEqual(list(mapper_list), mapper_list_items)
            for i in range(len(mapper_list)):
                self.assertIs(mapper_list[i], mapper_list_items[i])

            input_info = ArrayInfo(
                shape=[32, 31, 3], is_discrete=True,
                n_discrete_vals=256, min_val=0, max_val=255, dtype='uint8')
            self.assertEqual(mapper_list.fitted, False)
            output_info = mapper_list.fit(input_info)
            self.assertEqual(mapper_list.fitted, True)
            self.assertEqual(mapper_list.input_info, input_info)
            self.assertEqual(mapper_list.output_info, output_info)

            if mapper_list:
                mappers_list2 = ArrayMapperList(mapper_list)
                self.assertEqual(mappers_list2.fitted, True)
                self.assertEqual(mappers_list2.input_info, input_info)
                self.assertEqual(mappers_list2.output_info, output_info)
                self.assertEqual(list(mappers_list2), list(mapper_list))

            mappers = mappers_factory()
            output_info2 = input_info
            for m in mappers:
                output_info2 = m.fit(output_info2)
            self.assertEqual(mapper_list.output_info, output_info2)

            for batch_shape in ([], [2, 3]):
                x = np.random.randint(0, 256, size=batch_shape + input_info.shape)
                x = x.astype(np.uint8)
                y = mapper_list.transform(x)
                self.assertEqual(y.dtype, np.dtype(output_info.dtype))
                self.assertEqual(list(y.shape), batch_shape + output_info.shape)

                y0 = x
                for m in mappers:
                    y0 = m.transform(y0)
                np.testing.assert_allclose(y, y0, atol=1e-4, rtol=1e-6)
                np.testing.assert_allclose(mapper_list.inverse_transform(y), x)

        f(lambda: [])
        f(lambda: [ScaleToRange(-1., 1.)])
        f(lambda: [
            Flatten(),
            ScaleToRange(-2., 3.),
        ])

    def test_Identity(self):
        m = Identity()
        info = ArrayInfo(shape=[3, 2], min_val=1, max_val=3)
        self.assertEqual(m.fit(info), info)
        self.assertTrue(m.fitted)
        self.assertEqual(m.input_info, info)
        self.assertEqual(m.output_info, info)

        x = np.arange(24).reshape([4, 3, 2])
        np.testing.assert_equal(m.transform(x), x)
        np.testing.assert_equal(m.inverse_transform(x), x)

    def test_Reshape(self):
        with pytest.raises(ValueError,
                           match=r'At most one `-1` can be present in `shape`'):
            _ = Reshape([-1, -1])
        with pytest.raises(ValueError, match=r'Not a valid shape'):
            _ = Reshape([-2])
        with pytest.raises(ValueError, match=r'Cannot reshape array'):
            Reshape([2, 3]).fit(ArrayInfo(shape=[12]))
        with pytest.raises(ValueError, match=r'Cannot reshape array'):
            Reshape([-1, 3]).fit(ArrayInfo(shape=[7]))

        # test full shape specified
        info = ArrayInfo(shape=[3, 2], min_val=1, max_val=3)
        for x, y in [(np.arange(24).reshape([4, 3, 2]),
                      np.arange(24).reshape([4, 2, 3])),
                     (np.arange(6).reshape([3, 2]),
                      np.arange(6).reshape([2, 3]))]:
            for shape in ([2, 3], [-1, 3]):
                m = Reshape(shape)
                out_info = m.fit(info)
                self.assertTrue(m.fitted)
                self.assertEqual(m.input_info, info)
                self.assertEqual(out_info, info.copy(shape=[2, 3]))
                self.assertEqual(m.in_shape, [3, 2])
                self.assertEqual(m.out_shape, [2, 3])
                self.assertEqual(m.output_info, out_info)

                np.testing.assert_equal(m.transform(x), y)
                np.testing.assert_equal(m.inverse_transform(y), x)

    def test_Flatten(self):
        for shape in ([2, 3], [7]):
            info = ArrayInfo(shape=shape, min_val=1, max_val=3)
            x = np.random.randn(9, *shape)
            y = x.reshape([9, -1])

            m = Flatten()
            out_info = m.fit(info)
            self.assertTrue(m.fitted)
            self.assertEqual(m.input_info, info)
            self.assertEqual(out_info, info.copy(shape=[int(np.prod(shape))]))
            np.testing.assert_equal(m.transform(x), y)
            np.testing.assert_equal(m.inverse_transform(y), x)

    def test_Transpose(self):
        def f(true_shape, shape, perm):
            if shape is None:
                shape = true_shape
            info = ArrayInfo(shape=shape)
            new_shape = [shape[a] for a in perm]

            m = Transpose(perm)
            self.assertEqual(m.fit(info), info.copy(shape=new_shape))

            for batch_shape in ([], [2, 3]):
                x = np.random.randn(*(batch_shape + true_shape))
                y = m.transform(x)
                np.testing.assert_equal(
                    y,
                    np.transpose(x, list(range(len(batch_shape))) + perm)
                )
                np.testing.assert_equal(m.inverse_transform(y), x)

        f([2, 3, 4], None, [-3, -1, -2])
        f([2, 3, 4], [None, None, None], [-3, -1, -2])

    def test_Pad(self):
        def f(shape, padding, fill_value):
            info = ArrayInfo(shape=shape)
            new_shape = list(shape)
            for i, (s, (l, r)) in enumerate(
                    zip(reversed(shape), reversed(padding)), 1):
                new_shape[-i] = s + l + r

            m = Pad(padding, fill_value)
            self.assertEqual(m.fit(info), info.copy(shape=new_shape))

            for batch_shape in ([], [2, 3]):
                x = np.random.randn(*(batch_shape + shape))
                y = m.transform(x)
                ans = np.pad(
                    x,
                    [(0, 0)] * (len(batch_shape) + len(shape) - len(padding)) + padding,
                    mode='constant',
                    constant_values=fill_value
                )
                np.testing.assert_equal(y, ans)
                np.testing.assert_equal(m.inverse_transform(y), x)

        for fill_value in (0., 1., -1.):
            f([2, 3, 4], [(0, 3), (2, 0)], fill_value)

    def test_ChannelTranspose(self):
        def f(from_format, to_format, spatial_ndims, cls=None):
            make_shape = lambda s, n, fmt: (s + [n] if fmt == 'channel_last'
                                            else [n] + s)
            spatial_shape = [7, 8, 9][:spatial_ndims]
            from_shape = make_shape(spatial_shape, 5, from_format)
            to_shape = make_shape(spatial_shape, 5, to_format)

            if from_format == to_format:
                expected_m = Identity()
            elif from_format == 'channel_last':
                expected_m = Transpose([-1] + [-4, -3, -2][-spatial_ndims:])
            else:
                expected_m = Transpose(
                    [-3, -2, -1][-spatial_ndims:] +
                    [[-4, -3, -2][-spatial_ndims]]
                )

            info = ArrayInfo(shape=from_shape, min_val=0, max_val=255,
                             n_discrete_vals=256, is_discrete=True, dtype='uint8')
            if cls is None:
                m = ChannelTranspose(from_format, to_format)
            else:
                m = cls()
            out_info = m.fit(info)
            self.assertEqual(out_info.shape, to_shape)
            self.assertEqual(out_info, expected_m.fit(info))

            for batch_shape in ([], [2, 3]):
                x = np.random.randint(0, 256, size=batch_shape + from_shape,
                                      dtype=np.uint8)
                y = m.transform(x)
                np.testing.assert_equal(y, expected_m.transform(x))
                np.testing.assert_equal(m.inverse_transform(y), x)

        for from_format, to_format, spatial_ndims in itertools.product(
                    ('channel_last', 'channel_first'),
                    ('channel_last', 'channel_first'),
                    (1, 2, 3),
                ):
            f(from_format, to_format, spatial_ndims)

        for spatial_ndims in (1, 2, 3):
            f('channel_last', 'channel_first', spatial_ndims, ChannelLastToFirst)
            f('channel_first', 'channel_last', spatial_ndims, ChannelFirstToLast)

            if T.IS_CHANNEL_LAST:
                f('channel_last', 'channel_last', spatial_ndims, ChannelLastToDefault)
                f('channel_first', 'channel_last', spatial_ndims, ChannelFirstToDefault)
            else:
                f('channel_last', 'channel_first', spatial_ndims, ChannelLastToDefault)
                f('channel_first', 'channel_first', spatial_ndims, ChannelFirstToDefault)

    def test_Affine(self):
        # test with min_val and max_val
        m = Affine(scale=2.5, bias=-1.5, dtype='float64')
        info = ArrayInfo(dtype='float32', min_val=-1., max_val=2.)
        out_info = m.fit(info)
        self.assertEqual(m.input_info, info)
        self.assertEqual(out_info.dtype, 'float64')
        self.assertAlmostEqual(out_info.min_val, -1. * 2.5 - 1.5)
        self.assertAlmostEqual(out_info.max_val, 2. * 2.5 - 1.5)

        x = np.linspace(-1., 2., 1000).astype(np.float32)
        y = (x * 2.5 - 1.5).astype(np.float64)

        o = m.transform(x)
        self.assertEqual(o.dtype, np.float64)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)

        o = m.inverse_transform(y)
        self.assertEqual(o.dtype, np.float32)
        np.testing.assert_allclose(o, x, atol=1e-4, rtol=1e-6)

        # test without min_val and max_val
        m = Affine(scale=2.5, bias=-1.5)
        info = ArrayInfo()
        self.assertEqual(m.fit(info), info)
        o = m.transform(x)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)
        o = m.inverse_transform(y)
        np.testing.assert_allclose(o, x, atol=1e-4, rtol=1e-6)

        # test without max_val
        m = Affine(scale=2.5, bias=-1.5)
        info = ArrayInfo(min_val=-1.)
        self.assertEqual(m.fit(info), info.copy(min_val=-1. * 2.5 - 1.5))
        o = m.transform(x)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)
        o = m.inverse_transform(y)
        np.testing.assert_allclose(o, x, atol=1e-4, rtol=1e-6)

        # test without min_val
        m = Affine(scale=2.5, bias=-1.5)
        info = ArrayInfo(max_val=2.)
        self.assertEqual(m.fit(info), info.copy(max_val=2. * 2.5 - 1.5))
        o = m.transform(x)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)
        o = m.inverse_transform(y)
        np.testing.assert_allclose(o, x, atol=1e-4, rtol=1e-6)

        # test int -> float
        m = Affine(scale=2.5, bias=-1.5, dtype='float64')
        info = ArrayInfo(dtype='int32', min_val=-1., max_val=2)
        out_info = m.fit(info)
        self.assertEqual(out_info.dtype, 'float64')
        self.assertAlmostEqual(out_info.min_val, -1 * 2.5 - 1.5)
        self.assertAlmostEqual(out_info.max_val, 2 * 2.5 - 1.5)

        x = np.array([-1, 0, 1, 2]).astype(np.int32)
        y = (x * 2.5 - 1.5).astype(np.float64)

        o = m.transform(x)
        self.assertEqual(o.dtype, np.float64)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)

        o = m.inverse_transform(y)
        self.assertEqual(o.dtype, np.int32)
        np.testing.assert_equal(o, x)

        # test float -> int
        m = Affine(scale=2, bias=-3, dtype='int32')
        info = ArrayInfo(dtype='float32', min_val=-1, max_val=2)
        out_info = m.fit(info)
        self.assertEqual(out_info.dtype, 'int32')
        self.assertAlmostEqual(out_info.min_val, -1 * 2 - 3)
        self.assertAlmostEqual(out_info.max_val, 2 * 2 - 3)

        x = np.linspace(-1, 2, 1000).astype(np.float32)
        y = np.round(x * 2 - 3).astype(np.int32)

        o = m.transform(x)
        self.assertEqual(o.dtype, np.int32)
        np.testing.assert_equal(o, y)

        o = m.inverse_transform(y)
        self.assertEqual(o.dtype, np.float32)
        np.testing.assert_allclose(o, ((y + 3) / 2.), atol=1e-4, rtol=1e-6)

    def test_ScaleToRange(self):
        m = ScaleToRange(-0.5, 0.5, dtype='float64')
        info = ArrayInfo(dtype='float32', min_val=-1., max_val=2.)
        out_info = m.fit(info)
        self.assertEqual(out_info.dtype, 'float64')
        self.assertEqual(out_info.min_val, -0.5)
        self.assertEqual(out_info.max_val, 0.5)

        x = np.linspace(-1., 2., 1000).astype(np.float32)
        y = np.linspace(-0.5, 0.5, 1000).astype(np.float64)

        o = m.transform(x)
        self.assertEqual(o.dtype, np.float64)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)

        o = m.inverse_transform(y)
        self.assertEqual(o.dtype, np.float32)
        np.testing.assert_allclose(o, x, atol=1e-4, rtol=1e-6)

    def test_ReduceToBitDepth(self):
        # test 8bit int -> 5bit int
        m = ReduceToBitDepth(5)
        info = ArrayInfo(dtype='int32', min_val=0, max_val=255,
                         is_discrete=True, n_discrete_vals=256, bit_depth=8)
        self.assertEqual(
            m.fit(info),
            info.copy(max_val=0xf8, n_discrete_vals=32, bit_depth=5)
        )

        x = np.arange(0, 256, dtype=np.int32)
        y = (x >> 3) << 3

        o = m.transform(x)
        self.assertEqual(o.dtype, np.int32)
        np.testing.assert_equal(o, y)

        o = m.inverse_transform(y)
        self.assertEqual(o.dtype, np.int32)
        np.testing.assert_equal(o, y)

        with pytest.raises(RuntimeError,
                           match='`ReduceToBitDepth` is not strictly invertible'):
            _ = m.inverse_transform(y, strict=True)

        # test 8bit int -> 8bit int
        m = ReduceToBitDepth(8)
        info = ArrayInfo(dtype='uint8', min_val=-128, max_val=127,
                         is_discrete=True, n_discrete_vals=256, bit_depth=8)
        self.assertEqual(m.fit(info), info)

        x = np.arange(-128, 127, dtype=np.uint8)
        np.testing.assert_equal(m.transform(x), x)
        np.testing.assert_equal(m.inverse_transform(x), x)
        np.testing.assert_equal(m.inverse_transform(x, strict=True), x)

        # test 8bit float -> 5bit float
        m = ReduceToBitDepth(5)
        info = ArrayInfo(dtype='float32', min_val=-1., max_val=3.,
                         is_discrete=True, n_discrete_vals=256, bit_depth=8)
        out_info = m.fit(info)
        self.assertEqual(out_info.n_discrete_vals, 32)
        self.assertEqual(out_info.bit_depth, 5)
        self.assertAlmostEqual(out_info.min_val, -1.)
        self.assertAlmostEqual(out_info.max_val, 0xf8 * 4 / 255. - 1.)

        x = (np.arange(0, 256, dtype=np.float32) * 4 / 255. - 1.).astype(np.float32)
        y = (((np.arange(0, 256, dtype=np.int32) >> 3) << 3) * 4 / 255. - 1.).astype(np.float32)

        o = m.transform(x)
        self.assertEqual(o.dtype, np.float32)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)

        o = m.inverse_transform(y)
        self.assertEqual(o.dtype, np.float32)
        np.testing.assert_allclose(o, y, atol=1e-4, rtol=1e-6)

        # test errors
        with pytest.raises(ValueError, match=r'`info.n_discrete_vals != 2 '
                                             r'\*\* info.bit_depth`'):
            ReduceToBitDepth(5).fit(ArrayInfo(bit_depth=7, n_discrete_vals=256,
                                              is_discrete=True, min_val=0,
                                              max_val=127))
        with pytest.raises(ValueError, match=r'Cannot enlarge bit-depth'):
            ReduceToBitDepth(8).fit(ArrayInfo(bit_depth=5, n_discrete_vals=32,
                                              is_discrete=True, min_val=0,
                                              max_val=31))

    def test_Dequantize(self):
        np.random.seed(1234)

        for out_dtype in ('float32', 'float64'):
            # in_dtype == int
            m = Dequantize(out_dtype)
            info = ArrayInfo(dtype='int32', min_val=0, max_val=255,
                             is_discrete=True, n_discrete_vals=256, bit_depth=8)
            out_info = m.fit(info)
            self.assertEqual(out_info.dtype, out_dtype)
            self.assertEqual(out_info.is_discrete, False)
            self.assertAlmostEqual(out_info.min_val, -0.5)
            self.assertAlmostEqual(out_info.max_val, 255.5)

            x = np.arange(0, 256, dtype=np.int32)

            o = m.transform(x)
            self.assertEqual(o.dtype, out_dtype)
            np.testing.assert_array_less(np.abs(o - x), 0.5)

            o = m.inverse_transform(o)
            self.assertEqual(o.dtype, np.int32)
            np.testing.assert_equal(o, x)

            # in_dtype == float
            m = Dequantize(out_dtype)
            info = ArrayInfo(dtype='float32', min_val=-1.5, max_val=2.5,
                             is_discrete=True, n_discrete_vals=256, bit_depth=8)
            out_info = m.fit(info)
            self.assertEqual(out_info.dtype, out_dtype)
            self.assertEqual(out_info.is_discrete, False)
            self.assertAlmostEqual(out_info.min_val, -1.5 - 2 / 255.)
            self.assertAlmostEqual(out_info.max_val, 2.5 + 2 / 255.)

            x = np.linspace(-1.5, 2.5, 256, dtype=np.float32)

            o = m.transform(x)
            self.assertEqual(o.dtype, out_dtype)
            np.testing.assert_array_less(np.abs(o - x), 2 / 255.)

            o = m.inverse_transform(o)
            self.assertEqual(o.dtype, np.float32)
            np.testing.assert_allclose(o, x, atol=1e-4, rtol=1e-6)

    def test_BernoulliSample(self):
        np.random.seed(1234)
        for in_dtype, out_dtype in itertools.product(('float32', 'float64'),
                                                     ('int32', 'float64')):
            m = BernoulliSample(out_dtype)
            info = ArrayInfo(dtype=in_dtype, min_val=0, max_val=1)
            self.assertEqual(
                m.fit(info),
                info.copy(dtype=out_dtype, is_discrete=True, n_discrete_vals=2,
                          bit_depth=1)
            )

            x = np.random.rand(10000, 3, 4) * 0.999 + 5e-4
            o = m.transform(x)
            self.assertEqual(o.dtype, np.dtype(out_dtype))
            self.assertEqual(o.shape, x.shape)
            self.assertEqual(set(o.reshape([-1]).tolist()), {0, 1})

            std = np.exp(0.5 * (np.log(x) + np.log1p(-x)))
            ratio = np.sum(np.abs(x - o) < 3 * std) / np.prod(np.shape(x))
            self.assertGreater(ratio, 0.98)

            y = m.inverse_transform(o)
            self.assertEqual(y.dtype, np.dtype(in_dtype))
            np.testing.assert_equal(y, o)

            with pytest.raises(RuntimeError,
                               match='`BernoulliSampler` is not strictly '
                                     'invertible'):
                _ = m.inverse_transform(o, strict=True)

        with pytest.raises(ValueError,
                           match=r'The source array values are not continuous, '
                                 r'or not within the range \[0, 1\]'):
            BernoulliSample().fit(ArrayInfo(is_discrete=True, min_val=0, max_val=1))

        with pytest.raises(ValueError,
                           match=r'The source array values are not continuous, '
                                 r'or not within the range \[0, 1\]'):
            BernoulliSample().fit(ArrayInfo(is_discrete=False, min_val=-1, max_val=1))

        with pytest.raises(ValueError,
                           match=r'The source array values are not continuous, '
                                 r'or not within the range \[0, 1\]'):
            BernoulliSample().fit(ArrayInfo(is_discrete=False, min_val=0, max_val=2))

    def test_DownSample(self):
        x = np.random.randn(2, 4, 6, 8).astype(np.float32)

        # all empty
        info = ArrayInfo(shape=[], dtype='float32')
        m = DownSample(scale=[], dtype='float64')
        self.assertEqual(m.fit(info), info.copy(dtype='float64'))

        y = m.transform(x)
        self.assertEqual(y.dtype, np.float64)
        np.testing.assert_equal(y, x)
        np.testing.assert_equal(m.inverse_transform(y), x)

        # scale empty, shape not empty
        info = ArrayInfo(shape=[4, 6, 8], dtype='float32')
        m = DownSample(scale=[], dtype='float64')
        self.assertEqual(m.fit(info), info.copy(dtype='float64'))

        y = m.transform(x)
        self.assertEqual(y.dtype, np.float64)
        np.testing.assert_equal(y, x)
        np.testing.assert_equal(m.inverse_transform(y), x)

        # some axis scaled
        info = ArrayInfo(shape=[2, 4, 6, 8], dtype='float32')
        m = DownSample(scale=[1, 3, 2], dtype='float64')
        self.assertEqual(m.fit(info), info.copy(shape=[2, 4, 2, 4], dtype='float64'))

        y = m.transform(x)
        ans = (x[..., ::2] + x[..., 1::2]) / 2.
        ans = (ans[..., ::3, :] + ans[..., 1::3, :] + ans[..., 2::3, :]) / 3.
        self.assertEqual(y.dtype, np.float64)
        np.testing.assert_allclose(y, ans, rtol=1e-4, atol=1e-6)
        self.assertEqual(m.inverse_transform(y).dtype, np.float32)
        np.testing.assert_allclose(
            m.inverse_transform(y),
            np.reshape(
                np.tile(
                    np.reshape(y, [2, 4, 2, 1, 4, 1]),
                    [1, 1, 1, 3, 1, 2],
                ),
                [2, 4, 6, 8]
            ),
            rtol=1e-4, atol=1e-6
        )

        with pytest.raises(ValueError, match='`array` must be at least 3d'):
            _ = m.transform(np.random.randn(6, 8))

        with pytest.raises(ValueError, match='`array.shape` cannot be evenly '
                                             'divided by `scale`'):
            _ = m.transform(np.random.randn(2, 4, 7, 8))

        with pytest.raises(ValueError, match='`array` must be at least 3d'):
            _ = m.inverse_transform(np.random.randn(6, 8))

        # not deterministic shape
        info = ArrayInfo(shape=[None, None, 6, 8], dtype='float32')
        m = DownSample(scale=[2, 2, 1, 4], dtype='float64')
        self.assertEqual(m.fit(info),
                         info.copy(shape=[None, None, 6, 2], dtype='float64'))

        y = m.transform(x)
        ans = (x[..., ::4] + x[..., 1::4] + x[..., 2::4] + x[..., 3::4]) / 4.
        ans = (ans[..., ::2, :, :] + ans[..., 1::2, :, :]) / 2.
        ans = (ans[..., ::2, :, :, :] + ans[..., 1::2, :, :, :]) / 2.
        self.assertEqual(y.dtype, np.float64)
        np.testing.assert_allclose(y, ans, rtol=1e-4, atol=1e-6)
        self.assertEqual(m.inverse_transform(y).dtype, np.float32)
        np.testing.assert_allclose(
            m.inverse_transform(y),
            np.reshape(
                np.tile(
                    np.reshape(y, [1, 1, 2, 1, 6, 1, 2, 1]),
                    [1, 2, 1, 2, 1, 1, 1, 4],
                ),
                [2, 4, 6, 8]
            ),
            rtol=1e-4, atol=1e-6
        )

        # test errors
        with pytest.raises(ValueError, match='`info.shape` must be at least 2d'):
            DownSample([2, 3]).fit(ArrayInfo(shape=[1]))

        with pytest.raises(ValueError, match='`info.shape` cannot be evenly '
                                             'divided by `scale`'):
            DownSample([2, 3]).fit(ArrayInfo(shape=[1, 3, 7]))

        with pytest.raises(RuntimeError, match='`DownSample` is not strictly '
                                               'invertible'):
            m = DownSample([2, 3])
            m.fit(ArrayInfo(shape=[2, 4, 6]))
            m.inverse_transform(np.random.randn(1, 1, 1), strict=True)

    def test_UpSample(self):
        x = np.random.randn(1, 2, 3, 4).astype(np.float32)

        # empty
        info = ArrayInfo(shape=[4, 6, 8], dtype='float32')
        m = UpSample(scale=[], dtype='float64')
        self.assertEqual(m.fit(info), info.copy(dtype='float64'))

        y = m.transform(x)
        self.assertEqual(y.dtype, np.float64)
        np.testing.assert_equal(y, x)
        np.testing.assert_equal(m.inverse_transform(y), x)

        # scaled
        info = ArrayInfo(shape=[1, None, 3, 4], dtype='float32')
        m = UpSample(scale=[3, 1, 2], dtype='float64')
        self.assertEqual(m.fit(info), info.copy(shape=[1, None, 3, 8], dtype='float64'))

        y = m.transform(x)
        ans = np.reshape(
            np.tile(
                np.reshape(x, [1, 2, 1, 3, 1, 4, 1]),
                [1, 1, 3, 1, 1, 1, 2],
            ),
            [1, 6, 3, 8],
        )
        self.assertEqual(y.dtype, np.float64)
        np.testing.assert_allclose(y, ans, rtol=1e-4, atol=1e-6)
        self.assertEqual(m.inverse_transform(y).dtype, np.float32)
        np.testing.assert_allclose(m.inverse_transform(y), x)

        # errors
        with pytest.raises(ValueError, match='`info.shape` must be at least 2d'):
            UpSample([2, 3]).fit(ArrayInfo(shape=[7]))

    def test_GrayscaleToRGB(self):
        def cls_factory(channel_last):
            return GrayscaleToRGB(channel_last=channel_last)

        def fn(x, n_channels, max_val, dtype):
            return np.concatenate([x, x, x], axis=-1)

        def check_out_info(out_info, channel_last, the_channel_last,
                           n_channels, max_val, dtype):
            expected_shape = [31, 32]
            if n_channels is not None:
                if the_channel_last:
                    expected_shape = expected_shape + [n_channels * 3]
                else:
                    expected_shape = [n_channels * 3] + expected_shape
            self.assertEqual(out_info.shape, expected_shape)

        def is_supported(channel_last, n_channels, max_val, dtype):
            return n_channels == 1

        standard_image_mapper_check(
            self, [31, 32], lambda channel_last: cls_factory(channel_last),
            fn, check_out_info=check_out_info, is_supported=is_supported,
            is_invertible=True
        )
        standard_image_mapper_check(
            self, [31, 32], lambda channel_last: cls_factory(channel_last),
            fn, check_out_info=check_out_info, is_supported=is_supported,
            is_invertible=True
        )

        with pytest.raises(ValueError, match='Invalid shape'):
            GrayscaleToRGB(channel_last=True).fit(ArrayInfo(shape=[31, 32, 3]))

        with pytest.raises(ValueError, match='Invalid shape'):
            GrayscaleToRGB(channel_last=False).fit(ArrayInfo(shape=[3, 31, 32]))

    def test_CropImage(self):
        bbox = [3, 29, 5, 27]  # (top, bottom, left, right)

        def cls_factory(channel_last, use_bbox):
            if use_bbox:
                return CropImage(bbox=bbox, channel_last=channel_last)
            else:
                return CropImage(
                    pos=(bbox[0], bbox[2]),
                    size=[bbox[1] - bbox[0], bbox[3] - bbox[2]],
                    channel_last=channel_last
                )

        def fn(x, n_channels, max_val, dtype):
            return x[..., bbox[0]: bbox[1], bbox[2]: bbox[3], :]

        def check_out_info(out_info, channel_last, the_channel_last,
                           n_channels, max_val, dtype):
            expected_shape = [26, 22]
            if n_channels is not None:
                if the_channel_last:
                    expected_shape = expected_shape + [n_channels]
                else:
                    expected_shape = [n_channels] + expected_shape
            self.assertEqual(out_info.shape, expected_shape)

        standard_image_mapper_check(
            self, [31, 32], lambda channel_last: cls_factory(channel_last, True),
            fn, check_out_info=check_out_info)
        standard_image_mapper_check(
            self, [31, 32], lambda channel_last: cls_factory(channel_last, False),
            fn, check_out_info=check_out_info)

        # test errors
        with pytest.raises(ValueError,
                           match='Either `bbox`, or a pair of `pos` and `size` '
                                 'should be specified, but not both.'):
            _ = CropImage()

        with pytest.raises(ValueError,
                           match='Either `bbox`, or a pair of `pos` and `size` '
                                 'should be specified, but not both.'):
            _ = CropImage(size=[31, 32])

        with pytest.raises(ValueError,
                           match='Either `bbox`, or a pair of `pos` and `size` '
                                 'should be specified, but not both.'):
            _ = CropImage(pos=[31, 32])

        with pytest.raises(ValueError,
                           match='Either `bbox`, or a pair of `pos` and `size` '
                                 'should be specified, but not both.'):
            _ = CropImage(bbox=[1, 2, 3, 4], size=[31, 32], pos=[1, 2])

        with pytest.raises(ValueError,
                           match='`bbox` must be a sequence of 4 integers'):
            _ = CropImage(bbox=[10, 11, 12])

        with pytest.raises(ValueError,
                           match='`pos` and `size` must be sequences of 2 integers'):
            _ = CropImage(pos=[10, 11, 12], size=[31, 32])

        with pytest.raises(ValueError,
                           match='`pos` and `size` must be sequences of 2 integers'):
            _ = CropImage(pos=[10, 11], size=[31, 32, 33])

        with pytest.raises(ValueError,
                           match=r'Spatial shape `\[10, 12\]` cannot be cropped'):
            CropImage(bbox=bbox).fit(ArrayInfo(shape=[10, 12, 1]))

        with pytest.raises(RuntimeError, match='`CropImage` is not invertible'):
            m = CropImage(bbox=bbox)
            m.fit(ArrayInfo(shape=[31, 32, 1]))
            m.inverse_transform(np.random.randint(0, 2, size=[26, 22, 1]))

    @slow_test
    def test_ScaleImage(self):
        for size, mode in itertools.product(
                ([4, 4], [3, 3]),
                (ScaleImageMode.SCIPY_NO_AA,
                 ScaleImageMode.SCIPY_GAUSSIAN_AA)):
            def cls_factory(channel_last):
                return ScaleImage(size, mode, channel_last)

            def kernel(x, max_val, dtype):
                if mode == ScaleImageMode.SCIPY_NO_AA:
                    return transform.resize(x, size, anti_aliasing=False)
                elif mode == ScaleImageMode.SCIPY_GAUSSIAN_AA:
                    return transform.resize(x, size, anti_aliasing=True)

                raise NotImplementedError()

            def fn(x, n_channels, max_val, dtype):
                front_shape, back_shape = x.shape[:-3], x.shape[-3:]
                x = x.reshape([-1] + list(back_shape))
                if max_val == 255:
                    x = (x / 255.)
                x = np.stack([kernel(im, max_val, dtype) for im in x], axis=0)
                if max_val == 255:
                    x = x * 255
                x = np.reshape(x, front_shape + x.shape[-3:])
                x = x.astype(dtype)
                return x

            def check_out_info(out_info, channel_last, the_channel_last,
                               n_channels, max_val, dtype):
                expected_shape = size
                if n_channels is not None:
                    if the_channel_last:
                        expected_shape = expected_shape + [n_channels]
                    else:
                        expected_shape = [n_channels] + expected_shape
                self.assertEqual(out_info.shape, expected_shape)

            def comparer(x, y, max_val, dtype):
                return np.sum(np.abs(x - y)) / np.prod(x.shape) < (2. * max_val / 255)

            standard_image_mapper_check(self, [8, 8], cls_factory, fn,
                                        check_out_info=check_out_info,
                                        comparer=comparer)

        with pytest.raises(RuntimeError, match='`ScaleImage` is not invertible'):
            m = ScaleImage([4, 4])
            m.fit(ArrayInfo(shape=[8, 8, 1], max_val=255, min_val=0))
            m.inverse_transform(np.random.randint(0, 256, size=[8, 8, 1]))
