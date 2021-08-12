import itertools
from unittest import TestCase

import numpy as np

from utils.data import ArrayInfo, image_array_to_rgb
from utils.data.mappers import *


class ImageUtilsTestCase(TestCase):

    def test_image_array_to_rgb(self):
        np.random.seed(1234)

        def f(batch_size, n_channels, channel_last, the_channel_last,
              use_info, bit_depth, dequantize, scale_to):
            shape = [31, 32]
            if n_channels is not None:
                if the_channel_last:
                    shape = shape + [n_channels]
                else:
                    shape = [n_channels] + shape

            the_info = ArrayInfo(
                shape=shape, min_val=0, max_val=255, is_discrete=True,
                n_discrete_vals=256, bit_depth=8)
            x = np.random.randint(0, 256, size=batch_size + shape)

            mappers = []
            ans_mappers = None
            if bit_depth not in (None, 8):
                mappers.append(ReduceToBitDepth(bit_depth))
                ans_mappers = ReduceToBitDepth(bit_depth)
            if dequantize:
                mappers.append(Dequantize(epsilon=1e-5))
            if scale_to:
                mappers.append(ScaleToRange(*scale_to))

            if mappers:
                m = ArrayMapperList(mappers)
                y_the_info = m.fit(the_info)
                y = m.transform(x)
            else:
                y_the_info = the_info
                y = x

            ans = x
            if ans_mappers is not None:
                ans_mappers.fit(the_info)
                ans = ans_mappers.transform(ans)

            if n_channels is None:
                ans = np.reshape(ans, ans.shape + (1,))
            elif not the_channel_last:
                ans = np.transpose(
                    ans,
                    list(range(len(ans.shape) - 3)) + [-2, -1, -3]
                )

            info = y_the_info if use_info else None
            out = image_array_to_rgb(y, info, channel_last)
            self.assertEqual(out.dtype, np.uint8)
            np.testing.assert_equal(out, ans)

        for (batch_size, n_channels, channel_last, the_channel_last,
             use_info, bit_depth, dequantize, scale_to) in itertools.product(
                    ([], [7], [3, 4]),
                    (None, 1, 3),
                    (None, True, False),
                    (True, False),
                    (True, False),
                    (8, 5),
                    (True, False),
                    (None, (0, 1), (-1, 1),),
                ):
            # skip inconsistent parameter combination
            if (n_channels is None and channel_last is not None) or \
                    (channel_last is not None and channel_last != the_channel_last):
                continue
            if n_channels is None and batch_size:
                continue

            # use_info = False is not supported along with dequantize or bit-depth
            if not use_info and (dequantize or bit_depth != 8):
                continue

            f(batch_size, n_channels, channel_last, the_channel_last,
              use_info, bit_depth, dequantize, scale_to)
