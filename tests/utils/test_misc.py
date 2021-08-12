from unittest import TestCase

from utils import *


class MiscTestCase(TestCase):

    def test_get_bit_depth(self):
        for n in range(1, 256):
            bit_depth = get_bit_depth(n)
            self.assertLess(2 ** (bit_depth - 1), n)
            self.assertLessEqual(n, 2 ** bit_depth)
