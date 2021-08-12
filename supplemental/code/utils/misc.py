import math

__all__ = ['get_bit_depth']


def get_bit_depth(n: int) -> int:
    """
    Compute the bit-depth of `n` categories.

    `2 ** (bit_depth - 1) < n <= 2 ** bit_depth`

    Args:
        n: The number of categories.

    Returns:
        The bit-depth.
    """
    return int(math.ceil(math.log2(n)))
