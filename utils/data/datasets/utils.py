import mltk
import numpy as np

__all__ = ['arg_sample']


def arg_sample(data_count: int, n: int, with_replacement: bool) -> mltk.Array:
    if with_replacement:
        indexes = np.random.randint(0, data_count, size=[n], dtype=np.int32)
    else:
        indexes = np.arange(0, data_count, dtype=np.int32)
        for i in range(n):
            j = np.random.randint(i, data_count, dtype=np.int32)
            indexes[i], indexes[j] = indexes[j], indexes[i]
        indexes = indexes[:n]
    return indexes
