'''
load lsun dataset as numpy array

usage:

    import lsun

    (test_x, test_y) = load_lsun_test()

'''
import tarfile

from PIL import Image
from scipy.ndimage import filters
import os
import tensorflow as tf
import numpy as np
import io

TRAIN_X_ARR_PATH = '/home/cwx17/new_data/noise/train.npy'
TEST_X_ARR_PATH = '/home/cwx17/new_data/noise/test.npy'

TRAIN_X_28_ARR_PATH = '/home/cwx17/new_data/noise28/train.npy'
TEST_X_28_ARR_PATH = '/home/cwx17/new_data/noise28/test.npy'


def load_noise(x_shape=(32, 32, 3), x_dtype=np.float32, y_dtype=np.int32,
               normalize_x=False):
    """
    Load the lsun dataset as NumPy arrays.
    samilar to load_not_mnist

    Args:
        Unimplemented!(haven't found a good way to resize) x_shape: Reshape each digit into this shape.  Default ``(218, 178)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
            
    """

    train_x = np.load(TRAIN_X_ARR_PATH)
    test_x = np.load(TEST_X_ARR_PATH)
    train_y = None
    test_y = None

    return (train_x, train_y), (test_x, test_y)


def load_noise28(x_shape=(28, 28, 1), x_dtype=np.float32, y_dtype=np.int32,
                 normalize_x=False):
    """
    Load the lsun dataset as NumPy arrays.
    samilar to load_not_mnist

    Args:
        Unimplemented!(haven't found a good way to resize) x_shape: Reshape each digit into this shape.  Default ``(218, 178)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)

    """

    train_x = np.load(TRAIN_X_28_ARR_PATH)
    test_x = np.load(TEST_X_28_ARR_PATH)
    train_y = None
    test_y = None

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    load_noise()
