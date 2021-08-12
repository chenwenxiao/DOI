import hashlib

import numpy as np
import tfsnippet as spt

from .base import StandardImageDataSet, registry


__all__ = ['load_static_mnist', 'StaticMNISTDataSet']

TRAIN_URI = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
            'binarized_mnist/binarized_mnist_train.amat'
TRAIN_MD5 = 'db28b1a6ae0fe70cbd3da91acf46e477'
VALID_URI = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
            'binarized_mnist/binarized_mnist_valid.amat'
VALID_MD5 = 'bf37822a04e94123630de6a7dfd9c9ef'
TEST_URI = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
           'binarized_mnist/binarized_mnist_test.amat'
TEST_MD5 = '14e41ae11040adba6ab8cb291eda0dd8'


def _fetch_array(uri, md5sum):
    path = spt.utils.CacheDir('static_mnist').download(
        uri, hasher=hashlib.md5(), expected_hash=md5sum)
    return np.loadtxt(path)


def _validate_x_shape(x_shape):
    x_shape = tuple([int(v) for v in x_shape])
    if np.prod(x_shape) != 784:
        raise ValueError('`x_shape` does not product to 784: {!r}'.
                         format(x_shape))
    return x_shape


def load_static_mnist(x_shape=(28, 28, 1), x_dtype=np.float32,
                      normalize_x=False):
    """
    Load the StaticMNIST dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): The `(train_x, valid_x, test_x)`.
    """
    # check arguments
    x_shape = _validate_x_shape(x_shape)

    # load data
    train_x = _fetch_array(TRAIN_URI, TRAIN_MD5)
    valid_x = _fetch_array(VALID_URI, VALID_MD5)
    test_x = _fetch_array(TEST_URI, TEST_MD5)

    assert(len(train_x) == 50000)
    assert(len(valid_x) == 10000)
    assert(len(test_x) == 10000)

    # change shape
    train_x = train_x.reshape([len(train_x)] + list(x_shape))
    valid_x = valid_x.reshape([len(valid_x)] + list(x_shape))
    test_x = test_x.reshape([len(test_x)] + list(x_shape))

    # normalize x
    if not normalize_x:
        train_x *= 255
        valid_x *= 255
        test_x *= 255

    train_x = train_x.astype(x_dtype)
    valid_x = valid_x.astype(x_dtype)
    test_x = test_x.astype(x_dtype)

    return train_x, valid_x, test_x


class StaticMNISTDataSet(StandardImageDataSet):
    """StaticMNIST image dataset."""

    def __init__(self, use_validation=False, random_state=None):
        """
        Construct a new :class:`StaticMNISTDataSet`.

        Args:
            use_validation (bool): Whether or not to use validation data?
            random_state (np.RandomState): Optional random state.
        """
        super(StaticMNISTDataSet, self).__init__(
            loader_fn=load_static_mnist,
            value_shape=(28, 28, 1),
            color_depth=2,
            has_y=False,
            random_state=random_state,
            valid_size=10000 if use_validation else 0
        )

        if use_validation:
            assert(self.train_data_count == 50000)
            assert(self.valid_data_count == 10000)
        else:
            assert(self.train_data_count == 60000)
            assert(self.valid_data_count == 0)
        assert(self.test_data_count == 10000)


registry.register('StaticMNIST', StaticMNISTDataSet)
