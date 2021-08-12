import hashlib

import numpy as np
import tfsnippet as spt
from scipy.io import loadmat

from .base import StandardImageDataSet, registry

__all__ = ['load_omniglot', 'OmniglotDataSet']

DATASET_URI = 'https://raw.githubusercontent.com/yburda/iwae/master/datasets/' \
              'OMNIGLOT/chardata.mat'
DATASET_MD5 = '21ecaf34373511aa7691cf2e0802aa3d'


def _fetch_arrays():
    path = spt.utils.CacheDir('omniglot').download(
        DATASET_URI, hasher=hashlib.md5(), expected_hash=DATASET_MD5)
    m = loadmat(path)
    return m


def _validate_x_shape(x_shape):
    x_shape = tuple([int(v) for v in x_shape])
    if np.prod(x_shape) != 784:
        raise ValueError('`x_shape` does not product to 784: {!r}'.
                         format(x_shape))
    return x_shape


def load_omniglot(x_shape=(28, 28, 1), x_dtype=np.uint8, y_dtype=np.int32,
                  normalize_x=False):
    """
    Load the OMNIGLOT dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        y_dtype: Cast each label into this data type.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (background_x, background_y), (test_x, test_y)
    """
    # check arguments
    x_shape = _validate_x_shape(x_shape)

    # load data
    data = _fetch_arrays()
    train_x = np.transpose(np.asarray(data['data']), [1, 0])
    train_y = np.asarray(data['targetchar'][0], dtype=y_dtype)
    test_x = np.transpose(np.asarray(data['testdata']), [1, 0])
    test_y = np.asarray(data['testtargetchar'][0], dtype=y_dtype)

    assert(len(train_x) == len(train_y) == 24345)
    assert(len(test_x) == len(test_y) == 8070)

    # change shape
    train_x = train_x.reshape([len(train_x)] + list(x_shape))
    test_x = test_x.reshape([len(test_x)] + list(x_shape))

    # normalize x
    if not normalize_x:
        train_x *= 255
        test_x *= 255

    train_x = train_x.astype(x_dtype)
    test_x = test_x.astype(x_dtype)

    return (train_x, train_y), (test_x, test_y)


class OmniglotDataSet(StandardImageDataSet):
    """Omniglot image dataset."""

    def __init__(self, use_validation=False, random_state=None):
        """
        Construct a new :class:`OmniglotDataSet`.

        Args:
            use_validation (bool): Whether or not to use validation data?
            random_state (np.RandomState): Optional random state.
        """
        super(OmniglotDataSet, self).__init__(
            loader_fn=load_omniglot,
            value_shape=(28, 28, 1),
            color_depth=256,
            has_y=True,
            random_state=random_state,
            valid_size=4000 if use_validation else 0
        )

        if use_validation:
            assert(self.train_data_count == 20345)
            assert(self.valid_data_count == 4000)
        else:
            assert(self.train_data_count == 24345)
            assert(self.valid_data_count == 0)
        assert(self.test_data_count == 8070)


registry.register('Omniglot', OmniglotDataSet)
