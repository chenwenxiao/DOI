import gzip
import hashlib

import numpy as np
import idx2numpy

from tfsnippet.utils import CacheDir

TRAIN_X_PATH = '/home/cwx17/data/not_mnist/train-images-idx3-ubyte.gz'
TRAIN_Y_PATH = '/home/cwx17/data/not_mnist/train-labels-idx1-ubyte.gz'
TEST_X_PATH = '/home/cwx17/data/not_mnist/t10k-images-idx3-ubyte.gz'
TEST_Y_PATH = '/home/cwx17/data/not_mnist/t10k-labels-idx1-ubyte.gz'


def _fetch_array(path):
    """Fetch an MNIST array from the `uri` with cache."""
    with gzip.open(path, 'rb') as f:
        return idx2numpy.convert_from_file(f)


def load_not_mnist(x_shape=(28, 28, 1), x_dtype=np.uint8, y_dtype=np.int32,
                   normalize_x=False):
    """
    Load the MNIST dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.  Default ``(28, 28, 1)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
    """
    # check arguments
    # x_shape = _validate_x_shape(x_shape)

    # load data
    train_x = _fetch_array(TRAIN_X_PATH).astype(x_dtype)
    train_y = _fetch_array(TRAIN_Y_PATH).astype(y_dtype)
    test_x = _fetch_array(TEST_X_PATH).astype(x_dtype)
    test_y = _fetch_array(TEST_Y_PATH).astype(y_dtype)

    # assert (len(train_x) == len(train_y) == 60000)
    # assert (len(test_x) == len(test_y) == 10000)

    # change shape
    train_x = train_x.reshape([len(train_x)] + list(x_shape))
    test_x = test_x.reshape([len(test_x)] + list(x_shape))

    # normalize x
    if normalize_x:
        train_x /= np.asarray(255., dtype=train_x.dtype)
        test_x /= np.asarray(255., dtype=test_x.dtype)

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_not_mnist()
    print(x_train.shape)
    print(x_test.shape)

    im = np.array(x_train[19])
    im = im.reshape(28, 28)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.savefig('test.png')
