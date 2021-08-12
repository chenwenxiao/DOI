import numpy as np
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

TRAIN_PATH = '/home/cwx17/data/svhn/train_32x32.mat'
TEST_PATH = '/home/cwx17/data/svhn/test_32x32.mat'


def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


def load_svhn(x_shape=(32, 32, 3), x_dtype=np.uint8, y_dtype=np.int32,
              normalize_x=False):
    """
    Load the SVHN dataset as NumPy arrays.

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
    train_x, train_y = load_data(TRAIN_PATH)
    test_x, test_y = load_data(TEST_PATH)

    train_x, train_y = train_x.transpose((3, 0, 1, 2)), train_y[:, 0]
    test_x, test_y = test_x.transpose((3, 0, 1, 2)), test_y[:, 0]

    train_x, test_x = np.reshape(train_x, (-1,) + x_shape), np.reshape(test_x, (-1,) + x_shape)
    train_x, train_y = train_x.astype(x_dtype), train_y.astype(y_dtype)
    test_x, test_y = test_x.astype(x_dtype), test_y.astype(y_dtype)
    # normalize x
    if normalize_x:
        train_x /= np.asarray(255., dtype=train_x.dtype)
        test_x /= np.asarray(255., dtype=test_x.dtype)

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_svhn()
    print(x_train.shape)
    print(x_test.shape)

    im = np.array(x_train[19])
    im = im.reshape(32, 32)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.savefig('test.png')
