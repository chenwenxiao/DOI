'''
load lsun dataset as numpy array

usage:

    import lsun

    (test_x, test_y) = load_lsun_test()

'''
from PIL import Image
from scipy.ndimage import filters
import os
import tensorflow as tf
import numpy as np

TEST_X_PATH = '/home/cwx17/data/isun'
TEST_X_ARR_PATH = '/home/cwx17/data/isun/test.npy'


def _fetch_array_x(path):
    file_names = os.listdir(path)
    file_names.sort()
    imgs = []
    scale = 148 / float(64)
    sigma = np.sqrt(scale) / 2.0
    for name in file_names:
        im = Image.open(os.path.join(path, name))
        wd = im.size[0]
        he = im.size[1]
        side = min(wd, he)
        dhe = he - side
        dwd = wd - side

        im = im.crop((dwd / 2, dhe / 2, wd - dwd / 2, he - dhe / 2))
        img = np.asarray(im)
        # img.setflags(write=True)
        # for dim in range(img.shape[2]):
        # img[...,dim] = filters.gaussian_filter(img[...,dim], sigma=(sigma,sigma))
        if len(img.shape) > 2:
            imgs.append(img)

    return np.array(imgs)


def _fetch_array_y(path):
    evalue = []
    with open(path, 'rb') as f:
        for line in f.readlines():
            q = line.decode('utf-8')
            q = q.strip()
            q = int(q.split(' ')[1])
            evalue.append(q)
    return np.array(evalue)


def load_isun_test(x_shape=(32, 32, 3), x_dtype=np.float32, y_dtype=np.int32,
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
    if (not os.path.exists(TEST_X_PATH)):
        print("test dir not found")
        return

    test_x = np.load(TEST_X_ARR_PATH)

    test_y = None

    return (test_x, test_y)


# def prepare_arr():
#     if (not os.path.exists(TRAIN_X_ARR_PATH)):
#         print('train arr not found')
#         train_x = _fetch_array_x(TRAIN_X_PATH)
#         train_x /= np.asarray(255., dtype=np.int32)
#         np.save(train_x,TRAIN_X_ARR_PATH)
#     elif (not os.path.exists(TEST_X_ARR_PATH)):
#         print("test arr not found")
#         test_x = _fetch_array_x(TEST_X_PATH)
#         test_x /= np.asarray(255., dtype=np.int32)
#         np.save(test_x,TEST_X_ARR_PATH)
#     else:
#         return




if __name__ == '__main__':
    (x_test, y_test) = load_isun_test()
    print(x_test.shape)
    np.save(TEST_X_PATH, x_test)


    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
