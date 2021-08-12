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
import lmdb
import io

TEST_X_PATH = '/home/cwx17/data/lsun'
TRAIN_X_ARR_PATH = '/home/cwx17/data/lsun/train.npy'
TEST_X_ARR_PATH = '/home/cwx17/data/lsun/test.npy'


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


def load_lsun_test(x_shape=(32, 32, 3), x_dtype=np.float32, y_dtype=np.int32,
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

    # train_x = np.load(TRAIN_X_ARR_PATH)
    test_x = np.load(TEST_X_ARR_PATH)
    # train_y = range(0, len(train_x))
    test_y = None

    return (test_x, test_y)


def export_images(db_path, limit=-1):
    print('Exporting', db_path, 'to', db_path)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    imgs = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            im = Image.open(io.BytesIO(val))
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
            if img.shape == (32, 32, 3):
                imgs.append(img)
            else:
                print(img.shape)

            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')
    imgs = np.asarray(imgs)
    np.save(db_path, imgs)


def prepare_numpy(path):
    imgs = {}
    scale = 148 / float(64)
    sigma = np.sqrt(scale) / 2.0
    for root, dirs, files in os.walk(path):
        imgs[root] = []
        for name in files:
            try:
                im = Image.open(os.path.join(root, name))
                print(len(imgs), os.path.join(root, name))
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
                if img_.shape == (32, 32, 3):
                    imgs[root].append(img_)
                else:
                    print(img.shape)
            except Exception as e:
                print(e)

    np_arr = []
    label_arr = []
    class_num = 0
    for key, list in imgs.items():
        if len(list) > 0:
            np_arr.append(list)
            label_arr.append(np.ones(len(list)) * class_num)
            class_num += 1
            print(class_num, len(list), key)
    return np.concatenate(np_arr), np.concatenate(label_arr)


if __name__ == '__main__':
    load_lsun_test()
    # arr, label = prepare_numpy('/home/cwx17/data/tinyimagenet/tiny-imagenet-200/train/')
    # np.save('/home/cwx17/data/tinyimagenet/train', arr, allow_pickle=False)
    # np.save('/home/cwx17/data/tinyimagenet/train_label', label, allow_pickle=False)
    # print(arr.shape)
    # print(label.shape)

    # (x_test, y_test) = load_lsun_test()
    # print(x_test.shape)
    # np.save(TEST_X_PATH, x_test)
    # export_images('/home/cwx17/data/lsungit/bedroom_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/classroom_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/kitchen_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/bridge_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/conference_room_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/living_room_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/tower_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/church_outdoor_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/dining_room_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/restaurant_train_lmdb')
