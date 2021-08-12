'''
load CelebA dataset(cropped 64x64) as numpy array
len(train_x)==162770
len(validate_x)==19867
len(test_x)==19962

usage:

    import celeba

    train_x, validate_x, test_x = celea.load_celeba()



'''
import requests
import zipfile, os

import numpy as np
import matplotlib.image as mpimg
import random

from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from skimage import transform, filters
from tqdm import tqdm

DEBUG_IMG = '/Users/lwd/Downloads/img_align_celeba'
DEBUG_EVAL = '/Users/lwd/Downloads/list_eval_partition.txt'

PRE_DIR_PATH = '/home/cwx17/data/celeba'
IMG_ZIP_PATH = '/home/cwx17/data/celeba/img_align_celeba.zip'
IMG_PATH = '/home/cwx17/data/celeba/img_align_celeba'

EVAL_PATH = '/home/cwx17/data/celeba/list_eval_partition.txt'
MAP_DIR_PATH = '/home/cwx17/data/CelebA_mmp'
MAP_PATH = '/home/cwx17/data/CelebA_mmp'

debug = False

__all__ = ['CelebADataSet']


def load_celeba(mmap_base_dir=MAP_PATH, img_size=64):
    if mmap_base_dir is None:
        raise ValueError('`mmap_base_dir` is required for CelebA.')
    if img_size not in (32, 64):
        raise ValueError(f'`img_size` must be either 32 or 64: got {img_size}.')
    data_dir = os.path.join(mmap_base_dir, 'CelebA')
    pfx = f'{img_size}x{img_size}'
    train_x = np.memmap(
        os.path.join(data_dir, f'{pfx}/train.dat'), dtype=np.uint8,
        mode='r', shape=(162770, img_size, img_size, 3))
    valid_x = np.memmap(
        os.path.join(data_dir, f'{pfx}/val.dat'), dtype=np.uint8,
        mode='r', shape=(19867, img_size, img_size, 3))
    test_x = np.memmap(
        os.path.join(data_dir, f'{pfx}/test.dat'), dtype=np.uint8,
        mode='r', shape=(19962, img_size, img_size, 3))
    return train_x, valid_x, test_x


def _resize(img, img_size=64, bbox=(40, 218 - 30, 15, 178 - 15)):
    # this function is copied from:
    # https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py

    img = img[bbox[0]: bbox[1], bbox[2]: bbox[3]]

    # Smooth image before resize to avoid moire patterns
    scale = img.shape[0] / float(img_size)
    sigma = np.sqrt(scale) / 2.0
    img = img.astype(np.float32) / 255.
    img = filters.gaussian(img, sigma=sigma, multichannel=True)
    img = transform.resize(
        img, (img_size, img_size, 3), order=3,
        # Turn off anti-aliasing, since we have done gaussian filtering.
        # Note `anti_aliasing` defaults to `True` until skimage >= 0.15,
        # which version is released in 2019/04, while the repo
        # `andersbll/autoencoding_beyond_pixels` was released in 2015.
        anti_aliasing=False,
        # same reason as above
        mode='constant',
    )
    img = (img * 255).astype(np.uint8)
    return img


class CelebADataSet():
    @staticmethod
    def make_mmap(source_dir: str,
                  mmap_base_dir: str,
                  force: bool = False,
                  img_size: int = 64):
        """
        Generate the mmap files.

        The image pre-processing method is the same as
        `https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py`.

        Args:
            source_dir: The root directory of the original CelebA dataset.
                The following directory and file are expected to exist:
                * aligned images: `source_dir + "/img_align_celeba/img_align_celeba"`
                * partition file: `source_dir + "/list_eval_partition.txt"`
            mmap_base_dir: The mmap base directory.
            force: Whether or not to force generate the files even if they
                have been already generated?
        """
        # check file system paths
        image_dir = os.path.join(source_dir, 'img_align_celeba/img_align_celeba')
        partition_file = os.path.join(
            source_dir, 'list_eval_partition.txt')

        target_dir = os.path.join(mmap_base_dir, 'CelebA')

        # read the partition file
        df = pd.read_csv(partition_file,
                         sep=' ', header=None,
                         names=['file_name', 'set_id'],
                         dtype={'file_name': str, 'set_id': int},
                         engine='c')
        assert (len(df[df['set_id'] == 0]) == 162770)
        assert (len(df[df['set_id'] == 1]) == 19867)
        assert (len(df[df['set_id'] == 2]) == 19962)

        # process the images
        def process_set(set_id, target_file, img_size):
            df_set = df[df['set_id'] == set_id]
            set_length = len(df_set)
            image_shape = (img_size, img_size, 3)
            parent_dir = os.path.split(os.path.join(target_dir, target_file))[0]
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            processed_file = os.path.join(
                target_dir, target_file + '.processed')
            if not force and os.path.isfile(processed_file):
                return
            mmap_arr = np.memmap(
                os.path.join(target_dir, target_file),
                dtype=np.uint8,
                mode='w+',
                shape=(set_length,) + image_shape,
            )

            for i, (_, row) in enumerate(
                    tqdm(df_set.iterrows(), total=set_length,
                         ascii=True, desc=target_file, unit='image')):
                file_path = os.path.join(image_dir, row['file_name'])

                # read image into array, according to the method of:
                # https://github.com/andersbll/deeppy/blob/master/deeppy/dataset/celeba.py
                im = PILImage.open(file_path)
                im_arr = im_bytes = None
                try:
                    width, height = im.size
                    im_bytes = im.tobytes()
                    im_arr = np.frombuffer(im_bytes, dtype=np.uint8). \
                        reshape((height, width, 3))
                    mmap_arr[i, ...] = _resize(im_arr, img_size=img_size)
                finally:
                    im.close()
                    del im_arr
                    del im_bytes
                    del im

            # if all is okay, generate the processed file
            with open(processed_file, 'wb') as f:
                f.write(b'\n')

        pfx = f'{img_size}x{img_size}'
        process_set(2, f'{pfx}/test.dat', img_size)
        process_set(1, f'{pfx}/val.dat', img_size)
        process_set(0, f'{pfx}/train.dat', img_size)


def download_file_from_google_drive(id, destination):
    # usage : download_file_from_google_drive(file_id_on_google_drive, path)
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


class misc():
    @staticmethod
    def download_celeba_img(path):
        'url of aligned & cropped celeba https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
        ' size: 218*178'
        ' format: jpg'
        download_file_from_google_drive('0B7EVK8r0v71pZjFTYXZWM3FlRnM', path)

    @staticmethod
    def download_celeba_eval(path):
        'url of celeba eval https://drive.google.com/open?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk'
        download_file_from_google_drive('0B7EVK8r0v71pY0NSMzRuSXJEVkk', path)

    @staticmethod
    def unzip(src, dest):
        '''
        src: address of the zip
        dest: a directory to store the file
        '''
        f = zipfile.ZipFile(src)
        if not os.path.exists(dest):
            os.makedirs(dest)
        f.extractall(dest)

    celba_size = 202598


def prepare_celeba():
    """
    Load the CelebA dataset as NumPy arrays.
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
    if (not os.path.exists(IMG_PATH)):
        print('img file not exist')
        if os.path.exists(IMG_ZIP_PATH):
            print(f'zipped file exists\n unzipping\ndst: {IMG_PATH}')
            misc.unzip(IMG_ZIP_PATH, IMG_PATH)
            print('unzipped')
        else:
            print(f'zipped file dosen\'t exist\ndownloading img \ndst: {IMG_ZIP_PATH}')
            misc.download_celeba_img(IMG_ZIP_PATH)
            print(f'downloaded\nstart unzip\ndst: {IMG_PATH}')
            misc.unzip(IMG_ZIP_PATH, IMG_PATH)
            print('unzipped')
    if not os.path.exists(EVAL_PATH):
        print(f'eval doesn\'t exist\ndownloading eval \ndst: {EVAL_PATH}')
        misc.download_celeba_eval(EVAL_PATH)
        print('downloaded')


if __name__ == '__main__':
    prepare_celeba()
    CelebADataSet.make_mmap(PRE_DIR_PATH, MAP_DIR_PATH, True, 32)
    x_train, x_validate, x_test = load_celeba()

    print(x_train.shape)
    print(x_validate.shape)
    print(x_test.shape)

    # im = np.array(x_train[19])
    # im /= np.asarray(255., dtype=np.int32)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    # print(im)
    # plt.imshow(im)
    # plt.show()
