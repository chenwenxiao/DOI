from enum import Enum
from typing import *

import mltk
import numpy as np
from imgaug import augmenters as iaa

from utils.data import *
from utils.data.mappers import ArrayMapperList

__all__ = [
    'InMemoryDataSetName',
    'DataSetConfig',
    'ImageAugmentationMapper',
    'make_dataset',
    'get_mapper'
]


class InMemoryDataSetName(str, Enum):
    MNIST28 = 'mnist28'
    KMINIST28 = 'kmnist28'
    NOTMNIST28 = 'not_mnist28'
    FASHIONMIST28 = 'fashion_mnist28'
    OMNNIGLOT28 = 'omniglot28'
    NOISE28 = 'noise28'
    CONSTANT28 = 'constant28'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    SVHN = 'svhn'
    CELEBA = 'celeba'
    LSUN = 'lsun'
    ISUN = 'isun'
    TINYIMAGENET = 'tinyimagenet'
    NOISE = 'noise'
    CONSTANT = 'constant'


class DataSetConfig(mltk.Config):
    name: Optional[InMemoryDataSetName] = None
    """Name of in-memory dataset."""

    mmap_dir: Optional[str] = None
    """Root directory of mmap dataset."""

    enable_grayscale_to_rgb: bool = True
    """Convert grayscale images to RGB images (if required by the dataset)."""

    enable_train_aug: bool = True
    """Enable train data augmentation (if required by the dataset)."""

    @mltk.root_checker()
    def _root_checker(self, obj: 'DataSetConfig'):
        if (obj.name is None) == (obj.mmap_dir is None):
            raise ValueError(
                f'Either `name` or `mmap_dir` should be specified, '
                f'but not both: got `name` {obj.name!r}, '
                f'`mmap_dir` {obj.mmap_dir!r}'
            )


class ImageAugmentationMapper(mappers.ArrayMapper):
    aug: iaa.Augmenter

    def __init__(self, aug: iaa.Augmenter):
        super().__init__()
        self.aug = aug

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        return info

    def transform(self, array: mltk.Array) -> mltk.Array:
        return self.aug(images=np.asarray(array))

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        if strict:
            raise ValueError(f'`ImageAugmentationMapper` is not strictly invertible.')
        return array


def make_dataset(config: DataSetConfig) -> Tuple[DataSet, DataSet, DataSet]:
    """
    Construct train and test dataset objects according to the config.

    Args:
        config: The dataset config object.

    Returns:
        The train and test dataset objects.
    """
    # construct the original dataset object
    if config.name is None:
        dataset = MMapDataSet(config.mmap_dir)
    elif config.name[-2:] == '28':
        dataset = BaseMNISTLike(config.name)
    else:
        dataset = BaseCifar(config.name)

    train_dataset = dataset.apply_mappers(x=get_mapper(config, training=True))

    # test dataset
    test_dataset = dataset.apply_mappers(x=get_mapper(config, training=False))

    return train_dataset, test_dataset, dataset


def get_mapper(config: DataSetConfig, training=False):
    # assemble the pipelines
    def common_mappers():
        m = []
        if config.name[-2:] == '28':
            m.append(mappers.Pad([(2, 2), (2, 2), (0, 0)]))  # pad to 32x32x1
            if config.enable_grayscale_to_rgb:
                m.append(mappers.GrayscaleToRGB())  # copy to 32x32x3
        m.extend([
            mappers.Dequantize(),
            mappers.ScaleToRange(-1., 1.),
            mappers.ChannelLastToDefault(),
        ])
        return m

    # train dataset
    m = []
    if training:
        if config.enable_train_aug:
            print('Affine augmentation added.')
            aug = iaa.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                # order=3,  # turn on this if not just translation
                mode='edge',
                backend='cv2'
            )
            m.append(ImageAugmentationMapper(aug))
    m.extend(common_mappers())
    return ArrayMapperList(m)
