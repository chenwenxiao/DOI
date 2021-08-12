import numpy as np
import os

dataset_path = ""
__all_datasets__ = ['mnist28', 'fashion_mnist28', 'kmnist28', 'not_mnist28', 'omniglot28', 'celeba', 'tinyimagenet', 'svhn',
                    'cifar10', 'cifar100', 'isun', 'lsun', 'constant', 'noise', 'constant28', 'noise28']


def try_load(dataset_name, suffix):
    if dataset_path == "":
        raise RuntimeError("You need specify the dataset_path in experiments/datasets/overall.py")
    train_complexity_path = dataset_path + dataset_name + '/' + suffix + '.npy'
    if os.path.exists(train_complexity_path):
        return np.load(train_complexity_path)
    return None


def load_overall(dataset_name, dtype=np.uint8):
    x_train = try_load(dataset_name, 'train')
    x_test = try_load(dataset_name, 'test')
    y_train = try_load(dataset_name, 'train_label')
    y_test = try_load(dataset_name, 'test_label')

    if x_train is None:
        x_train = x_test
    x_train = x_train.astype(dtype)
    x_test = x_test.astype(dtype)
    if y_train is None:
        y_train = np.random.randint(0, 10, len(x_train))
    if y_test is None:
        y_test = np.random.randint(0, 10, len(x_test))
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def load_complexity(dataset_name, compressor):
    x_train_complexity = try_load(dataset_name, 'train_complexity')
    x_test_complexity = try_load(dataset_name, 'test_complexity')
    if x_train_complexity is None:
        x_train_complexity = x_test_complexity
    print(x_train_complexity.shape, x_test_complexity.shape)
    if dataset_name[-2:] != '28':
        x_train_complexity = x_train_complexity / (32 * 32 * 3 * np.log(2))
        x_test_complexity = x_test_complexity / (32 * 32 * 3 * np.log(2))
    else:
        x_train_complexity = x_train_complexity / (28 * 28 * 1 * np.log(2))
        x_test_complexity = x_test_complexity / (28 * 28 * 1 * np.log(2))

    return x_train_complexity[..., compressor], x_test_complexity[..., compressor]
