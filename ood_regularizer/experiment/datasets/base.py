from functools import partial

import numpy as np
import tfsnippet as spt

__all__ = [
    'registry', 'ImageDataSet', 'StandardImageDataSet',
]

registry = spt.utils.ClassRegistry()


class ImageDataSet(object):
    """Base class for image datasets."""

    @property
    def has_y(self):
        """Whether or not this dataset object has y array?"""
        raise NotImplementedError()

    @property
    def has_validation_data(self):
        """Whether or not this dataset object has validation data?"""
        raise NotImplementedError()

    @property
    def value_shape(self):
        """
        Get the image shape.

        Returns:
            (int, int, int): The `(height, weight, channels)` of image shape.
        """
        raise NotImplementedError()

    @property
    def value_size(self):
        """
        Get the size of each image, i.e., ``prod(value_shape)``.

        Returns:
            int: The size of each image.
        """
        return int(np.prod(self.value_shape))

    @property
    def color_depth(self):
        """
        Get the color depth of this dataset.

        k-bit images should have a color depth of 2**k.  For example,
        256-color images should have a color depth of 2**8 = 256.

        Returns:
            int: The color depth.

        Notes:
            The pixel values should always range from 0 ~ 255.
            For color depth < 256, the high `log2(color_depth)` bits
            are the bits which carry real color information, while
            the remaining low bits are not used.
        """
        return 256

    def train_flow(self, batch_size, use_y=False, shuffle=True,
                   skip_incomplete=True, random_state=None):
        """
        Get the data flow for training.

        Args:
            batch_size (int): Size of mini-batches.
            use_y (bool): Whether or not to use the labels?
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`True`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`True`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).

        Returns:
            DataFlow: The data flow for training.
        """
        raise NotImplementedError()

    @property
    def train_data_count(self):
        """Get the number of training data."""
        raise NotImplementedError()

    def valid_flow(self, batch_size, use_y=False, shuffle=True,
                   random_state=None):
        """
        Get the data flow for validation.

        Args:
            batch_size (int): Size of mini-batches.
            use_y (bool): Whether or not to use the labels?
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`True`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).

        Returns:
            DataFlow: The data flow for validation.
        """
        raise NotImplementedError()

    @property
    def valid_data_count(self):
        """Get the number of validation data."""
        raise NotImplementedError()

    def test_flow(self, batch_size, use_y=False, shuffle=True,
                  random_state=None):
        """
        Get the data flow for testing.

        Args:
            batch_size (int): Size of mini-batches.
            use_y (bool): Whether or not to use the labels?
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`True`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).

        Returns:
            DataFlow: The data flow for training.
        """
        raise NotImplementedError()

    @property
    def test_data_count(self):
        """Get the number of testing data."""
        raise NotImplementedError()


class StandardImageDataSet(ImageDataSet):
    """
    General implementation for standard image dataset which is loaded via
    a function `(x_shape, x_dtype, y_dtype, normalize_x) -> arrays`.
    """

    def __init__(self, loader_fn, value_shape, color_depth, has_y,
                 random_state=None, valid_size=0, from_file=False):
        self._value_shape = tuple(int(s) for s in value_shape)
        self._color_depth = int(color_depth)
        self._has_y = bool(has_y)
        self._valid_size = int(valid_size)
        random_state = random_state or \
                       np.random.RandomState(seed=spt.utils.generate_random_seed())

        # load data
        if has_y:
            loader_fn = partial(loader_fn, y_dtype=np.int32)
        data = loader_fn(x_shape=value_shape, x_dtype=np.uint8,
                         normalize_x=False)
        assert (isinstance(data, tuple))
        assert (len(data) == 2 or len(data) == 3)
        x_arrays = []
        y_arrays = [] if has_y else None

        def collect_x(x):
            if not from_file:
                assert (isinstance(x, np.ndarray))
                assert (x.dtype == np.uint8)
                assert (len(x.shape) == 4)
                assert (x.shape[-3:] == value_shape)
            x_arrays.append(x)

        def collect_y(y):
            assert (isinstance(y, np.ndarray))
            assert (y.dtype == np.int32)
            assert (len(y.shape) == 1)
            y_arrays.append(y)

        if has_y:
            for t in data:
                assert (isinstance(t, tuple))
                assert (len(t) == 2)
                collect_x(t[0])
                collect_y(t[1])
        else:
            for a in data:
                collect_x(a)

        # split or join validation set
        if len(x_arrays) == 3:
            if valid_size == 0:
                train_x = np.concatenate([x_arrays[0], x_arrays[1]], axis=0)
                x_arrays = [train_x, x_arrays[-1]]
                if has_y:
                    train_y = np.concatenate([y_arrays[0], y_arrays[1]], axis=0)
                    y_arrays = [train_y, y_arrays[-1]]
            else:
                assert (valid_size == len(x_arrays[1]))

        else:
            if valid_size > 0:
                if has_y:
                    [train_x, train_y], [valid_x, valid_y] = \
                        spt.utils.split_numpy_arrays(
                            [x_arrays[0], y_arrays[0]],
                            size=valid_size,
                            shuffle=True,
                            random_state=random_state
                        )
                    x_arrays = [train_x, valid_x, x_arrays[-1]]
                    y_arrays = [train_y, valid_y, y_arrays[-1]]
                else:
                    [train_x], [valid_x] = \
                        spt.utils.split_numpy_arrays(
                            [x_arrays[0]],
                            size=valid_size,
                            shuffle=True,
                            random_state=random_state
                        )
                    x_arrays = [train_x, valid_x, x_arrays[-1]]
        self._x_arrays = tuple(x_arrays)
        self._y_arrays = y_arrays and tuple(y_arrays)

    @property
    def has_y(self):
        return self._has_y

    @property
    def has_validation_data(self):
        return self._valid_size > 0

    @property
    def value_shape(self):
        return self._value_shape

    def train_flow(self, batch_size, use_y=False, shuffle=True,
                   skip_incomplete=True, random_state=None):
        if use_y and not self._has_y:
            raise RuntimeError('Dataset class {} does not provide y data.'.
                               format(self.__class__.__name__))
        arrays = [self._x_arrays[0]]
        if use_y:
            arrays.append(self._y_arrays[0])
        return spt.DataFlow.arrays(
            arrays, batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state
        )

    @property
    def train_data_count(self):
        return len(self._x_arrays[0])

    def valid_flow(self, batch_size, use_y=False, shuffle=True,
                   random_state=None):
        if self.valid_data_count == 0:
            raise RuntimeError('Dataset object {} does not have validation '
                               'data.'.format(self))
        if use_y and not self._has_y:
            raise RuntimeError('Dataset class {} does not provide y data.'.
                               format(self.__class__.__name__))
        arrays = [self._x_arrays[1]]
        if use_y:
            arrays.append(self._y_arrays[1])
        return spt.DataFlow.arrays(
            arrays, batch_size=batch_size, shuffle=shuffle,
            random_state=random_state
        )

    @property
    def valid_data_count(self):
        return self._valid_size

    def test_flow(self, batch_size, use_y=False, shuffle=True,
                  random_state=None):
        if use_y and not self._has_y:
            raise RuntimeError('Dataset class {} does not provide y data.'.
                               format(self.__class__.__name__))
        arrays = [self._x_arrays[-1]]
        if use_y:
            arrays.append(self._y_arrays[-1])
        return spt.DataFlow.arrays(
            arrays, batch_size=batch_size, shuffle=shuffle,
            random_state=random_state
        )

    @property
    def test_data_count(self):
        return len(self._x_arrays[-1])
