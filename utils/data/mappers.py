from enum import Enum
from typing import *

import mltk
import numpy as np

from .types import *

__all__ = [
    'ArrayMapper', 'ArrayMapperList',
    'Identity', 'Reshape', 'Flatten', 'Transpose', 'Pad',
    'ChannelTranspose', 'ChannelFirstToLast', 'ChannelLastToFirst',
    'ChannelLastToDefault', 'ChannelFirstToDefault',
    'Affine', 'ScaleToRange', 'ReduceToBitDepth', 'Dequantize',
    'BernoulliSample',
    'DownSample', 'UpSample',
    'GrayscaleToRGB', 'CropImage', 'ScaleImageMode', 'ScaleImage',
]

NumberType = Union[int, float, np.ndarray]


class ArrayMapper(object):

    _input_info: ArrayInfo = None
    _output_info: ArrayInfo = None

    @property
    def fitted(self) -> bool:
        return self._input_info is not None

    @property
    def input_info(self) -> ArrayInfo:
        if self._input_info is None:
            raise RuntimeError(f'`fit()` has not been called: {self!r}')
        return self._input_info

    @property
    def output_info(self) -> ArrayInfo:
        if self._output_info is None:
            raise RuntimeError(f'`fit()` has not been called: {self!r}')
        return self._output_info

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        raise NotImplementedError()

    def fit(self, info: ArrayInfo) -> ArrayInfo:
        if self._input_info is not None:
            raise RuntimeError(f'`fit()` has already been called: {self!r}')
        self._input_info = info
        self._output_info = self._fit(info)
        return self._output_info

    def fit_dataset(self, dataset, slot: str) -> ArrayInfo:
        return self.fit(dataset.slots[slot])

    def transform(self, array: mltk.Array) -> mltk.Array:
        raise NotImplementedError()

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        raise NotImplementedError()


class ArrayMapperList(ArrayMapper, Sequence[ArrayMapper]):

    mappers: List[ArrayMapper]

    def __init__(self, mappers: Iterable[ArrayMapper]):
        mappers = list(mappers)
        if mappers:
            fitted = mappers[0].fitted
            for mapper in mappers[1:]:
                if mapper.fitted != fitted:
                    raise ValueError(f'The `mappers` must be all fitted or '
                                     f'all not fitted.')
            if fitted:
                self._input_info = mappers[0].input_info
                self._output_info = mappers[-1].output_info
        self.mappers = mappers

    def __getitem__(self, item: int) -> ArrayMapper:
        return self.mappers[item]

    def __len__(self) -> int:
        return len(self.mappers)

    def __iter__(self) -> Iterator[ArrayMapper]:
        return iter(self.mappers)

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        for mapper in self.mappers:
            info = mapper.fit(info)
        return info

    def transform(self, array: mltk.Array) -> mltk.Array:
        for mapper in self.mappers:
            array = mapper.transform(array)
        return array

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        for mapper in reversed(self.mappers):
            array = mapper.inverse_transform(array, strict=strict)
        return array


class Identity(ArrayMapper):

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        return info

    def transform(self, array: mltk.Array) -> mltk.Array:
        return array

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        return array


class Reshape(ArrayMapper):

    in_shape: List[int]
    out_shape: List[int]

    def __init__(self, shape: List[int]):
        out_shape = []
        neg_one_count = 0
        for s in shape:
            if s == -1:
                if neg_one_count > 0:
                    raise ValueError(f'At most one `-1` can be present in '
                                     f'`shape`: got {shape!r}')
                neg_one_count += 1
            elif s < 0:
                raise ValueError(f'Not a valid shape: {shape!r}')
            out_shape.append(s)

        self.out_shape = out_shape

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_shape(deterministic=False)
        self.in_shape = list(info.shape)
        in_size = int(np.prod(self.in_shape))
        out_size = int(np.prod(self.out_shape))
        if out_size >= 0 and out_size != in_size or \
                out_size < 0 and in_size % (-out_size) != 0:
            raise ValueError(f'Cannot reshape array from {self.in_shape!r} to '
                             f'{self.out_shape}')
        if out_size < 0:
            for i, s in enumerate(self.out_shape):
                if s == -1:
                    self.out_shape[i] = in_size // (-out_size)
                    break
        return info.copy(shape=self.out_shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        arr_shape = mltk.utils.get_array_shape(array)
        pos = len(arr_shape) - len(self.in_shape)
        return np.reshape(array, list(arr_shape[:pos]) + self.out_shape)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        arr_shape = mltk.utils.get_array_shape(array)
        pos = len(arr_shape) - len(self.out_shape)
        return np.reshape(array, list(arr_shape[:pos]) + self.in_shape)


class Flatten(Reshape):

    def __init__(self):
        super().__init__([-1])


class Transpose(ArrayMapper):

    perm: List[int]
    inverse_perm: List[int]

    def __init__(self, perm: List[int]):
        perm = list(map(int, perm))
        for a in perm:
            if a >= 0:
                raise ValueError(f'`perm` must all be negative integers.')
        reverse_axis = [0] * len(perm)
        for i in range(-len(perm), 0):
            reverse_axis[perm[i]] = i
        self.perm = perm
        self.inverse_perm = reverse_axis

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_shape(deterministic=False)
        in_shape = info.shape
        if len(in_shape) != len(self.perm):
            raise ValueError(f'The input shape is required to be '
                             f'{len(self.perm)}d for transpose axis {self.perm!r}: '
                             f'got input shape {in_shape!r}')
        new_shape = []
        for a in self.perm:
            new_shape.append(in_shape[a])
        return info.copy(shape=new_shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        front_perm = list(range(0, len(mltk.utils.get_array_shape(array)) - len(self.perm)))
        return np.transpose(array, front_perm + self.perm)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        front_perm = list(range(0, len(mltk.utils.get_array_shape(array)) - len(self.inverse_perm)))
        return np.transpose(array, front_perm + self.inverse_perm)


class Pad(ArrayMapper):

    fill_value: Union[int, float]
    padding: List[Tuple[int, int]]
    inv_slices: List[slice]

    def __init__(self,
                 padding: Sequence[Tuple[int, int]],
                 fill_value: Union[int, float] = 0):
        self.padding = list(padding)
        self.fill_value = fill_value
        self.inv_slices = [
            slice(l, -r) if r > 0 else (
                slice(l, None) if l > 0 else slice(None))
            for l, r in self.padding
        ]

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        if len(info.shape) < len(self.padding):
            raise ValueError(
                f'`info.shape` must be at least {len(self.padding)}d: '
                f'got shape {info.shape}')
        shape = list(info.shape)
        for i, (s, (l, r)) in enumerate(
                zip(reversed(info.shape), reversed(self.padding)), 1):
            shape[-i] = s + l + r
        return info.copy(shape=shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        arr_shape = mltk.utils.get_array_shape(array)
        padding = ([(0, 0)] * (len(arr_shape) - len(self.padding))) + self.padding
        return np.pad(array, padding, mode='constant', constant_values=self.fill_value)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        arr_shape = mltk.utils.get_array_shape(array)
        slc = (
            [slice(None)] * (len(arr_shape) - len(self.inv_slices)) +
            self.inv_slices
        )
        return array[slc]


class ChannelTranspose(ArrayMapper):

    from_format: ChannelFormat
    to_format: ChannelFormat
    internal_mapper: ArrayMapper

    def __init__(self,
                 from_format: Union[str, ChannelFormat],
                 to_format: Union[str, ChannelFormat]):
        self.from_format = ChannelFormat(from_format)
        self.to_format = ChannelFormat(to_format)

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_shape(deterministic=False)
        spatial_ndims = len(info.shape) - 1
        if spatial_ndims not in (1, 2, 3):
            raise ValueError(f'Shape not supported by `ChannelTranspose`: '
                             f'shape is {info.shape!r}')

        if self.from_format == self.to_format:
            perm = None
        elif self.from_format == 'channel_last':
            perm = [-1] + list(range(-spatial_ndims - 1, -1))
        else:
            perm = list(range(-spatial_ndims, 0)) + [-spatial_ndims - 1]

        if perm is None:
            self.internal_mapper = Identity()
        else:
            self.internal_mapper = Transpose(perm)
        return self.internal_mapper.fit(info)

    def transform(self, array: mltk.Array) -> mltk.Array:
        return self.internal_mapper.transform(array)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        return self.internal_mapper.inverse_transform(array, strict)


class ChannelLastToFirst(ChannelTranspose):
    def __init__(self):
        super().__init__(ChannelFormat.CHANNEL_LAST, ChannelFormat.CHANNEL_FIRST)


class ChannelFirstToLast(ChannelTranspose):
    def __init__(self):
        super().__init__(ChannelFormat.CHANNEL_FIRST, ChannelFormat.CHANNEL_LAST)


class ChannelLastToDefault(ChannelTranspose):
    def __init__(self):
        from tensorkit import tensor as T
        super().__init__(
            ChannelFormat.CHANNEL_LAST,
            ChannelFormat.CHANNEL_LAST if T.IS_CHANNEL_LAST else ChannelFormat.CHANNEL_FIRST
        )


class ChannelFirstToDefault(ChannelTranspose):
    def __init__(self):
        from tensorkit import tensor as T
        super().__init__(
            ChannelFormat.CHANNEL_FIRST,
            ChannelFormat.CHANNEL_LAST if T.IS_CHANNEL_LAST else ChannelFormat.CHANNEL_FIRST
        )


def maybe_apply_affine(x, scale, bias):
    if x is not None:
        return x * scale + bias


def maybe_clip(x, low, high):
    if low is not None and high is not None:
        return np.clip(x, low, high)
    elif low is None and high is None:
        return x
    elif low is not None:
        return np.maximum(x, low)
    elif high is not None:
        return np.minimum(x, high)


def is_integer_dtype(dtype: str) -> bool:
    return dtype in ('int8', 'uint8', 'int16', 'int32', 'int64', 'int', 'long')


class _BaseAffine(ArrayMapper):
    need_transform: bool = True
    in_dtype: str
    in_dtype_is_int: bool
    out_dtype: str
    out_dtype_is_int: bool
    in_range: Tuple[Optional[NumberType], Optional[NumberType]]
    out_range: Tuple[Optional[NumberType], Optional[NumberType]]
    scale: NumberType
    bias: NumberType

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        self.in_dtype = info.dtype
        self.in_dtype_is_int = is_integer_dtype(self.in_dtype)
        self.out_dtype_is_int = is_integer_dtype(self.out_dtype)
        self.in_range = (info.min_val, info.max_val)
        return info.copy(
            dtype=self.out_dtype,
            min_val=self.out_range[0],
            max_val=self.out_range[1],
        )

    def transform(self, array: mltk.Array) -> mltk.Array:
        if not self.need_transform:
            return array
        array = maybe_clip(array * self.scale + self.bias, *self.out_range)
        if self.out_dtype_is_int:
            array = np.round(array)
        return array.astype(self.out_dtype)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        if not self.need_transform:
            return array
        array = maybe_clip((array - self.bias) / float(self.scale), *self.in_range)
        if self.in_dtype_is_int:
            array = np.round(array)
        return array.astype(self.in_dtype)


class Affine(_BaseAffine):
    """Scale the input array by affine transformation `Ax+b`."""

    def __init__(self, scale: NumberType, bias: NumberType,
                 dtype: str = FLOAT_X):
        self.out_dtype = dtype
        self.scale = scale
        self.bias = bias

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        self.out_range = (
            maybe_apply_affine(info.min_val, self.scale, self.bias),
            maybe_apply_affine(info.max_val, self.scale, self.bias),
        )
        return super()._fit(info)


class ScaleToRange(_BaseAffine):
    """Scale the input array to a specific value range."""

    min_val: NumberType
    max_val: NumberType

    def __init__(self, min_val: NumberType, max_val: NumberType,
                 dtype: str = FLOAT_X):
        self.out_dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.out_range = (self.min_val, self.max_val)

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_min_max_val()
        self.need_transform = (self.min_val != info.min_val) or (self.max_val != info.max_val)
        self.scale = (self.max_val - self.min_val) / float(info.max_val - info.min_val)
        self.bias = self.min_val - info.min_val * self.scale
        return super()._fit(info)


class ReduceToBitDepth(ArrayMapper):
    """Reduce the bit depth of a discrete array."""

    bit_depth: int
    bit_depth_diff: int

    dtype_is_int: bool
    need_transform: bool
    in_scale: NumberType
    out_bin_size: NumberType
    min_val: NumberType

    def __init__(self, bit_depth: int):
        self.bit_depth = bit_depth

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_discrete()
        info.require_min_max_val()
        if 2 ** info.bit_depth != info.n_discrete_vals:
            raise ValueError(f'`info.n_discrete_vals != 2 ** info.bit_depth`.')
        if info.bit_depth < self.bit_depth:
            raise ValueError(f'Cannot enlarge bit-depth with `ReduceToBitDepth` mapper.')

        new_bit_depth = min(info.bit_depth, self.bit_depth)
        bit_depth_diff = info.bit_depth - new_bit_depth
        new_n_vals = 2 ** new_bit_depth  # the new n_discrete_vals
        in_bin_size = (info.max_val - info.min_val) / (info.n_discrete_vals - 1.)
        out_bin_size = in_bin_size * (2 ** bit_depth_diff)
        need_transform = new_bit_depth != info.bit_depth

        self.need_transform = need_transform
        if not need_transform:
            return info

        self.bit_depth_diff = bit_depth_diff
        self.dtype_is_int = is_integer_dtype(info.dtype)
        self.in_bin_size = in_bin_size
        self.out_bin_size = out_bin_size
        self.min_val = info.min_val
        self.max_val = self.min_val + out_bin_size * (new_n_vals - 1)
        if self.dtype_is_int:
            self.in_bin_size = int(round(self.in_bin_size))
            self.out_bin_size = int(round(self.out_bin_size))
            self.max_val = int(round(self.max_val))

        return info.copy(
            n_discrete_vals=new_n_vals,
            bit_depth=new_bit_depth,
            max_val=self.max_val
        )

    def transform(self, array: mltk.Array) -> mltk.Array:
        if not self.need_transform:
            return array

        ret = np.round((array - self.min_val) / self.in_bin_size).astype(np.int32)
        ret = ret >> self.bit_depth_diff
        ret = self.min_val + ret * self.out_bin_size
        if self.dtype_is_int:
            ret = np.round(ret)
        ret = ret.astype(array.dtype)
        return ret

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        if strict and self.need_transform:
            raise RuntimeError(f'`ReduceToBitDepth` is not strictly invertible.')
        return array


class Dequantize(ArrayMapper):
    """Adds uniform noise to discrete array, making it continuous."""

    in_dtype: str
    out_dtype: str
    bin_size: NumberType
    in_min_val: NumberType

    # small infinitesimal to ensure the generated noise reside in [-0.5, 0.5),
    # rather than [-0.5, 0.5]
    epsilon: float

    def __init__(self,
                 dtype: str = FLOAT_X,
                 epsilon: float = 0.0):
        self.out_dtype = dtype
        self.epsilon = epsilon

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_discrete()
        info.require_min_max_val()
        self.in_dtype = info.dtype
        self.in_min_val = info.min_val
        self.bin_size = np.asarray(
            (info.max_val - info.min_val) / (info.n_categories - 1.),
            dtype=self.out_dtype
        )
        return info.copy(
            dtype=self.out_dtype,
            min_val=float(info.min_val - 0.5 * self.bin_size),
            max_val=float(info.max_val + 0.5 * self.bin_size),
            is_discrete=False,
        )

    def transform(self, array: mltk.Array) -> mltk.Array:
        noise = np.random.random(size=mltk.utils.get_array_shape(array))
        if self.epsilon > 0.:
            noise = np.minimum(noise, 1. - self.epsilon)
        noise = (noise - 0.5) * self.bin_size
        array = array + noise
        return array.astype(self.out_dtype)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        array = self.in_min_val + self.bin_size * np.asarray(
            (array - self.output_info.min_val) / self.bin_size,
            dtype=np.int32
        )
        return array.astype(self.in_dtype)


class BernoulliSample(ArrayMapper):

    in_dtype: str
    out_dtype: str

    def __init__(self, dtype: str = 'int32'):
        self.out_dtype = dtype

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        if info.is_discrete or info.min_val != 0 or info.max_val != 1:
            raise ValueError('The source array values are not continuous, or '
                             'not within the range [0, 1]. ')
        self.in_dtype = info.dtype
        return info.copy(dtype=self.out_dtype, is_discrete=True,
                         n_discrete_vals=2, bit_depth=1)

    def transform(self, array: mltk.Array) -> mltk.Array:
        return np.random.binomial(n=1, p=array).astype(self.out_dtype)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        if strict:
            raise RuntimeError('`BernoulliSampler` is not strictly invertible.')
        return np.asarray(array, dtype=self.in_dtype)


def down_sample(array, scale: List[int]):
    shape = mltk.utils.get_array_shape(array)
    n = len(shape)
    k = len(scale)
    if len(shape) < k:
        raise ValueError(f'`array` must be at least {k}d.')

    temp_shape = list(shape[:n - k])
    reduce_axis = []
    next_axis = n - k

    for a, b in zip(shape[n-k:], scale):
        if a % b != 0:
            raise ValueError(
                f'`array.shape` cannot be evenly divided by `scale`: '
                f'shape {shape} vs scale {scale}')
        temp_shape.extend((a // b, b))
        reduce_axis.append(next_axis + 1)
        next_axis += 2

    array = np.reshape(array, temp_shape)
    array = np.mean(array, axis=tuple(reduce_axis), keepdims=False)
    return array


def up_sample(array, scale: List[int]):
    shape = mltk.utils.get_array_shape(array)
    n = len(shape)
    k = len(scale)
    if len(shape) < k:
        raise ValueError(f'`array` must be at least {k}d.')

    temp_shape = list(shape[:n - k])
    out_shape = list(temp_shape)
    tile_rep = [1] * (n - k)

    for a, b in zip(shape[n-k:], scale):
        temp_shape.extend((a, 1))
        out_shape.append(a * b)
        tile_rep.extend((1, b))

    array = np.reshape(array, temp_shape)
    array = np.tile(array, tile_rep)
    array = array.reshape(out_shape)
    return array


class DownSample(ArrayMapper):
    """Down-sampling by averaging over multiple pixels."""

    in_dtype: str
    out_dtype: str
    scale: List[int]

    def __init__(self, scale: Sequence[int], dtype=FLOAT_X):
        self.scale = list(scale)
        self.out_dtype = dtype

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_shape(deterministic=False)
        shape = list(info.shape)
        k = len(self.scale)
        if len(shape) < k:
            raise ValueError(f'`info.shape` must be at least {k}d.')

        for i, (size, ratio) in enumerate(
                zip(reversed(shape), reversed(self.scale))):
            if size is not None:
                if size % ratio != 0:
                    raise ValueError(
                        f'`info.shape` cannot be evenly divided by `scale`: '
                        f'shape {info.shape} vs scale {self.scale}')
                shape[-(i + 1)] = size // ratio

        self.in_dtype = info.dtype
        return info.copy(dtype=self.out_dtype, shape=shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        return down_sample(array, self.scale).astype(self.out_dtype)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        if strict:
            raise RuntimeError('`DownSample` is not strictly invertible.')
        return up_sample(array, self.scale).astype(self.in_dtype)


class UpSample(ArrayMapper):
    in_dtype: str
    out_dtype: str
    scale: List[int]

    def __init__(self, scale: Sequence[int], dtype=FLOAT_X):
        self.scale = list(scale)
        self.out_dtype = dtype

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        info.require_shape(deterministic=False)
        shape = list(info.shape)
        k = len(self.scale)
        if len(shape) < k:
            raise ValueError(f'`info.shape` must be at least {k}d.')

        for i, (size, ratio) in enumerate(
                zip(reversed(shape), reversed(self.scale))):
            if size is not None:
                shape[-(i + 1)] = size * ratio

        self.in_dtype = info.dtype
        return info.copy(dtype=self.out_dtype, shape=shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        return up_sample(array, self.scale).astype(self.out_dtype)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        return down_sample(array, self.scale).astype(self.in_dtype)


class BaseImageMapper(ArrayMapper):

    channel_last: Optional[bool]

    def __init__(self, channel_last: Optional[bool] = None):
        self.channel_last = channel_last

    def _fit_channel_axis(self, info: ArrayInfo):
        info.require_shape(False)

        if len(info.shape) != 3:
            raise ValueError(f'Invalid shape: {info.shape}')

        if self.channel_last is None:
            if (info.shape[-1] in (1, 3)) == (info.shape[-3] in (1, 3)):
                raise ValueError('`channel_last` cannot be determined automatically.')
            self.channel_last = info.shape[-1] in (1, 3)

        if (self.channel_last and info.shape[-1] not in (1, 3)) or \
                (not self.channel_last and info.shape[-3] not in (1, 3)):
            raise ValueError(f'Invalid shape {info.shape!r} for `channel_last` '
                             f'{self.channel_last!r}')

    def _get_spatial_shape(self, shape):
        if self.channel_last is True:
            return shape[:-1]
        elif self.channel_last is False:
            return shape[1:]
        else:
            return shape

    def _replace_spatial_shape(self, shape, spatial):
        if self.channel_last is True:
            return list(spatial) + [shape[-1]]
        elif self.channel_last is False:
            return [shape[0]] + list(spatial)
        else:
            return list(spatial)


class GrayscaleToRGB(BaseImageMapper):

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        self._fit_channel_axis(info)
        shape = list(info.shape)
        if self.channel_last:
            if shape[-1] != 1:
                raise ValueError(f'Invalid shape: {shape}')
            shape[-1] = 3
        else:
            if shape[-3] != 1:
                raise ValueError(f'Invalid shape: {shape}')
            shape[-3] = 3
        return info.copy(shape=shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        shape = mltk.utils.get_array_shape(array)
        reps = [1] * len(shape)
        reps[-1 if self.channel_last else -3] = 3
        return np.tile(array, reps)

    def inverse_transform(self,
                          array: mltk.Array,
                          strict: bool = False) -> mltk.Array:
        if self.channel_last:
            return array[..., 0:1]
        else:
            return array[..., 0:1, :, :]


class CropImage(BaseImageMapper):

    pos: Tuple[int, int]
    size: Tuple[int, int]

    def __init__(self,
                 *,
                 bbox: Optional[Sequence[int]] = None,  # (top, bottom, left, right)
                 pos: Optional[Sequence[int]] = None,
                 size: Optional[Sequence[int]] = None,
                 channel_last: Optional[bool] = None):
        if not ((bbox is None and pos is not None and size is not None) or
                (bbox is not None and pos is None and size is None)):
            raise ValueError('Either `bbox`, or a pair of `pos` and `size` '
                             'should be specified, but not both.')

        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError(f'`bbox` must be a sequence of 4 integers.')
            pos = (bbox[0], bbox[2])
            size = (bbox[1] - bbox[0], bbox[3] - bbox[2])
        else:
            if len(pos) != 2 or len(size) != 2:
                raise ValueError(f'`pos` and `size` must be sequences of 2 '
                                 f'integers.')
            pos = tuple(pos)
            size = tuple(size)

        super().__init__(channel_last)
        self.pos = pos
        self.size = size

    def _check_shape_against_bbox(self, shape):
        if (shape[0] is not None and self.size[0] > shape[0]) or \
                (shape[1] is not None and self.size[1] > shape[1]):
            raise ValueError(f'Spatial shape `{shape!r}` cannot be cropped: pos '
                             f'= {self.pos}, size = {self.size}.')

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        self._fit_channel_axis(info)
        self._check_shape_against_bbox(self._get_spatial_shape(info.shape))
        new_shape = self._replace_spatial_shape(info.shape, list(self.size))
        return info.copy(shape=new_shape)

    def transform(self, array: mltk.Array) -> mltk.Array:
        p = self.pos
        s = self.size
        if self.channel_last is True:
            return array[..., p[0]: p[0] + s[0], p[1]: p[1] + s[1], :]
        else:
            return array[..., p[0]: p[0] + s[0], p[1]: p[1] + s[1]]

    def inverse_transform(self, array: mltk.Array, strict: bool = False) -> mltk.Array:
        raise RuntimeError('`CropImage` is not invertible.')


class ScaleImageMode(int, Enum):

    SCIPY_NO_AA = 0
    """Use no antialias method with SciPy resize kernel."""

    SCIPY_GAUSSIAN_AA = 1
    """Use the default gaussian filter antialias method with SciPy resize kernel."""

    CELEBA_GAUSSIAN_AA = 99
    """
    The scale method from:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """


class ScaleImage(BaseImageMapper):

    mode: ScaleImageMode
    resize_kernel: Callable[[np.ndarray], np.ndarray]
    value_range: Tuple[Union[int, float]]
    size: List[int]

    def __init__(self,
                 size: Sequence[int],  # (height, width)
                 mode: Union[ScaleImageMode, int] = ScaleImageMode.SCIPY_GAUSSIAN_AA,
                 channel_last: Optional[bool] = None):
        if len(size) != 2:
            raise ValueError(f'`size` must be a sequence of two integers.')

        super().__init__(channel_last)
        self.mode = ScaleImageMode(mode)
        self.size = list(size)
        self.resize_kernel = self._scipy_kernel

    def _fit(self, info: ArrayInfo) -> ArrayInfo:
        self._fit_channel_axis(info)
        r = (info.min_val, info.max_val)
        if r != (0, 255) and r != (0, 1):
            raise ValueError(f'Images pixel values must within range (0, 255) '
                             f'or (0, 1): got {r!r}')
        self.value_range = r
        return info.copy(shape=self._replace_spatial_shape(info.shape, self.size))

    def _scipy_kernel(self, img: np.ndarray):
        from skimage import transform, filters
        dtype = img.dtype

        # ensure axis order: (H, W, C)
        if self.channel_last is False:
            img = img.transpose([-2, -1, -3])

        # get the spatial shape
        shape = img.shape[-3: -1] if self.channel_last is True else img.shape[-2:]

        # ensure image value range is in (0, 1)
        if self.value_range[1] == 255:
            img = img.astype(np.float32) / 255.

        # scale the image
        new_size = tuple(self.size + [img.shape[-1]])

        if self.mode == ScaleImageMode.SCIPY_NO_AA:
            img = transform.resize(img, new_size, anti_aliasing=False)

        elif self.mode == ScaleImageMode.SCIPY_GAUSSIAN_AA:
            img = transform.resize(img, new_size, anti_aliasing=True)

        elif self.mode == ScaleImageMode.CELEBA_GAUSSIAN_AA:
            # https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
            scale = (shape[0] * 1. / self.size[0], shape[1] * 1. / self.size[1])
            sigma = (np.sqrt(scale[0]) / 2., np.sqrt(scale[1]) / 2.)
            img = filters.gaussian(img, sigma=sigma, multichannel=len(img.shape) > 2)
            img = transform.resize(
                img, tuple(self.size + [img.shape[-1]]), order=3,
                # Turn off anti-aliasing, since we have done gaussian filtering.
                # Note `anti_aliasing` defaults to `True` until skimage >= 0.15,
                # which version is released in 2019/04, while the repo
                # `andersbll/autoencoding_beyond_pixels` was released in 2015.
                anti_aliasing=False,
                # same reason as above
                mode='constant',
            )

        # scale back to the original range
        if self.value_range[1] == 255:
            img = (img * 255)

        # back to the original axis order
        if self.channel_last is False:
            img = img.transpose([-1, -3, -2])

        # return the image
        img = img.astype(dtype)
        return img

    def transform(self, array: mltk.Array) -> mltk.Array:
        value_ndims = 3
        shape = mltk.utils.get_array_shape(array)
        if len(shape) < value_ndims:
            raise ValueError(f'`array` must be at least {value_ndims}d: '
                             f'got shape {shape!r}.')

        front_shape, back_shape = shape[:-value_ndims], shape[-value_ndims:]
        array = np.reshape(array, [-1] + list(back_shape))
        array = np.stack([self.resize_kernel(np.asarray(im)) for im in array],
                         axis=0)
        array = np.reshape(array, front_shape + array.shape[-value_ndims:])
        return array

    def inverse_transform(self, array: mltk.Array, strict: bool = False) -> mltk.Array:
        raise RuntimeError(f'`ScaleImage` is not invertible.')
