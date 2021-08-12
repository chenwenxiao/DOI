import tempfile
from typing import *

import imageio
import numpy as np

from .types import ArrayInfo

__all__ = ['image_array_to_rgb', 'make_images_grid', 'save_image_to_file']


def _check_images_channel(images: np.ndarray,
                          channel_last: Optional[bool] = None,
                          require_one_image: bool = False) -> np.ndarray:
    if require_one_image and len(images.shape) != 3:
        raise ValueError(f'`image` must be exactly 3d: got shape {images.shape!r}')
    if len(images.shape) < 3:
        raise ValueError(f'`images` must be at least 3d array: got shape {images.shape!r}')

    # detect channel order
    if channel_last is None:
        if (images.shape[-1] in (1, 3)) == (images.shape[-3] in (1, 3)):
            raise ValueError(f'Cannot detect `channel_last` automatically.  '
                             f'Please specify this argument manually.')
        channel_last = images.shape[-1] in (1, 3)
    else:
        if (channel_last and images.shape[-1] not in (1, 3)) or \
                (not channel_last and images.shape[-3] not in (1, 3)):
            raise ValueError(f'Invalid image(s) shape: {images.shape!r}')

    # change to channel-last
    if not channel_last:
        images = images.transpose(
            list(range(0, len(images.shape) - 3)) +
            [-2, -1, -3]
        )

    return images


def image_array_to_rgb(images: np.ndarray,
                       info: Optional[ArrayInfo] = None,
                       channel_last: Optional[bool] = None) -> np.ndarray:
    """
    Convert image array(s) into RGB pixel values.

    Args:
        images: The input image array.
        info: The optional array information.  If not specified, will attempt
            to detect the input images format.
        channel_last: Whether or not the channel axis is the last axis?
            If not specified, will detect automatically.

    Returns:
        The converted images array, whose dtype should be `np.uint8`.
    """
    if len(images.shape) == 2:
        images = images.reshape(images.shape + (1,))
    images = _check_images_channel(images, channel_last)

    # detect the input array numerical format
    if info is not None:
        info.require_min_max_val()

        is_discrete = info.is_discrete
        in_min_val = info.min_val
        in_bin_size = (
            (info.max_val - info.min_val) /
            float(info.n_discrete_vals - int(info.is_discrete))
        )
        out_bin_size = 2 ** (8 - info.bit_depth)
    else:
        # dequantized values are not supported in auto-infer mode
        is_discrete = True
        in_min_val = 0.
        in_bin_size = 1.
        out_bin_size = 1.

        array_max = np.max(images)
        array_min = np.min(images)
        for c_min, c_max in [(0., 1.), (-1., 1.)]:
            if array_max < c_max + 1e-6 and array_min > c_min - 1e-6:
                in_min_val = c_min
                in_bin_size = (c_max - c_min) / 255.
                break

    # now transform the pixel values according to the source format
    if is_discrete:
        images = (images - in_min_val) / in_bin_size * out_bin_size
    else:
        images = np.asarray((images - in_min_val) / in_bin_size, dtype=np.int32)
        images = images * out_bin_size

    images = np.clip(np.round(images), 0, 255)
    images = images.astype(np.uint8)

    return images


def make_images_grid(images: Sequence[np.ndarray],
                     n_cols: Optional[int] = None,
                     bg_color: Union[np.ndarray, float, int] = 0,
                     border: int = 0,
                     channel_last: Optional[bool] = None,
                     ):
    # check the arguments
    bg_color = np.asarray(bg_color, dtype=np.uint8)
    if len(bg_color.shape) > 1:
        raise ValueError(f'`bg_color` must be at most 1d array.')

    images = list(images)
    if not images:
        raise ValueError(f'`images` must not be empty.')

    for i, im in enumerate(images):
        if len(im.shape) == 2:
            images[i] = im.reshape(im.shape + (1,))

    for im in images:
        if len(im.shape) != 3 or im.dtype != np.uint8:
            raise ValueError(f'Each image must be a 3d uint8 array.')
        if channel_last is None:
            if (im.shape[-3] in (1, 3)) == (im.shape[-1] in (1, 3)):
                raise ValueError(f'Cannot determine `channel_last` automatically.')
            else:
                channel_last = im.shape[-1] in (1, 3)
        else:
            if (channel_last and im.shape[-1] not in (1, 3)) or \
                    (not channel_last and im.shape[-3] not in (1, 3)):
                raise ValueError(f'Invalid shape of image: {im.shape!r}')

    if not channel_last:
        for i, im in enumerate(images):
            images[i] = np.transpose(
                im,
                list(range(0, len(im.shape) - 3)) + [-2, -1, -3]
            )
    n_ch = images[0].shape[-1]
    if len(bg_color.shape) == 0:
        bg_color = np.asarray([bg_color] * n_ch, dtype=np.uint8)
    elif bg_color.shape[-1] != n_ch:
        raise ValueError(f'Channel count of `bg_color` does not agree with '
                         f'the images.')

    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(len(images))))
    n_rows = (len(images) + n_cols - 1) // n_cols

    cell_h, cell_w = 0, 0
    for im in images:
        cell_h = max(cell_h, im.shape[-3])
        cell_w = max(cell_w, im.shape[-2])

    # now generate the images grid
    h = (n_rows - 1) * border + n_rows * cell_h
    w = (n_cols - 1) * border + n_cols * cell_w

    ret = np.zeros([h, w, n_ch], dtype=np.uint8)
    ret[:, :] = bg_color

    for i, im in enumerate(images):
        row = i // n_cols
        col = i - row * n_cols

        im_x = row * (cell_h + border)
        im_y = col * (cell_w + border)
        im_h = im.shape[-3]
        im_w = im.shape[-2]

        ret[im_x: im_x + im_h, im_y: im_y + im_w, :] = im

    return ret


def save_image_to_file(image: np.ndarray,
                       output_file: Optional[str] = None,
                       format: Optional[str] = None,
                       channel_last: Optional[bool] = None) -> str:
    if len(image.shape) == 2:
        image = image.reshape(image.shape + (1,))
    if output_file is None:
        output_file = tempfile.mktemp(suffix='.jpg')

    image = _check_images_channel(
        image, require_one_image=True, channel_last=channel_last)
    imageio.imsave(output_file, image, format=format)
    return output_file
