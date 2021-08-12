import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import *

import imageio
import mltk
import numpy as np
import tensorflow as tf
import tfsnippet as spt
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from ood_regularizer.experiment.datasets.celeba import load_celeba
from .global_config import global_config

__all__ = [
    'TensorLike',
    'get_activation_fn', 'get_kernel_regularizer', 'create_optimizer',
    'get_scope_name', 'GraphNodes',
    'batch_norm_2d', 'get_z_moments', 'unit_ball_regularization_loss',
    'conv2d_with_output_shape', 'deconv2d_with_output_shape',
    'NetworkLogger', 'NetworkLoggers', 'DummyNetworkLogger',
    'get_network_logger', 'Timer',
    'save_images_collection', 'find_largest_batch_size',
]

TensorLike = Union[tf.Tensor, spt.StochasticTensor]


def get_activation_fn():
    activation_functions = {
        'leaky_relu': tf.nn.leaky_relu,
        'relu': tf.nn.relu,
    }
    return activation_functions[global_config.activation_fn]


def get_kernel_regularizer():
    return spt.layers.l2_regularizer(global_config.kernel_l2_reg)


def create_optimizer(name: str,
                     learning_rate: Union[float, TensorLike]
                     ) -> tf.train.Optimizer:
    optimizer_cls = {
        'Adam': tf.train.AdamOptimizer,
    }[name]
    return optimizer_cls(learning_rate)


def get_scope_name(default_name: str,
                   name: Optional[str] = None,
                   obj: Optional[Any] = None):
    return spt.utils.get_default_scope_name(name or default_name, obj)


class GraphNodes(Dict[str, TensorLike]):
    """A dict that maps name to TensorFlow graph nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in self.items():
            if not spt.utils.is_tensor_object(v):
                raise TypeError(f'The value of `{k}` is not a tensor: {v!r}.')

    def eval(self,
             session: tf.Session = None,
             feed_dict: Dict[tf.Tensor, Any] = None) -> Dict[str, Any]:
        """
        Evaluate all the nodes with the specified `session`.

        Args:
            session: The TensorFlow session.
            feed_dict: The feed dict.

        Returns:
            The node evaluation outputs.
        """
        if session is None:
            session = spt.utils.get_default_session_or_error()

        keys = list(self)
        tensors = [self[key] for key in keys]
        outputs = session.run(tensors, feed_dict=feed_dict)

        return dict(zip(keys, outputs))

    def add_prefix(self, prefix: str) -> 'GraphNodes':
        """
        Add a common prefix to all metrics in this collection.

        Args:
             prefix: The common prefix.
        """
        return GraphNodes({f'{prefix}{k}': v for k, v in self.items()})


@add_arg_scope
def batch_norm_2d(input: TensorLike,
                  scope: str,
                  training: Union[bool, TensorLike] = False) -> tf.Tensor:
    """
    Apply batch normalization on 2D convolutional layer.

    Args:
        input: The input tensor.
        scope: TensorFlow variable scope.
        training: Whether or not the model is under training stage?

    Returns:
        The normalized tensor.
    """
    with tf.variable_scope(scope):
        input, s1, s2 = spt.ops.flatten_to_ndims(input, ndims=4)
        output = tf.layers.batch_normalization(
            input,
            axis=-1,
            training=training,
            name='norm'
        )
        output = spt.ops.unflatten_from_ndims(output, s1, s2)
        return output


def get_z_moments(z: TensorLike,
                  value_ndims: int,
                  name: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    value_ndims = spt.utils.validate_enum_arg(
        'value_ndims', value_ndims, [1, 3])

    if value_ndims == 1:
        z = spt.utils.InputSpec(shape=['...', '*']).validate('z', z)
    else:
        z = spt.utils.InputSpec(shape=['...', '?', '?', '*']).validate('z', z)

    with tf.name_scope(name, default_name='get_z_moments', values=[z]):
        rank = len(spt.utils.get_static_shape(z))
        if value_ndims == 1:
            axes = list(range(0, rank - 1))
        else:
            axes = list(range(0, rank - 3))
        mean, variance = tf.nn.moments(z, axes=axes)
        return mean, variance


def unit_ball_regularization_loss(mean: TensorLike,
                                  var: TensorLike,
                                  desired_var: Optional[float] = 1.,
                                  name: Optional[str] = None) -> tf.Tensor:
    mean = tf.convert_to_tensor(mean)
    var = tf.convert_to_tensor(var)

    with tf.name_scope(name,
                       default_name='unit_ball_regularization_loss',
                       values=[mean, var]):
        loss = tf.reduce_mean(tf.square(mean))
        if desired_var is not None:
            loss += tf.reduce_mean(tf.square(var - desired_var))
        return loss


def get_strides_for_conv(input_size, output_size, kernel_size):
    input_size = int(input_size)
    output_size = int(output_size)
    if output_size > input_size:
        raise ValueError('`output_size` must <= `input_size`: output_size {} '
                         'vs input_size {}'.format(output_size, input_size))

    for j in range(1, input_size + 1):
        out_size = (input_size + j - 1) // j
        if out_size == output_size:
            return j
    raise ValueError('No strides can transform input_size {} into '
                     'output_size {} with a convolution operation'.
                     format(input_size, output_size))


def get_strides_for_deconv(input_size, output_size, kernel_size):
    input_size = int(input_size)
    output_size = int(output_size)
    if output_size < input_size:
        raise ValueError('`output_size` must >= `input_size`: output_size {} '
                         'vs input_size {}'.format(output_size, input_size))

    for j in range(1, output_size + 1):
        in_size = (output_size + j - 1) // j
        if in_size == input_size:
            return j
    raise ValueError('No strides can transform input_size {} into '
                     'output_size {} with a deconvolution operation'.
                     format(input_size, output_size))


def conv2d_with_output_shape(conv_fn,
                             input: TensorLike,
                             output_shape: Iterable[int],
                             kernel_size: Union[int, Tuple[int, int]],
                             network_logger: Optional['NetworkLogger'] = None,
                             **kwargs) -> tf.Tensor:
    """
    2D convolutional layer, with `strides` determined by the input shape and
    output shape.

    Args:
        conv_fn: The convolution function.
        input: The input tensor, at least 4-d, `(...,?,H,W,C)`.
        output_shape: The output shape, `(H,W,C)`.
        kernel_size: Kernel size over spatial dimensions.
        network_logger: If specified, log the transformation.
        \\**kwargs: Other named arguments passed to `conv_fn`.

    Returns:
        The output tensor.
    """
    input = (spt.utils.InputSpec(shape=['...', '?', '*', '*', '*']).
             validate('input', input))
    input_shape = spt.utils.get_static_shape(input)[-3:]
    output_shape = tuple(output_shape)
    assert (len(output_shape) == 3 and None not in output_shape)

    # compute the strides
    strides = [
        get_strides_for_conv(input_shape[i], output_shape[i], kernel_size)
        for i in [0, 1]
    ]

    # wrap network_logger if necessary
    if network_logger is not None:
        fn = partial(network_logger.log_apply, conv_fn, input)
    else:
        fn = partial(conv_fn, input)

    # derive output by convolution
    ret = fn(
        out_channels=output_shape[-1],
        kernel_size=kernel_size,
        strides=strides,
        channels_last=True,
        **kwargs
    )

    # validate the output shape
    spt.utils.InputSpec(shape=('...', '?') + output_shape).validate('ret', ret)

    return ret


def deconv2d_with_output_shape(deconv_fn,
                               input: TensorLike,
                               output_shape: Iterable[int],
                               kernel_size: Union[int, Tuple[int, int]],
                               network_logger: Optional['NetworkLogger'] = None,
                               **kwargs) -> tf.Tensor:
    """
    2D deconvolutional layer, with `strides` determined by the input shape and
    output shape.

    Args:
        deconv_fn: The deconvolution function.
        input: The input tensor, at least 4-d, `(...,?,H,W,C)`.
        output_shape: The output shape, `(H,W,C)`.
        kernel_size: Kernel size over spatial dimensions.
        network_logger: If specified, log the transformation.
        \\**kwargs: Other named arguments passed to `conv_fn`.

    Returns:
        The output tensor.
    """
    input = (spt.utils.InputSpec(shape=['...', '?', '*', '*', '*']).
             validate('input', input))
    input_shape = spt.utils.get_static_shape(input)[-3:]
    output_shape = tuple(output_shape)
    assert (len(output_shape) == 3 and None not in output_shape)

    # compute the strides
    strides = [
        get_strides_for_deconv(input_shape[i], output_shape[i], kernel_size)
        for i in [0, 1]
    ]

    # wrap network_logger if necessary
    if network_logger is not None:
        fn = partial(network_logger.log_apply, deconv_fn, input)
    else:
        fn = partial(deconv_fn, input)

    # derive output by convolution
    ret = fn(
        out_channels=output_shape[-1],
        output_shape=output_shape[:-1],
        kernel_size=kernel_size,
        strides=strides,
        channels_last=True,
        **kwargs
    )

    # validate the output shape
    spt.utils.InputSpec(shape=('...', '?') + output_shape).validate('ret', ret)

    return ret


class NetworkLogger(object):
    """
    Class to log the high-level human readable network structures.

    Although TensorFlow computation graph has enough information to figure out
    what exactly the network architecture is, it is too verbose, and is not a
    very easy tool to diagnose the network structure.  Thus we have this class
    to log some high-level human readable network structures.
    """

    def __init__(self):
        """Construct a new :class:`NetworkLogger`."""
        self._logs = []

    @property
    def logs(self) -> List[str]:
        return self._logs

    def log(self, input: TensorLike, transform: str, output: TensorLike):
        """
        Log a transformation.

        Args:
            input: The input tensor.
            transform: The transformation description.
            output: The output tensor.
        """
        self._logs.append((input, str(transform), output))

    def log_apply(self,
                  fn: Callable[..., tf.Tensor],
                  input: TensorLike,
                  log_arg_names_: Iterable[str] = (
                          'units',  # for dense
                          'strides', 'kernel_size',  # for conv & deconv
                          'shortcut_kernel_size',  # for resnet
                          'vertical_kernel_size',
                          'horizontal_kernel_size',  # for pixelcnn
                          'name', 'scope',  # for general layers
                          'ndims', 'shape',  # for reshape_tail
                          'rate',  # for dropout
                  ),
                  *args, **kwargs) -> tf.Tensor:
        """
        Do a transformation and log it.

        Args:
            fn: The transformation function, with accepts `input` as its
                first argument.
            input: The input tensor.
            log_arg_names_: Names of the arguments in
                `kwargs` to be logged.
            *args: The arguments.
            \\**kwargs: The named arguments.
        """
        fn_name = fn.__name__ or repr(fn)
        if log_arg_names_:
            fn_args = []
            log_arg_names_ = tuple(log_arg_names_)
            for key in log_arg_names_:
                if key in kwargs:
                    fn_args.append('{}={!r}'.format(key, kwargs[key]))
            if fn_args:
                fn_name = fn_name + '({})'.format(','.join(fn_args))

        ret = fn(input, *args, **kwargs)
        self.log(input, fn_name, ret)
        return ret


class DummyNetworkLogger(NetworkLogger):
    """
    A dummy network logger that logs nothing.
    """

    def log(self, input: TensorLike, transform: str, output: TensorLike):
        pass

    def log_apply(self, fn, input, log_arg_names_=(), *args, **kwargs):
        return fn(input, *args, **kwargs)


class NetworkLoggers(object):
    """Class to maintain a collection of :class:`NetworkLogger` instances."""

    def __init__(self):
        self._loggers = OrderedDict()

    def get_logger(self, name: str) -> NetworkLogger:
        """
        Get a :class:`NetworkLogger` with specified `name`.

        Args:
            name: Name of the network.

        Returns:
            The network logger instance.
        """
        if name not in self._loggers:
            self._loggers[name] = NetworkLogger()
        return self._loggers[name]

    def format_logs(self, title: str = 'Network Structure') -> str:
        """
        Format the network structure log.

        Args:
            title: Title of the log.

        Returns:
            The formatted log.
        """

        def format_shape(t):
            if isinstance(t, spt.layers.PixelCNN2DOutput):
                shape = spt.utils.get_static_shape(t.horizontal)
            else:
                shape = spt.utils.get_static_shape(t)
            shape = ['?' if s is None else str(s) for s in shape]
            return '(' + ','.join(shape) + ')'

        table = spt.utils.ConsoleTable(3, col_align=['<', '^', '>'])
        table.add_title(title)
        table.add_hr('=')

        for key in self._loggers:
            logger = self._loggers[key]
            table.add_skip()
            table.add_title('{} Structure'.format(key))
            table.add_row(['x', 'y=f(x)', 'y'])
            table.add_hr('-')
            if len(logger.logs) > 0:
                for x, transform, y in logger.logs:
                    table.add_row([format_shape(x), transform, format_shape(y)])
            else:
                table.add_row(['', '(null)', ''])

        return str(table)

    def print_logs(self, title: str = 'Network Structure'):
        """
        Print the network structure log.

        Args:
            title: Title of the log.
        """
        print(self.format_logs(title))

    @contextmanager
    def as_default(self) -> Generator['NetworkLoggers', None, None]:
        """Push this object to the top of context stack."""
        try:
            _network_loggers_stack.push(self)
            yield self
        finally:
            _network_loggers_stack.pop()


_network_loggers_stack = spt.utils.ContextStack()


def get_network_logger(name: str) -> NetworkLogger:
    """
    Get a :class:`NetworkLogger` from the default loggers collection.

    Args:
        name: Name of the network.

    Returns:
        The network logger instance.
    """
    if _network_loggers_stack.top() is None:
        return DummyNetworkLogger()
    else:
        return _network_loggers_stack.top().get_logger(name)


class Timer(object):
    """A very simple timer."""

    def __init__(self):
        self._time_ticks = []
        self.restart()

    def timeit(self, tag):
        self._time_ticks.append((tag, time.time()))

    def print_logs(self):
        if len(self._time_ticks) > 1:
            print(
                mltk.format_key_values(
                    [
                        (tag, spt.utils.humanize_duration(stop - start))
                        for (_, start), (tag, stop) in zip(
                        self._time_ticks[:-1], self._time_ticks[1:]
                    )
                    ],
                    title='Time Consumption',
                )
            )
            print('')

    def restart(self):
        self._time_ticks = [(None, time.time())]


def save_images_collection(images: np.ndarray,
                           filename: str,
                           grid_size: Tuple[int, int],
                           border_size: int = 0):
    """
    Save a collection of images as a large image, arranged in grid.

    Args:
        images: The images collection.  Each element should be a Numpy array,
            in the shape of ``(H, W)``, ``(H, W, C)`` (if `channels_last` is
            :obj:`True`) or ``(C, H, W)``.
        filename: The target filename.
        grid_size: The ``(rows, columns)`` of the grid.
        border_size: Size of the border, for separating images.
            (default 0, no border)
    """

    # check the arguments
    def validate_image(img):
        if len(img.shape) == 2:
            img = np.reshape(img, img.shape + (1,))
        elif len(images[0].shape) == 3:
            if img.shape[2] not in (1, 3, 4):
                raise ValueError('Unexpected image shape: {!r}'.
                                 format(img.shape))
        else:
            raise ValueError('Unexpected image shape: {!r}'.format(img.shape))
        return img

    images = [validate_image(img) for img in images]
    h, w = images[0].shape[:2]
    rows, cols = grid_size[0], grid_size[1]
    buf_h = rows * h + (rows - 1) * border_size
    buf_w = cols * w + (cols - 1) * border_size

    # copy the images to canvas
    n_channels = images[0].shape[2]
    buf = np.zeros((buf_h, buf_w, n_channels), dtype=images[0].dtype)
    for j in range(rows):
        for i in range(cols):
            total_idx = j * cols + i
            if total_idx < len(images):
                img = images[total_idx]
                buf[j * (h + border_size): (j + 1) * h + j * border_size,
                i * (w + border_size): (i + 1) * w + i * border_size,
                :] = img[:, :, :]

    # save the image
    if n_channels == 1:
        buf = np.reshape(buf, (buf_h, buf_w))

    parent_dir = os.path.split(os.path.abspath(filename))[0]
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    imageio.imwrite(filename, buf)


def find_largest_batch_size(test_metrics: GraphNodes,
                            input_x: tf.Tensor,
                            test_x: np.ndarray,
                            feed_dict: Optional[Dict[tf.Tensor, Any]] = None,
                            max_batch_size: int = 256
                            ) -> int:
    """
    Find the largest possible mini-batch size for given output `test_metrics`
    and the input `test_x`.

    Args:
        test_metrics: The dict of outputs, which should be evaluated.
        input_x: The input x placeholder.
        test_x: The test input data.
        feed_dict: The extra data to be fed into `session.run`.
        max_batch_size: The maximum allowed batch size (inclusive).

    Returns:
        The detected largest batch size.
    """

    def next_size():
        return low + int((high - low + 1) // 2)

    if len(test_x) < max_batch_size:
        raise ValueError(f'`len(test_x) < max_batch_size` is not allowed: '
                         f'len(test_x) = {len(test_x)}, max_batch_size = '
                         f'{max_batch_size}')
    assert (max_batch_size > 0)

    low, high = 0, max_batch_size
    batch_feed_dict = dict(feed_dict or {})

    print('Start detecting maximum batch size ...')
    print('>' * 79)
    while True:
        batch_size = next_size()
        if batch_size == low:
            break

        try:
            batch_x = test_x[:batch_size]
            batch_feed_dict[input_x] = batch_x
            _ = test_metrics.eval(feed_dict=batch_feed_dict)
        except ResourceExhaustedError:
            high = batch_size - 1
        else:
            low = batch_size

    print('<' * 79)
    print(f'Maximum batch size has been detected: {low}')
    return low


def get_noise_array(config, array, normalized=True):
    if config.self_ood:
        if config.noise_type == "mutation":
            random_array = np.random.randint(0, 256, size=array.shape, dtype=np.uint8)
            if normalized:
                random_array = (random_array - 127.5) / 256 * 2.0
            mixed_array = np.where(np.random.random(size=array.shape) < config.mutation_rate,
                                   random_array, array)
            mixed_array = mixed_array.astype(array.dtype)
        elif config.noise_type == "gaussian":
            if normalized:
                cifar_train = array * 256.0 / 2 + 127.5
            random_array = np.reshape(np.random.randn(len(cifar_train) * config.x_shape_multiple),
                                      (-1,) + config.x_shape) * config.mutation_rate * 255
            mixed_array = np.clip(np.round(random_array + cifar_train), 0, 255)
            if normalized:
                mixed_array = (mixed_array - 127.5) / 256.0 * 2
            mixed_array = mixed_array.astype(array.dtype)
        elif config.noise_type == "unit":
            if normalized:
                cifar_train = array * 256.0 / 2 + 127.5
            random_array = np.reshape((np.random.rand(len(cifar_train) * config.x_shape_multiple) * 2 - 1),
                                      (-1,) + config.x_shape) * config.mutation_rate * 255
            mixed_array = np.clip(np.round(random_array + cifar_train), 0, 255)
            if normalized:
                mixed_array = (mixed_array - 127.5) / 256.0 * 2
            mixed_array = mixed_array.astype(array.dtype)
        else:
            raise RuntimeError("noise type in {} is not supported".format(config))
    else:
        mixed_array = array
    return mixed_array


def get_mixed_array(config, cifar_train, cifar_test, svhn_train, svhn_test, normalized=True):
    mixed_array = None
    if config.self_ood:
        mixed_array = cifar_train
    else:
        if config.use_transductive:
            mixed_array = np.concatenate([cifar_test[:int(len(cifar_test) * config.in_dataset_test_ratio)], svhn_test])
            if config.mixed_train:
                mixed_array = np.concatenate([cifar_train, mixed_array])
        else:
            mixed_array = svhn_train

    shuffle_index = np.arange(0, len(mixed_array))
    np.random.shuffle(shuffle_index)
    if hasattr(config, 'mixed_ratio'):
        limit_nummber = int(len(shuffle_index) * config.mixed_ratio)
        print(limit_nummber)
        shuffle_index = shuffle_index[:limit_nummber]
        # shuffle_index = np.repeat(shuffle_index, int(1.0 / config.mixed_ratio))
        mixed_array = mixed_array[shuffle_index]
    return mixed_array
