from typing import *

import mltk
import numpy as np
import tensorflow as tf
import tfsnippet as spt
from tensorflow.contrib.framework import arg_scope
from tfsnippet.layers.flows.utils import ZeroLogDet

from .utils import *

__all__ = ['RealNVPConfig', 'DepthToSpaceFlow', 'make_real_nvp']


class RealNVPConfig(mltk.Config):
    ############################################
    # general configurations for RealNVP flows #
    ############################################
    flow_depth: Optional[int] = mltk.ConfigField(
        default=None, nullable=True,
        description='The flow depth K of RealNVP.',
    )

    use_invertible_flow: bool = mltk.ConfigField(
        default=True,
        description='Whether or not to use the invertible dense / conv2d? '
                    'If set to False, will use feature reversing layer '
                    'instead of the invertible dense / conv2d proposed by '
                    'Glow.'
    )

    strict_invertible: bool = mltk.ConfigField(
        default=False,
        description='Whether or not to use LU decomposition to derive a '
                    'strictly invertible dense layer?'
    )

    use_actnorm_flow: bool = mltk.ConfigField(
        default=True,
        description='Whether or not to use actnorm flow layer?'
    )

    use_leaky_relu_flow: bool = mltk.ConfigField(
        default=False,
        description='Whether or not to use the LeakyReLU as a flow layer?'
    )

    coupling_scale_type: str = mltk.ConfigField(
        default='sigmoid',
        description='Type of the `scale` activation function in affine '
                    'coupling layer.',
        choices=['sigmoid', 'exp']
    )

    coupling_sigmoid_scale_bias: float = mltk.ConfigField(
        default=2.,
        description='The constant bias for `scale` when using sigmoid '
                    'activation function in affine coupling layer.'
    )

    coupling_scale_shift_initializer: str = mltk.ConfigField(
        default='zero',
        description='Initializer of the `scale` and `shift` layer in '
                    'affine coupling layer.',
        choices=['zero', 'normal'],
    )
    coupling_scale_shift_normal_initializer_stddev: float = 0.001

    #############################################
    # configurations for the dense RealNVP flow #
    #############################################
    dense_coupling_n_hidden_layers: int = mltk.ConfigField(
        default=1,
        description='The number of hidden layers in each affine coupling layer.'
    )

    dense_coupling_n_hidden_units: int = mltk.ConfigField(
        default=256,
        description='The number of units in the hidden layers in each affine '
                    'coupling layer.'
    )

    #####################################################
    # configurations for the convolutional RealNVP flow #
    #####################################################
    conv_coupling_n_blocks: int = mltk.ConfigField(
        default=1,
        description='Number of blocks for convolutional coupling layers. '
                    'The depth of each block will be `flow_depth // n_blocks`. '
                    'Blocks except the last block are ended with '
                    'a split flow and a squeeze flow.',
    )

    conv_coupling_squeeze_before_first_block: bool = mltk.ConfigField(
        default=False,
        description='Whether or not to squeeze the spatial dimensions '
                    '(i.e., use SpaceToDepth flow) before the first '
                    'convolutional coupling block?'
    )

    conv_coupling_hidden_channels: List[int] = mltk.ConfigField(
        default=[64, 64],
        description='The channels for resnet blocks in convolutional coupling '
                    'layers.'
    )

    conv_coupling_hidden_kernel_size: Union[int, List[int]] = mltk.ConfigField(
        default=3,
        description='The kernel size for resnet blocks in convolutional '
                    'coupling layers.'
    )


class DepthToSpaceFlow(spt.layers.BaseFlow):
    """
    A flow which computes ``y = depth_to_space(x)``, and conversely
    ``x = space_to_depth(y)``.
    """

    @spt.utils.add_name_and_scope_arg_doc
    def __init__(self, block_size, channels_last=True, name=None, scope=None):
        """
        Construct a new :class:`DepthToSpaceFlow`.

        Args:
            block_size (int): An int >= 2, the size of the spatial block.
            channels_last (bool): Whether or not the channels axis
                is the last axis in the input tensor?
        """
        block_size = int(block_size)
        if block_size < 2:
            raise ValueError('`block_size` must be at least 2.')

        self._block_size = block_size
        self._channels_last = bool(channels_last)
        super(DepthToSpaceFlow, self).__init__(
            x_value_ndims=3, y_value_ndims=3, require_batch_dims=True,
            name=name, scope=scope
        )

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        # TODO: maybe add more shape check here.
        pass

    def _transform(self, x, compute_y, compute_log_det):
        # compute y
        y = None
        if compute_y:
            y = spt.ops.depth_to_space(x, block_size=self._block_size,
                                       channels_last=self._channels_last)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(shape=spt.utils.get_shape(x)[:-3],
                                 dtype=x.dtype.base_dtype)

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        # compute x
        x = None
        if compute_x:
            x = spt.ops.space_to_depth(y, block_size=self._block_size,
                                       channels_last=self._channels_last)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(shape=spt.utils.get_shape(y)[:-3],
                                 dtype=y.dtype.base_dtype)

        return x, log_det


class FeatureReversingFlow(spt.layers.FeatureMappingFlow):
    def __init__(self, axis=-1, value_ndims=1, name=None, scope=None):
        super(FeatureReversingFlow, self).__init__(
            axis=int(axis), value_ndims=value_ndims, name=name, scope=scope)

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        pass

    def _reverse_feature(self, x, compute_y, compute_log_det):
        n_features = spt.utils.get_static_shape(x)[self.axis]
        if n_features is None:
            raise ValueError('The feature dimension must be fixed.')
        assert (0 > self.axis >= -self.value_ndims >=
                -len(spt.utils.get_static_shape(x)))
        permutation = np.asarray(list(reversed(range(n_features))),
                                 dtype=np.int32)

        # compute y
        y = None
        if compute_y:
            y = tf.gather(x, permutation, axis=self.axis)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(spt.utils.get_shape(x)[:-self.value_ndims],
                                 x.dtype.base_dtype)

        return y, log_det

    def _transform(self, x, compute_y, compute_log_det):
        return self._reverse_feature(x, compute_y, compute_log_det)

    def _inverse_transform(self, y, compute_x, compute_log_det):
        return self._reverse_feature(y, compute_x, compute_log_det)


def _conv_real_nvp(config: RealNVPConfig,
                   is_prior_flow: bool,
                   normalizer_fn,
                   scope: str) -> spt.layers.BaseFlow:
    def shift_and_scale(x1, n2):
        logger = get_network_logger(tf.get_variable_scope().name)

        n_layers = len(config.conv_coupling_hidden_channels)
        kernel_sizes = config.conv_coupling_hidden_kernel_size \
            if isinstance(config.conv_coupling_hidden_kernel_size, (list, tuple)) \
            else [config.conv_coupling_hidden_kernel_size] * n_layers
        with arg_scope([spt.layers.resnet_conv2d_block],
                       activation_fn=get_activation_fn(),
                       kernel_regularizer=get_kernel_regularizer(),
                       normalizer_fn=normalizer_fn):
            h = x1
            for j, (out_channels, kernel_size) in enumerate(
                    zip(config.conv_coupling_hidden_channels, kernel_sizes)):
                h = logger.log_apply(
                    spt.layers.resnet_conv2d_block,
                    h,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=1,
                    scope='resnet_{}'.format(j)
                )

        # compute shift and scale
        if config.coupling_scale_shift_initializer == 'zero':
            pre_params_initializer = tf.zeros_initializer()
        else:
            pre_params_initializer = tf.random_normal_initializer(
                stddev=config.coupling_scale_shift_normal_initializer_stddev)
        pre_params = logger.log_apply(
            spt.layers.conv2d,
            h,
            out_channels=n2 * 2,
            kernel_size=(1, 1),
            strides=1,
            kernel_initializer=pre_params_initializer,
            scope='shift_and_scale',
        )
        shift = pre_params[..., :n2]
        scale = pre_params[..., n2:]

        return shift, scale

    def coupling_block(block_id, squeeze):
        if block_id < config.conv_coupling_n_blocks - 1:
            depth = block_depth
        else:
            depth = config.flow_depth - (block_depth *
                                         (config.conv_coupling_n_blocks - 1))

        with tf.variable_scope(f'block_{block_id}'):
            block_flows = []
            if squeeze:
                block_flows.append(spt.layers.SpaceToDepthFlow(2))

            for i in range(depth):
                block_level = []
                if config.use_actnorm_flow:
                    block_level.append(spt.layers.ActNorm(value_ndims=3))
                if config.use_invertible_flow:
                    block_level.append(
                        spt.layers.InvertibleConv2d(
                            strict_invertible=config.strict_invertible)
                    )
                else:
                    block_level.append(FeatureReversingFlow(value_ndims=3))
                block_level.append(
                    spt.layers.CouplingLayer(
                        tf.make_template(
                            'coupling',
                            shift_and_scale,
                            create_scope_now_=True
                        ),
                        value_ndims=3,
                        scale_type=config.coupling_scale_type,
                        sigmoid_scale_bias=config.coupling_sigmoid_scale_bias,
                    )
                )
                if config.use_leaky_relu_flow:
                    block_level.append(
                        spt.layers.LeakyReLU().as_flow(value_ndims=3))
                block_flows.extend(block_level)

            if block_id < config.conv_coupling_n_blocks - 1:
                block_flows.append(spt.layers.SplitFlow(
                    split_axis=-1,
                    left=coupling_block(block_id + 1, squeeze=True)
                ))

            if squeeze:
                block_flows.append(DepthToSpaceFlow(2))

            return spt.layers.SequentialFlow(block_flows)

    block_depth = config.flow_depth // config.conv_coupling_n_blocks
    with tf.variable_scope(scope):
        flow = coupling_block(
            0,
            squeeze=config.conv_coupling_squeeze_before_first_block
        )

    if is_prior_flow:
        flow = flow.invert()

    return flow


def _dense_real_nvp(config: RealNVPConfig,
                    is_prior_flow: bool,
                    normalizer_fn,
                    scope: str) -> spt.layers.BaseFlow:
    def shift_and_scale(x1, n2):
        logger = get_network_logger(tf.get_variable_scope().name)

        with arg_scope([spt.layers.dense],
                       activation_fn=get_activation_fn(),
                       kernel_regularizer=get_kernel_regularizer(),
                       normalizer_fn=normalizer_fn):
            h = x1
            for j in range(config.dense_coupling_n_hidden_layers):
                h = logger.log_apply(
                    spt.layers.dense,
                    h,
                    units=config.dense_coupling_n_hidden_units,
                    scope='hidden_{}'.format(j)
                )

        # compute shift and scale
        if config.coupling_scale_shift_initializer == 'zero':
            pre_params_initializer = tf.zeros_initializer()
        else:
            pre_params_initializer = tf.random_normal_initializer(
                stddev=config.coupling_scale_shift_normal_initializer_stddev)
        pre_params = logger.log_apply(
            spt.layers.dense,
            h,
            units=n2 * 2,
            kernel_initializer=pre_params_initializer,
            scope='shift_and_scale',
        )
        shift = pre_params[..., :n2]
        scale = pre_params[..., n2:]

        return shift, scale

    with tf.variable_scope(scope):
        flows = []
        for i in range(config.flow_depth):
            level = []
            if config.use_actnorm_flow:
                level.append(spt.layers.ActNorm())
            if config.use_invertible_flow:
                level.append(
                    spt.layers.InvertibleDense(
                        strict_invertible=config.strict_invertible)
                )
            else:
                level.append(FeatureReversingFlow())
            level.append(
                spt.layers.CouplingLayer(
                    tf.make_template(
                        'coupling', shift_and_scale, create_scope_now_=True),
                    scale_type=config.coupling_scale_type,
                    sigmoid_scale_bias=config.coupling_sigmoid_scale_bias,
                )
            )
            flows.extend(level)
        flow = spt.layers.SequentialFlow(flows)

    if is_prior_flow:
        flow = flow.invert()

    return flow


def make_real_nvp(rnvp_config: RealNVPConfig,
                  is_conv: bool,
                  is_prior_flow: bool,
                  normalizer_fn,
                  scope: str) -> spt.layers.BaseFlow:
    """
    Construct a RealNVP flow.

    Args:
        rnvp_config: RealNVP hyper-parameters.
        is_conv: Whether or not to use
        is_prior_flow: Whether or not this flow is constructed for `p(z)`?
        scope: TensorFlow variable scope.
    """
    if is_conv:
        return _conv_real_nvp(rnvp_config, is_prior_flow, normalizer_fn, scope)
    else:
        return _dense_real_nvp(rnvp_config, is_prior_flow, normalizer_fn, scope)
