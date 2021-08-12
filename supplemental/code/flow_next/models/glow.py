from enum import Enum
from typing import *

import mltk
import tensorkit as tk
from dataclasses import dataclass
from tensorkit import tensor as T
from tensorkit.tensor import Tensor, split, shape

from flow_next.components import *
from utils.data import *

__all__ = [
    'GlowPermutationType', 'ActivationFlowType',
    'GlowConfig',
    'GlowCouplingShiftAndPreScale', 'GlowBlock', 'Glow',
]


class GlowPermutationType(str, Enum):
    REVERSE = 'reverse'
    SHUFFLE = 'shuffle'
    INVERTIBLE_CONV = 'invertible_conv'


class ActivationFlowType(str, Enum):
    NONE = 'none'
    LEAKY_RELU = 'leaky_relu'


class GlowConfig(mltk.Config):

    depth: int
    """Number of glow blocks in each level."""

    levels: int
    """Number of glow levels."""

    prior_use_learnt_mean_std: bool = True
    """Whether or not to use learnt mean and std for the priors of each level?"""

    prior_conv_kernel_size: int = 3
    """Kernel size of the convolution layers for learning the priors."""

    bottom_space_to_depth: bool = True
    """Whether or not to use a SpaceToDepth flow at the input?"""

    use_actnorm_flow: bool = True
    """Whether or not to use ActNorm flow in each GlowBlock?"""

    activation_flow: ActivationFlowType = ActivationFlowType.NONE
    """
    Type of the activation flows after each coupling layer.
    Defaults to `ActivationFlowType.NONE`, do not use activation flows.  
    """

    permutation_type: GlowPermutationType = GlowPermutationType.INVERTIBLE_CONV
    """The flow permutation type."""

    strict_invertible: bool = False
    """Whether or not to use strict invertible kernel for the conv 1x1 flow?"""

    conv_edge_bias: bool = True
    """
    Whether or not to add a `1`s channel, to make the padded edges biased in 
    the convolution layers of the coupling layers?  This applies to both the
    hidden convolution layers and the scale-bias convolution layers of the
    coupling layers.
    """

    scale_bias_conv_kernel_size: int = 3
    """Kernel size of the `scale` and `bias` outputs of the coupling layers."""

    hidden_conv_strides: List[int] = [1, 1]
    """Strides of the hidden convolution layers."""

    hidden_conv_channels: List[int] = [512, 512]
    """Number of channels of the hidden convolution layers."""

    hidden_conv_kernel_sizes: List[int] = [3, 1]
    """Kernel sizes of the hidden convolution layers."""

    hidden_conv_use_resnet: bool = False
    """Whether or not to use resnet in coupling layers?"""

    hidden_conv_act_norm: bool = True
    """Whether or not to use layer act-norm in coupling layers?"""

    hidden_conv_weight_norm: bool = False
    """Whether or not to use layer weight-norm in coupling layers?"""

    hidden_conv_activation: str = 'relu'
    """The activation of the convolution layers."""


def conv2d_zeros(in_channels: int, out_channels: int, kernel_size: int,
                 padding: str = 'half'):
    return tk.layers.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        weight_init=tk.init.zeros
    )


class GlowCouplingShiftAndPreScale(tk.layers.BaseLayer):

    __constants__ = ('channel_axis',)

    shared: T.Module
    channel_axis: int

    def __init__(self, shared: T.Module):
        super().__init__()
        self.shared = shared
        self.channel_axis = -1 if T.IS_CHANNEL_LAST else -3

    def forward(self, input: Tensor) -> List[Tensor]:
        input = self.shared(input)
        k = shape(input)[self.channel_axis] // 2
        return split(input, [k, k], axis=self.channel_axis)


class GlowBlock(tk.flows.SequentialFlow):

    def __init__(self, n_channels: int, config: GlowConfig):
        super().__init__(self.build_flows(n_channels, config))

    def build_flows(self,
                    n_channels: int,
                    config: GlowConfig
                    ) -> List[tk.flows.Flow]:
        flows = []

        # ActNorm
        if config.use_actnorm_flow:
            flows.append(
                tk.flows.ActNorm2d(
                    num_features=n_channels,
                    scale='exp',
                )
            )

        # Feature Re-ordering
        if config.permutation_type == GlowPermutationType.REVERSE:
            flows.append(tk.flows.FeatureReversingFlow2d())
        elif config.permutation_type == GlowPermutationType.SHUFFLE:
            flows.append(tk.flows.FeatureShufflingFlow2d(n_channels))
        elif config.permutation_type == GlowPermutationType.INVERTIBLE_CONV:
            flows.append(tk.flows.InvertibleConv2d(
                n_channels,
                strict=config.strict_invertible,
            ))

        # Coupling layer
        flows.append(tk.flows.CouplingLayer2d(
            shift_and_pre_scale=GlowCouplingShiftAndPreScale(
                self.build_coupling_shared_net(n_channels, config)),
            scale='sigmoid' if config.use_actnorm_flow else 'exp',
        ))

        # Activation flow
        if config.activation_flow == ActivationFlowType.LEAKY_RELU:
            shift = 3.
            flows.append(ShiftFlow(shift=shift, event_ndims=3))
            flows.append(LeakyReLUFlow(event_ndims=3))
            flows.append(ShiftFlow(shift=-shift, event_ndims=3))
        elif config.activation_flow == ActivationFlowType.NONE:
            pass  # Do nothing
        else:
            raise ValueError(f'Unsupported activation flow: {config.activation_flow}')

        return flows

    def build_coupling_shared_net(self,
                                  n_channels: int,
                                  config: GlowConfig
                                  ) -> T.Module:
        n1 = n_channels // 2
        n2 = n_channels - n1
        normalizer_cls = (
            tk.layers.ActNorm2d if config.hidden_conv_act_norm
            else tk.layers.BatchNorm2d
        )

        in_channels = n1
        layers = []
        for ksize, stride, out_channels in zip(config.hidden_conv_kernel_sizes,
                                               config.hidden_conv_strides,
                                               config.hidden_conv_channels):
            # edge bias
            if config.conv_edge_bias:
                layers.append(tk.layers.AddOnesChannel2d())
                in_channels += 1

            # the conv layer
            conv_cls = (
                tk.layers.ResBlock2d if config.hidden_conv_use_resnet
                else tk.layers.Conv2d
            )
            layers.append(conv_cls(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksize,
                padding='half',
                activation=tk.layers.get_activation_class(config.hidden_conv_activation),
                normalizer=normalizer_cls,
                weight_norm=config.hidden_conv_weight_norm,
            ))

            # prepare for the next layer
            in_channels = out_channels

        # the final output layer
        if config.conv_edge_bias:
            layers.append(tk.layers.AddOnesChannel2d())
            in_channels += 1

        layers.append(conv2d_zeros(
            in_channels=in_channels,
            out_channels=n2 * 2,  # for shift & scale
            kernel_size=config.scale_bias_conv_kernel_size,
        ))

        return tk.layers.Sequential(layers) if len(layers) > 1 else layers[0]


@dataclass
class GlowLevelOutput(object):
    log_det: T.Tensor
    """Log-determinant of this level given the input."""

    left: Optional[T.Tensor]
    """
    Output of the left branch, which should be sent into the next level.
    For `is_top == True`, this will be None.
    """

    right: T.Tensor
    """Output of the right branch, or the whole output for top level."""

    right_log_prob: T.Tensor
    """Log-prob of the right output w.r.t. the level prior."""


class GlowLevel(tk.layers.BaseLayer):

    is_top: bool
    """Whether or not this level is the top level?"""

    in_channels: int
    """Number of input channels."""

    out_channels: int
    """Number of output channels of this level (left + right)."""

    channel_axis: int
    """Axis of the channel."""

    space_to_depth: bool
    """Whether or not to apply `space_to_depth` on the input?"""

    device: str
    """Device, where to put tensors."""

    flow: tk.flows.Flow
    """The composed flow of this level."""

    inv_flow: tk.flows.Flow
    """The inverse flow of `flow`."""

    prior_mean_logstd_net: Optional[T.Module]
    """
    If specified, will use this module to produce the `mean` and `logstd`
    of the Gaussian prior, given the context.  Otherwise will use unit
    Gaussian prior. 
    """

    def __init__(self,
                 config: GlowConfig,
                 is_top: bool,
                 in_channels: int,
                 space_to_depth: bool = True,
                 prior_use_learnt_mean_std: bool = True,
                 device: Optional[str] = None):
        super().__init__()

        self.is_top = bool(is_top)
        self.in_channels = in_channels = int(in_channels)
        self.channel_axis = -1 if T.IS_CHANNEL_LAST else -3
        self.space_to_depth = space_to_depth = bool(space_to_depth)
        self.device = device or T.current_device()

        # construct the flow module
        self.out_channels = in_channels
        if space_to_depth:
            self.out_channels *= 4
        self.flow = tk.flows.SequentialFlow([
            GlowBlock(self.out_channels, config)
            for _ in range(config.depth)
        ])
        self.inv_flow = self.flow.invert()

        # construct the prior module
        if prior_use_learnt_mean_std:
            prior_channels = self.out_channels
            if not self.is_top:
                prior_channels //= 2
            self.prior_mean_logstd_net = conv2d_zeros(
                in_channels=prior_channels,
                out_channels=2 * prior_channels,
                kernel_size=config.prior_use_learnt_mean_std,
                padding='half'
            )
        else:
            self.prior_mean_logstd_net = None

    @property
    def use_learnt_prior(self) -> bool:
        return self.prior_mean_logstd_net is not None

    def jit_compile(self):
        self.prior_mean_logstd_net = tk.layers.jit_compile(self.prior_mean_logstd_net)
        self.flow = tk.layers.jit_compile(self.flow)
        self.inv_flow = tk.flows.InverseFlow(self.flow)

    def get_prior_mean_logstd(self,
                              shape: List[int],  # 3 elements, the height, width and channels
                              context: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if self.use_learnt_prior:
            if context is None:
                if self.is_top:
                    context = T.zeros([1] + shape, device=self.device)
                else:
                    raise ValueError(
                        f'`context` must be specified when `prior_mean_logstd_net` '
                        f'is not None, and `is_top` is not True.')

            if T.rank(context) == 3:
                context = self.prior_mean_logstd_net(T.expand_dim(context, axis=0))
                context = T.squeeze_axis(context, axis=0)
            else:
                context = self.prior_mean_logstd_net(context)

            n = T.shape(context)[self.channel_axis]
            mean, logstd = T.split(context, [n // 2, n // 2], axis=self.channel_axis)
        else:
            mean = logstd = T.zeros(shape, device=self.device)
        return mean, logstd

    def forward(self, input: T.Tensor) -> GlowLevelOutput:
        if self.space_to_depth:
            input = T.nn.space_to_depth2d(input, block_size=2)

        input, log_det = self.flow(input, compute_log_det=True)

        if self.is_top:
            left = None
            right = input
        else:
            n = T.shape(input)[self.channel_axis]
            k = n // 2
            left, right = T.split(input, [k, k], axis=self.channel_axis)

        input = None  # de-reference early to allow gc take place

        mean, logstd = self.get_prior_mean_logstd(
            shape=T.shape(right)[-3:],
            context=left,
        )
        right_log_prob = T.random.normal_log_pdf(right, mean, logstd, group_ndims=3)

        return GlowLevelOutput(
            log_det=log_det, left=left, right=right, right_log_prob=right_log_prob)

    def sample(self,
               shape: List[int] = None,   # this is the shape of the prior
               context: Optional[T.Tensor] = None,
               n_samples: Optional[int] = None,  # for each temperature, obtain this many samples
               temperatures: Optional[List[float]] = None) -> T.Tensor:
        if shape is None:
            if context is None:
                raise ValueError(f'`shape` must be specified when `context` is None.')
            shape = T.shape(context)
        if context is not None and n_samples is not None and len(T.shape(context)) != 3:
            raise ValueError(f'`context` must be 3d if `n_samples` is not None.')
        if not self.is_top and context is None:
            raise ValueError(f'`context` must be specified when `is_top` is False.')

        # obtain the sample from the right prior (context may be the left)
        mean, logstd = self.get_prior_mean_logstd(shape, context)
        std = T.exp(logstd)
        logstd = None  # de-reference for gc

        if temperatures is not None:
            temperatures = T.as_tensor(temperatures, device=self.device)
            temperatures = T.reshape(temperatures, [-1, 1, 1, 1, 1])
            if n_samples is not None:
                std = T.repeat(std * temperatures, [1, n_samples, 1, 1, 1])
            else:
                std = std * temperatures
            std = T.reshape(std, [-1] + T.shape(std)[-3:])
        else:
            std = T.repeat(std, [n_samples or 1, 1, 1, 1])

        samples = T.random.normal(mean, std)

        # compose the sample at the output
        if not self.is_top:
            samples = T.concat([context, samples], axis=self.channel_axis)

        # feed back through the flow
        samples, _ = self.inv_flow(samples, compute_log_det=False)

        # now reshape if required
        if self.space_to_depth:
            samples = T.nn.depth_to_space2d(samples, block_size=2)

        return samples


class Glow(tk.layers.BaseLayer):

    data_info: ArrayInfo
    config: GlowConfig
    levels: tk.layers.ModuleList
    top_prior_shape: List[int]

    def __init__(self, data_info: ArrayInfo, config: GlowConfig):
        super().__init__()

        channel_axis = -1 if T.IS_CHANNEL_LAST else -3
        data_info.require_shape(deterministic=True)
        if len(data_info.shape) != 3:
            raise ValueError(f'The data is required to be 3d: '
                             f'got data shape {data_info.shape!r}')
        if data_info.shape[channel_axis] not in (1, 3):
            raise ValueError(f'The channel axis {channel_axis} is required to '
                             f'be 1 or 3: got data shape {data_info.shape!r}')

        self.data_info = data_info
        self.config = config

        # assemble the levels and the flows
        in_channels = self.data_info.shape[channel_axis]
        levels = []
        for i in range(config.levels):
            level = GlowLevel(
                config=config,
                is_top=(i == config.levels - 1),
                in_channels=in_channels,
                space_to_depth= i > 0 or config.bottom_space_to_depth,
                prior_use_learnt_mean_std=config.prior_use_learnt_mean_std,
            )
            levels.append(level)
            in_channels = level.out_channels // 2  # n_channels of the left branch
        self.levels = tk.layers.ModuleList(levels)

        # compute the shape of the top prior
        ratio = 2 ** config.levels - 1 + int(config.bottom_space_to_depth)
        top_prior_shape = [s // ratio for s in self.data_info.shape]
        top_prior_shape[channel_axis] = levels[-1].out_channels
        self.top_prior_shape = top_prior_shape

    def forward(self, input: Tensor) -> Tuple[Tensor, List[GlowLevelOutput]]:
        outputs = []

        for level in self.levels:
            output = level(input)  # type: GlowLevelOutput
            input = output.left
            outputs.append(output)

        log_prob = outputs[0].log_det + outputs[0].right_log_prob
        for output in outputs[1:]:
            log_prob = log_prob + output.log_det + output.right_log_prob

        return log_prob, outputs

    def sample(self,
               n_samples: Optional[int],
               temperatures: Optional[List[float]] = None) -> T.Tensor:
        """
        For each `temperatures`, sample `n_samples` images.
        If `temperatures` is not specified, assume it is `[1.]`.
        """
        samples = self.levels[-1].sample(
            self.top_prior_shape, n_samples=n_samples, temperatures=temperatures)
        for level in self.levels[-2::-1]:
            samples = level.sample(context=samples)
        return samples

    def custom_compile_module(self):
        for level in self.levels:
            level.jit_compile()

    def initialize(self, x):
        x = T.as_tensor(x)
        _ = self(x)  # trigger initialization
        tk.layers.jit_compile_children(
            self,
            # do not compile invert flow, otherwise the modules will be
            # compiled twice (once for `self.flow`).
            filter=lambda m: m is not self.invert_flow
        )
        return self(x)  # trigger JIT
