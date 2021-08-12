import mltk

__all__ = ['global_config', 'GlobalConfig']


class GlobalConfig(mltk.Config):
    """
    Global hyper-parameters that affects all components in all models, but
    is not studied in our paper.  We just collect them here.
    """

    activation_fn: str = mltk.ConfigField(
        default='leaky_relu',
        choices=['leaky_relu', 'relu']
    )

    kernel_l2_reg: float = mltk.ConfigField(
        default=0.0001,
        description='L2 regularization coefficient for all layer kernels, '
                    'i.e., `W` of dense layers, and kernel of convolutional '
                    'layers.'
    )


# The global config instance, shared by all other python files.
global_config: GlobalConfig = GlobalConfig()