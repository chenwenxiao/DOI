"""
Usage::

    python flow_next.glow train [train options ...]
    python flow_next.glow test [test options ...]
"""

import sys

import click
import mltk
from tensorkit import tensor as T

from flow_next.common import *
from flow_next.models.glow import *
from utils.data import MMapDataSet


class ExperimentConfig(mltk.Config):
    train = TrainConfig(
        init_batch_size=256,
        batch_size=64,
        test_batch_size=64,
        test_epoch_freq=10,
        max_epoch=500,
        grad_global_clip_norm=100.0,
        # z_mean_var_reg=1.0,
        debug=True,
    )
    model = GlowConfig(
        hidden_conv_activation='leaky_relu',
        activation_flow='leaky_relu',
        depth=10,
        levels=3,
    )
    dataset = DataSetConfig()


def train():
    with mltk.Experiment(ExperimentConfig, args=sys.argv[2:]) as exp, \
            T.use_device(T.first_gpu_device()):
        # load the dataset
        train_dataset, test_dataset = make_dataset(exp.config.dataset)
        print('DataSet loaded.')

        # construct the model
        model = Glow(train_dataset.slots['x'], exp.config.model)
        print('Model constructed.')

        # train the model
        train_model(exp, model, train_dataset, test_dataset)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) >= 2 and sys.argv[1] == 'test':
        test()
    else:
        click.echo('python flow_next.glow train|test options...')
        sys.exit(1)
