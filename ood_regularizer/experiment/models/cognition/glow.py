# -*- coding: utf-8 -*-
import mltk
from mltk.data import ArraysDataStream, DataStream
from tensorkit import tensor as T
import sys
from argparse import ArgumentParser

from pprint import pformat

from matplotlib import pyplot
import torch

import tfsnippet as spt
from tfsnippet.examples.utils import (MLResults,
                                      print_with_title)
import numpy as np

from flow_next.common import TrainConfig, DataSetConfig, make_dataset, train_model, get_mapper
from flow_next.models.glow import GlowConfig, Glow
from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import plot_fig, make_diagram_torch, get_ele_torch

from utils.data import SplitInfo
from utils.evaluation import dequantized_bpd
import torch.autograd as autograd
from imgaug import augmenters as iaa


class ExperimentConfig(mltk.Config):
    # model parameters
    z_dim = 256
    act_norm = False
    weight_norm = False
    batch_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    nf_layers = 20

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 400
    warm_up_start = 200
    initial_beta = -3.0
    uniform_scale = False
    use_transductive = True
    mixed_train = False
    mixed_train_epoch = 256
    mixed_train_skip = 64
    mixed_times = 64
    mixed_replace = 64
    mixed_replace_ratio = 1.0
    dynamic_epochs = False
    retrain_for_batch = True
    pretrain = True
    stand_weight = 0.1

    compressor = 2  # 0 for jpeg, 1 for png, 2 for flif

    max_step = None
    batch_size = 64
    smallest_step = 5e-5
    initial_lr = 0.0005
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = []
    lr_anneal_step_freq = None
    clip_norm = 5

    n_critical = 5
    # evaluation parameters
    train_n_qz = 1
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 200
    plot_epoch_freq = 20
    distill_ratio = 1.0
    distill_epoch = 5000

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    extra_stride = 2

    train = TrainConfig(
        optimizer='adamax',
        init_batch_size=128,
        batch_size=64,
        test_batch_size=64,
        test_epoch_freq=10,
        max_epoch=50,
        # grad_global_clip_norm=None,
        grad_global_clip_norm=1.0,
        debug=True
    )
    model = GlowConfig(
        hidden_conv_activation='relu',
        hidden_conv_channels=[128, 128],
        depth=3,
        levels=3,
    )
    in_dataset = 'cifar10'
    out_dataset = 'svhn'
    count_experiment = False


def main():
    with mltk.Experiment(ExperimentConfig, args=sys.argv[1:]) as exp, \
            T.use_device(T.first_gpu_device()):
        while True:
            try:
                exp.make_dirs('plotting')
                break
            except Exception:
                pass
        config = exp.config
        # prepare for training and testing data
        config.in_dataset = DataSetConfig(name=config.in_dataset)
        config.out_dataset = DataSetConfig(name=config.out_dataset)
        x_train_complexity, x_test_complexity = load_complexity(config.in_dataset.name, config.compressor)
        svhn_train_complexity, svhn_test_complexity = load_complexity(config.out_dataset.name, config.compressor)

        if config.count_experiment:
            with open('/home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/count_experiments',
                      'a') as f:
                f.write(exp.abspath("") + '\n')
                f.close()

        experiment_dict = {
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/b0/e5/02c52d867e43f4e461f5',
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/f9/d5/02812baa4f70f4e461f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/6c/d5/02732c28dc8df4e461f5',
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/02/e5/02279d802d3af4e461f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/e9/d5/02812baa4f70f4e461f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/db/d5/02812baa4f70f19e02f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/25/e5/02c52d867e43435322f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/80/e5/02c52d867e43f4e461f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/a0/e5/02c52d867e43f4e461f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/90/e5/02c52d867e43f4e461f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/12/e5/02279d802d3af4e461f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/7c/d5/02732c28dc8df4e461f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/d5/e5/02732c28dc8d622303f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/c5/e5/02732c28dc8d622303f5'
        }
        print(experiment_dict)
        if config.in_dataset.name in experiment_dict:
            restore_checkpoint = experiment_dict[config.in_dataset.name]
        else:
            restore_checkpoint = None
        print('restore model from {}'.format(restore_checkpoint))

        # load the dataset
        cifar_train_dataset, cifar_test_dataset, cifar_dataset = make_dataset(config.in_dataset)
        print('CIFAR DataSet loaded.')
        svhn_train_dataset, svhn_test_dataset, svhn_dataset = make_dataset(config.out_dataset)
        print('SVHN DataSet loaded.')

        cifar_train_flow = cifar_test_dataset.get_stream('train', 'x', config.batch_size)
        cifar_test_flow = cifar_test_dataset.get_stream('test', 'x', config.batch_size)
        svhn_train_flow = svhn_test_dataset.get_stream('train', 'x', config.batch_size)
        svhn_test_flow = svhn_test_dataset.get_stream('test', 'x', config.batch_size)

        if restore_checkpoint is not None:
            model = torch.load(restore_checkpoint + '/model.pkl')
        else:
            # construct the model
            model = Glow(cifar_train_dataset.slots['x'], exp.config.model)
            print('Model constructed.')

            # train the model
            train_model(exp, model, cifar_train_dataset, cifar_test_dataset)

        torch.save(model, 'model.pkl')

        with mltk.TestLoop() as loop:
            @torch.no_grad()
            def eval_ll(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                bpd = -dequantized_bpd(ll, cifar_train_dataset.slots['x'])
                return T.to_numpy(bpd)

            x_test = cifar_dataset.get_array('test', 'x')
            svhn_test = svhn_dataset.get_array('test', 'x')
            if x_test.shape[-1] == 3:
                config.stand_weight = 0.2
            print(x_test.shape)
            mixed_array = np.concatenate([
                x_test, svhn_test
            ])
            index = np.arange(0, len(mixed_array))
            np.random.shuffle(index)
            index = index[:len(index) // config.mixed_times]
            config.mixed_train_skip = config.mixed_train_skip // config.mixed_times
            config.mixed_train_epoch = config.mixed_train_epoch * config.mixed_times
            index = index[:100]
            mixed_array = mixed_array[index]
            mixed_kl = []

            test_mapper = get_mapper(config.in_dataset, training=False)
            train_mapper = get_mapper(config.in_dataset, training=True)
            test_mapper.fit(cifar_dataset.slots['x'])
            train_mapper.fit(cifar_dataset.slots['x'])
            mixed_stream = ArraysDataStream(
                [mixed_array], batch_size=config.batch_size, shuffle=False,
                skip_incomplete=False).map(
                lambda x: test_mapper.transform(x))

            mixed_ll = get_ele_torch(eval_ll, mixed_stream)

            def stand(base, another_arrays=None):
                mean, std = np.mean(base), np.std(base)
                return_arrays = []
                for array in another_arrays:
                    return_arrays.append(-np.abs((array - mean) / std) * config.stand_weight)
                return return_arrays

            cifar_train_nll = get_ele_torch(eval_ll, cifar_train_flow)
            [mixed_stand] = stand(cifar_train_nll, [mixed_ll])

            if not config.pretrain:
                model = Glow(cifar_train_dataset.slots['x'], exp.config.model)
            torch.save(model, 'last.pkl')

            for i in range(0, len(mixed_array), config.mixed_train_skip):
                def data_generator():
                    mixed_index = np.random.randint(i if config.retrain_for_batch else 0,
                                                    min(len(mixed_array), i + config.mixed_train_skip),
                                                    config.batch_size)

                    batch_x = mixed_array[mixed_index]
                    aug = iaa.Affine(
                        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                        # order=3,  # turn on this if not just translation
                        rotate=(-30, 30),
                        mode='edge',
                        backend='cv2'
                    )
                    batch_x = aug(images=batch_x)
                    batch_x = train_mapper.transform(batch_x)
                    ll = mixed_ll[mixed_index]
                    # print(batch_x.shape)

                    if config.distill_ratio != 1.0:
                        ll_omega = eval_ll(batch_x)
                        batch_index = np.argsort(ll - ll_omega)
                        batch_index = batch_index[:int(len(batch_index) * config.distill_ratio)]
                        batch_x = batch_x[batch_index]
                    yield [T.from_numpy(batch_x)]

                if config.dynamic_epochs:
                    repeat_epoch = int(
                        config.mixed_train_epoch * len(mixed_array) / (9 * i + len(mixed_array)))
                    repeat_epoch = max(1, repeat_epoch)
                else:
                    repeat_epoch = config.mixed_train_epoch
                repeat_epoch = repeat_epoch * config.mixed_train_skip // config.batch_size
                # data generator generate data for each batch
                # repeat_epoch will determine how much time it generates
                exp.config.train.lr = 0.001 / 16
                exp.config.train.warmup_epochs = None
                exp.config.train.max_epoch = repeat_epoch
                exp.config.train.test_epoch_freq = exp.config.train.max_epoch + 1
                if config.retrain_for_batch:
                    model = torch.load('last.pkl')
                try:
                    train_model(exp, model, svhn_train_dataset, None,
                                DataStream.generator(data_generator))
                except Exception as e:
                    print(e)

                mixed_kl.append(get_ele_torch(eval_ll, ArraysDataStream(
                    [mixed_array[i: i + config.mixed_train_skip]], batch_size=config.batch_size, shuffle=False,
                    skip_incomplete=False).map(lambda x: test_mapper.transform(x))))
                loop.add_metrics(increment_process=len(mixed_kl) / len(mixed_array))
                print(mixed_kl[i] - mixed_ll[i], index[i] < len(x_test))

            mixed_kl = np.concatenate(mixed_kl)
            mixed_kl = mixed_kl - mixed_ll
            cifar_kl = mixed_kl[index < len(x_test)]
            svhn_kl = mixed_kl[index >= len(x_test)]
            loop.add_metrics(kl_histogram=plot_fig([-cifar_kl, -svhn_kl],
                                                   ['red', 'green'],
                                                   [config.in_dataset.name + ' Test',
                                                    config.out_dataset.name + ' Test'],
                                                   'log(bit/dims)',
                                                   'kl_histogram'))
            mixed_kl = mixed_kl - mixed_stand
            cifar_kl = mixed_kl[index < len(x_test)]
            svhn_kl = mixed_kl[index >= len(x_test)]
            loop.add_metrics(kl_with_stand_histogram=plot_fig([-cifar_kl, -svhn_kl],
                                                              ['red', 'green'],
                                                              [config.in_dataset.name + ' Test',
                                                               config.out_dataset.name + ' Test'], 'log(bit/dims)',
                                                              'kl_with_stand_histogram'))


if __name__ == '__main__':
    main()
