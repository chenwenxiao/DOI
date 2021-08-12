# -*- coding: utf-8 -*-
import mltk
from mltk.data import ArraysDataStream, DataStream
import tensorkit as tk
from tensorkit import tensor as T
import sys
import torch
import numpy as np

from flow_next.common import TrainConfig, DataSetConfig, make_dataset, train_model, get_mapper
from flow_next.models.glow import GlowConfig, Glow
from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import plot_fig, make_diagram_torch

from utils.evaluation import dequantized_bpd
import torch.autograd as autograd


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
    mixed_ratio1 = 0.1
    mixed_ratio2 = 0.9
    self_ood = False
    in_dataset_test_ratio = 1.0

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
        grad_global_clip_norm=None,
        # grad_global_clip_norm=100.0,
        debug=True
    )
    model = GlowConfig(
        hidden_conv_activation='relu',
        hidden_conv_channels=[128, 128],
        depth=6,
        levels=3,
        hidden_conv_act_norm=False
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

        experiment_dict = {
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

        def set_train_mode(m):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False

        def set_eval_mode(m):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = True

        model.apply(set_train_mode)
        tk.layers.set_eval_mode(model)
        with mltk.TestLoop() as loop:
            x_train = cifar_dataset.get_array('train', 'x')
            y_train = cifar_dataset.get_array('train', 'y')
            x_test = cifar_dataset.get_array('test', 'x')
            y_test = cifar_dataset.get_array('test', 'y')
            svhn_train = svhn_dataset.get_array('train', 'x')
            svhn_test = svhn_dataset.get_array('test', 'x')

            test_mapper = get_mapper(config.in_dataset, training=False)
            test_mapper.fit(cifar_dataset.slots['x'])

            mixed_array = np.concatenate([
                x_test, svhn_test
            ])
            mixed_test_flow = ArraysDataStream([mixed_array], config.test_batch_size, shuffle=True,
                                               skip_incomplete=True).map(lambda x: test_mapper.transform(x))
            tmp_train_flow = ArraysDataStream([x_train], config.test_batch_size, shuffle=True,
                                              skip_incomplete=True).map(lambda x: test_mapper.transform(x))

            @torch.no_grad()
            def eval_ll(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                bpd = -dequantized_bpd(ll, cifar_train_dataset.slots['x'])
                return T.to_numpy(bpd)

            @torch.no_grad()
            def eval_without_batch_norm_ll(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                bpd = -dequantized_bpd(ll, cifar_train_dataset.slots['x'])
                return T.to_numpy(bpd)

            def permutation_test(flow, ratio):
                R = min(max(1, int(ratio * config.test_batch_size - 1)), config.test_batch_size - 1)
                print('R={}'.format(R))
                packs = []
                for [batch_x] in flow:
                    for i in range(len(batch_x)):
                        for [batch_y] in mixed_test_flow:
                            for [batch_z] in tmp_train_flow:
                                batch = np.concatenate(
                                    [batch_x[i:i + 1], batch_y[:R], batch_z[:config.test_batch_size - R - 1]],
                                    axis=0)
                                pack = eval_ll(batch)
                                pack = np.asarray(pack)[:1]
                                break
                            break
                        packs.append(pack)
                packs = np.concatenate(packs, axis=0)  # [len_of_flow]
                print(packs.shape)
                return packs

            def delta_test(flow):
                return permutation_test(flow, config.mixed_ratio1) - permutation_test(flow, config.mixed_ratio2)

            cifar_r1 = permutation_test(cifar_test_flow, config.mixed_ratio1)
            cifar_r2 = permutation_test(cifar_test_flow, config.mixed_ratio2)
            svhn_r1 = permutation_test(svhn_test_flow, config.mixed_ratio1)
            svhn_r2 = permutation_test(svhn_test_flow, config.mixed_ratio2)

            loop.add_metrics(r1_histogram=plot_fig(
                [cifar_r1, cifar_r2, svhn_r1, svhn_r2],
                ['red', 'salmon', 'green', 'lightgreen'],
                [config.in_dataset.name + ' r1', config.in_dataset.name + ' r2',
                 config.out_dataset.name + ' r1', config.out_dataset.name + ' r2'],
                'log(bit/dims)',
                'batch_norm_log_pro_histogram', auc_pair=(0, 2)))

            loop.add_metrics(r2_histogram=plot_fig(
                [cifar_r1, cifar_r2, svhn_r1, svhn_r2],
                ['red', 'salmon', 'green', 'lightgreen'],
                [config.in_dataset.name + ' r1', config.in_dataset.name + ' r2',
                 config.out_dataset.name + ' r1', config.out_dataset.name + ' r2'],
                'log(bit/dims)',
                'batch_norm_log_pro_histogram', auc_pair=(1, 3)))

            loop.add_metrics(r1_r2_histogram=plot_fig(
                [cifar_r1 - cifar_r2, svhn_r1 - svhn_r2],
                ['red', 'green'],
                [config.in_dataset.name + ' test',
                 config.out_dataset.name + ' test'],
                'log(bit/dims)',
                'r1_r2_log_pro_histogram',
                auc_pair=(0, 1)))

            loop.add_metrics(abs_r1_r2_histogram=plot_fig(
                [-np.abs(cifar_r1 - cifar_r2), -np.abs(svhn_r1 - svhn_r2)],
                ['red', 'green'],
                [config.in_dataset.name + ' test',
                 config.out_dataset.name + ' test'],
                'log(bit/dims)',
                'abs_r1_r2_log_pro_histogram',
                auc_pair=(0, 1)))

            make_diagram_torch(loop,
                               eval_ll,
                               [cifar_test_flow, svhn_test_flow],
                               names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                               fig_name='log_prob_with_batch_norm_histogram'
                               )

            model.apply(set_eval_mode)
            make_diagram_torch(loop,
                               eval_without_batch_norm_ll,
                               [cifar_test_flow, svhn_test_flow],
                               names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                               fig_name='log_prob_without_batch_norm_histogram'
                               )


if __name__ == '__main__':
    main()
