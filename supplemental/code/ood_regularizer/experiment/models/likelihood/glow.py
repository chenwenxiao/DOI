# -*- coding: utf-8 -*-
import mltk
from mltk.data import ArraysDataStream, DataStream
from tensorkit import tensor as T
import sys
import torch
import numpy as np
import tensorkit as tk

from flow_next.common import TrainConfig, DataSetConfig, make_dataset, train_model, get_mapper
from flow_next.models.glow import GlowConfig, Glow
from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.models.utils import get_mixed_array, get_noise_array
from ood_regularizer.experiment.utils import plot_fig, make_diagram_torch, get_ele_torch
from utils.data.mappers import ArrayMapperList

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
    self_ood = False
    mixed_ratio = 1.0
    mutation_rate = 0.1
    noise_type = "mutation"  # or unit
    in_dataset_test_ratio = 1.0
    pretrain = False
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
    distill_ratio = 0.5
    distill_epoch = 20

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
    )
    in_dataset = 'cifar10'
    out_dataset = 'svhn'
    count_experiment = False


epoch_counter = 0


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

        cifar_single_train_flow = cifar_test_dataset.get_stream('train', 'x', 1)
        cifar_single_test_flow = cifar_test_dataset.get_stream('test', 'x', 1)
        svhn_single_train_flow = svhn_test_dataset.get_stream('train', 'x', 1)
        svhn_single_test_flow = svhn_test_dataset.get_stream('test', 'x', 1)

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

            @torch.no_grad()
            def eval_log_det(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                log_det = outputs[0].log_det
                for output in outputs[1:]:
                    log_det = log_det + output.log_det
                log_det = -dequantized_bpd(log_det, cifar_train_dataset.slots['x'])
                return T.to_numpy(log_det)

            cifar_train_ll, svhn_train_ll, cifar_test_ll, svhn_test_ll = make_diagram_torch(
                loop, eval_ll,
                [cifar_train_flow, svhn_train_flow, cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Train', config.out_dataset.name + ' Train',
                       config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='log_prob_histogram'
            )

            def stand(base, another_arrays=None):
                mean, std = np.mean(base), np.std(base)
                return_arrays = []
                for array in another_arrays:
                    return_arrays.append(-np.abs((array - mean) / std) * config.stand_weight)
                return return_arrays

            [cifar_train_stand, cifar_test_stand, svhn_train_stand, svhn_test_stand] = stand(
                cifar_train_ll, [cifar_train_ll, cifar_test_ll, svhn_train_ll, svhn_test_ll])

            loop.add_metrics(stand_histogram=plot_fig(
                data_list=[cifar_test_stand, svhn_test_stand],
                color_list=['red', 'green'],
                label_list=[config.out_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                x_label='bits/dim', fig_name='stand_histogram'))

            if config.self_ood:
                def t_perm(base, another_arrays=None):
                    base = sorted(base)
                    N = len(base)
                    return_arrays = []
                    for array in another_arrays:
                        return_arrays.append(-np.abs(np.searchsorted(base, array) - N // 2))
                    return return_arrays

                [cifar_train_nll_t, cifar_test_nll_t, svhn_train_nll_t, svhn_test_nll_t] = t_perm(
                    cifar_train_ll, [cifar_train_ll, cifar_test_ll, svhn_train_ll, svhn_test_ll])

                loop.add_metrics(T_perm_histogram=plot_fig(
                    data_list=[cifar_test_nll_t, svhn_test_nll_t],
                    color_list=['red', 'green'],
                    label_list=[config.out_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                    x_label='bits/dim', fig_name='T_perm_histogram'))

                loop.add_metrics(ll_with_complexity_histogram=plot_fig(
                    data_list=[cifar_test_ll + x_test_complexity, svhn_test_ll + svhn_test_complexity],
                    color_list=['red', 'green'],
                    label_list=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                    x_label='bits/dim', fig_name='ll_with_complexity_histogram'))

                cifar_test_det, svhn_test_det = make_diagram_torch(
                    loop, eval_log_det,
                    [cifar_test_flow, svhn_test_flow],
                    names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                    fig_name='log_det_histogram')

                def eval_grad_norm(x):
                    x = T.from_numpy(x)
                    x.requires_grad = True
                    ll, outputs = model(x)
                    gradients = autograd.grad(ll, x, grad_outputs=torch.ones(ll.size()).cuda())[0]
                    grad_norm = gradients.view(gradients.size()[0], -1).norm(2, 1)
                    return T.to_numpy(grad_norm)

                theta_params = tk.layers.iter_parameters(model)

                def eval_grad_theta(x):
                    x = T.from_numpy(x)
                    x.requires_grad = True
                    ll, outputs = model(x)
                    gradients = autograd.grad(ll, theta_params, grad_outputs=torch.ones(ll.size()).cuda())
                    grad_norm = 0
                    for grad in gradients:
                        grad_norm = grad_norm + grad.norm(2)
                    grad_norm = T.expand_dim(grad_norm, axis=-1)

                    return T.to_numpy(grad_norm)

                make_diagram_torch(
                    loop, eval_grad_norm,
                    [cifar_single_test_flow, svhn_single_test_flow],
                    names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                    fig_name='grad_norm_histogram')

                loop.add_metrics(origin_log_prob_histogram=plot_fig(
                    data_list=[cifar_test_ll - cifar_test_det, svhn_test_ll - svhn_test_det],
                    color_list=['red', 'green'],
                    label_list=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                    x_label='bits/dim', fig_name='origin_log_prob_histogram'))

            fast_end = False
            if config.use_transductive is False and config.out_dataset.name in experiment_dict and config.mixed_ratio == 1.0:
                fast_end = True

            if config.self_ood and restore_checkpoint is not None:
                model = torch.load(restore_checkpoint + '/omega_model.pkl')
            elif fast_end:
                restore_checkpoint = experiment_dict[config.out_dataset.name]
                model = torch.load(restore_checkpoint + '/model.pkl')
            else:
                mixed_array = get_mixed_array(
                    config,
                    cifar_dataset.get_array('train', 'x'),
                    cifar_dataset.get_array('test', 'x'),
                    svhn_dataset.get_array('train', 'x'),
                    svhn_dataset.get_array('test', 'x'), normalized=False
                )
                print(mixed_array.shape)
                test_mapper = get_mapper(config.in_dataset, training=False)
                train_mapper = get_mapper(config.in_dataset, training=True)
                test_mapper.fit(cifar_dataset.slots['x'])
                train_mapper.fit(cifar_dataset.slots['x'])
                mixed_stream = ArraysDataStream(
                    [mixed_array], batch_size=config.batch_size, shuffle=False,
                    skip_incomplete=False).map(
                    lambda x: test_mapper.transform(x))
                mixed_ll = get_ele_torch(eval_ll, mixed_stream)
                mixed_stream = ArraysDataStream([mixed_array, mixed_ll], batch_size=config.batch_size, shuffle=True,
                                                skip_incomplete=True)
                if not config.pretrain:
                    model = Glow(cifar_train_dataset.slots['x'], exp.config.model)

                def data_generator():
                    global epoch_counter
                    epoch_counter = epoch_counter + 1
                    print('epoch_counter = {}'.format(epoch_counter))
                    for [x, ll] in mixed_stream:
                        if config.self_ood:
                            x = get_noise_array(config, x, normalized=False)
                        x = train_mapper.transform(x)
                        if not config.self_ood:
                            if config.distill_ratio != 1.0 and config.use_transductive and epoch_counter > config.distill_epoch:
                                ll_omega = eval_ll(x)
                                batch_index = np.argsort(ll - ll_omega)
                                batch_index = batch_index[:int(len(batch_index) * config.distill_ratio)]
                                x = x[batch_index]
                        yield [T.from_numpy(x)]

                if config.use_transductive or config.self_ood:
                    train_model(exp, model, svhn_train_dataset, svhn_test_dataset, DataStream.generator(data_generator))
                else:
                    train_model(exp, model, svhn_train_dataset, svhn_test_dataset)

            torch.save(model, 'omega_model.pkl')

            make_diagram_torch(
                loop, eval_ll,
                [cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='log_prob_mixed_histogram'
            )

            make_diagram_torch(
                loop, lambda x: -eval_ll(x),
                [cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='kl_histogram',
                addtion_data=[cifar_test_ll, svhn_test_ll]
            )

            make_diagram_torch(
                loop, lambda x: -eval_ll(x),
                [cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='kl_with_stand_histogram',
                addtion_data=[cifar_test_ll + cifar_test_stand, svhn_test_ll + svhn_test_stand]
            )


if __name__ == '__main__':
    main()
