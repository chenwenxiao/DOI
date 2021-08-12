# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
import tensorflow as tf
from pprint import pformat

from matplotlib import pyplot
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet import DiscretizedLogistic
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np
from tfsnippet.layers import pixelcnn_2d_output

from tfsnippet.preprocessing import UniformNoiseSampler

from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.datasets.svhn import load_svhn
from ood_regularizer.experiment.models.utils import get_mixed_array, get_noise_array
from ood_regularizer.experiment.utils import make_diagram, get_ele, plot_fig
import os


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 256
    act_norm = False
    weight_norm = False
    batch_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    nf_layers = 20
    pixelcnn_level = 5

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 80
    warm_up_start = 40
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

    in_dataset = 'cifar10'
    out_dataset = 'svhn'
    compressor = 2  # 0 for jpeg, 1 for png, 2 for flif

    max_step = None
    batch_size = 32
    smallest_step = 5e-5
    initial_lr = 0.0002
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = []
    lr_anneal_step_freq = None

    n_critical = 5
    # evaluation parameters
    train_n_qz = 1
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 200
    plot_epoch_freq = 20
    distill_ratio = 1.0
    distill_epoch = 60

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    extra_stride = 2
    count_experiment = False


config = ExpConfig()


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


@add_arg_scope
def dropout(inputs, training=False, scope=None):
    return spt.layers.dropout(inputs, rate=0.2, training=training, name=scope)


@add_arg_scope
@spt.global_reuse
def p_net(input):
    input = tf.to_float(input)
    # prepare for the convolution stack
    output = spt.layers.pixelcnn_2d_input(input)

    # apply the PixelCNN 2D layers.
    for i in range(config.pixelcnn_level):
        output = spt.layers.pixelcnn_conv2d_resnet(
            output,
            out_channels=64,
            vertical_kernel_size=(2, 3),
            horizontal_kernel_size=(2, 2),
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=None,
            dropout_fn=dropout
        )
    output_list = [spt.layers.pixelcnn_conv2d_resnet(
        output,
        out_channels=256,
        vertical_kernel_size=(2, 3),
        horizontal_kernel_size=(2, 2),
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=None,
        dropout_fn=dropout
    ) for i in range(config.x_shape[-1])]
    # get the final output of the PixelCNN 2D network.
    output_list = [pixelcnn_2d_output(output) for output in output_list]
    output = tf.stack(output_list, axis=-2)
    print(output)
    output = tf.reshape(output, (-1,) + config.x_shape + (256,))  # [batch, height, weight, channel, 256]
    return output


@add_arg_scope
@spt.global_reuse
def p_omega_net(input):
    input = tf.to_float(input)
    # prepare for the convolution stack
    output = spt.layers.pixelcnn_2d_input(input)

    # apply the PixelCNN 2D layers.
    for i in range(config.pixelcnn_level):
        output = spt.layers.pixelcnn_conv2d_resnet(
            output,
            out_channels=64,
            vertical_kernel_size=(2, 3),
            horizontal_kernel_size=(2, 2),
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=None,
            dropout_fn=dropout
        )
    output_list = [spt.layers.pixelcnn_conv2d_resnet(
        output,
        out_channels=256,
        vertical_kernel_size=(2, 3),
        horizontal_kernel_size=(2, 2),
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=None,
        dropout_fn=dropout
    ) for i in range(config.x_shape[-1])]
    # get the final output of the PixelCNN 2D network.
    output_list = [pixelcnn_2d_output(output) for output in output_list]
    output = tf.stack(output_list, axis=-2)
    print(output)
    output = tf.reshape(output, (-1,) + config.x_shape + (256,))  # [batch, height, weight, channel, 256]
    return output


class MyIterator(object):
    def __init__(self, iterator):
        self._iterator = iter(iterator)
        self._next = None
        self._has_next = True
        self.next()

    @property
    def has_next(self):
        return self._has_next

    def next(self):
        if not self._has_next:
            raise StopIteration()

        ret = self._next
        try:
            self._next = next(self._iterator)
        except StopIteration:
            self._next = None
            self._has_next = False
        else:
            self._has_next = True
        return ret

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def limited(iterator, n):
    i = 0
    try:
        while i < n:
            yield next(iterator)
            i += 1
    except StopIteration:
        pass


def get_var(name):
    pfx = name.rsplit('/', 1)
    if len(pfx) == 2:
        vars = tf.global_variables(pfx[0] + '/')
    else:
        vars = tf.global_variables()
    for var in vars:
        if var.name.split(':', 1)[0] == name:
            return var
    raise NameError('Variable {} not exist.'.format(name))


def main():
    # parse the arguments
    arg_parser = ArgumentParser()
    spt.register_config_arguments(config, arg_parser, title='Model options')
    spt.register_config_arguments(spt.settings, arg_parser, prefix='tfsnippet',
                                  title='TFSnippet options')
    arg_parser.parse_args(sys.argv[1:])

    # print the config
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    while True:
        try:
            results.make_dirs('plotting/sample', exist_ok=True)
            results.make_dirs('plotting/z_plot', exist_ok=True)
            results.make_dirs('plotting/train.reconstruct', exist_ok=True)
            results.make_dirs('plotting/test.reconstruct', exist_ok=True)
            results.make_dirs('train_summary', exist_ok=True)
            results.make_dirs('checkpoint/checkpoint', exist_ok=True)
            break
        except Exception:
            pass

    if config.count_experiment:
        with open('/home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/count_experiments', 'a') as f:
            f.write(results.system_path("") + '\n')
            f.close()

    # prepare for training and testing data
    # It is important: the `x_shape` must have channel dimension, even it is 1! (i.e. (28, 28, 1) for MNIST)
    # And the value of images should not be normalized, ranged from 0 to 255.
    # prepare for training and testing data
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    (svhn_train, svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)
    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + config.x_shape, name='input_x')
    input_complexity = tf.placeholder(
        dtype=tf.float32, shape=(None,), name='input_complexity')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm, dropout], training=True):
        train_p_net = p_net(input_x)
        train_p_omega_net = p_omega_net(input_x)
        theta_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_x, logits=train_p_net),
            axis=np.arange(-len(config.x_shape), 0)
        )
        theta_loss = tf.reduce_mean(theta_loss)

        omega_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_x, logits=train_p_omega_net),
            axis=np.arange(-len(config.x_shape), 0)
        )
        omega_loss = tf.reduce_mean(omega_loss)
        theta_loss += tf.losses.get_regularization_loss()
        omega_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_p_net = p_net(input_x)
        ele_test_ll = -tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_x, logits=test_p_net),
            axis=np.arange(-len(config.x_shape), 0)
        ) / config.x_shape_multiple / np.log(2)

        test_p_omega_net = p_omega_net(input_x)
        ele_test_omega_ll = -tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_x, logits=test_p_omega_net),
            axis=np.arange(-len(config.x_shape), 0)
        ) / config.x_shape_multiple / np.log(2)

        ele_test_kl = ele_test_omega_ll - ele_test_ll

        # grad_x = tf.gradients(ele_test_ll, [input_x])[0]
        # grad_x_norm = tf.sqrt(tf.reduce_sum((grad_x ** 2), axis=[-1, -2, -3]))

    # derive the optimizer
    with tf.name_scope('optimizing'):
        theta_params = tf.trainable_variables('p_net')
        omega_params = tf.trainable_variables('p_omega_net')
        with tf.variable_scope('theta_optimizer'):
            theta_optimizer = tf.train.AdamOptimizer(learning_rate)
            theta_grads = theta_optimizer.compute_gradients(theta_loss, theta_params)
        with tf.variable_scope('omega_optimizer'):
            omega_optimizer = tf.train.AdamOptimizer(learning_rate)
            omega_grads = omega_optimizer.compute_gradients(omega_loss, omega_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            omega_train_op = omega_optimizer.apply_gradients(omega_grads)
            theta_train_op = theta_optimizer.apply_gradients(theta_grads)
        copy_ops = []
        for i in range(len(theta_params)):
            copy_ops.append(tf.assign(omega_params[i], theta_params[i]))
        copy_ops = tf.group(*copy_ops)

        print(ele_test_ll)
        grad_theta = tf.gradients(ele_test_ll, theta_params)
        print(grad_theta)
        sum_counter = 0
        for grad in grad_theta:
            if grad is not None:
                sum_counter = sum_counter + tf.reduce_sum(grad ** 2)
        sum_counter = tf.sqrt(sum_counter)
        sum_counter = tf.expand_dims(sum_counter, axis=-1)
        print(sum_counter)

    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size)

    cifar_single_train_flow = spt.DataFlow.arrays([x_train], 1)
    cifar_single_test_flow = spt.DataFlow.arrays([x_test], 1)
    svhn_single_train_flow = spt.DataFlow.arrays([svhn_train], 1)
    svhn_single_test_flow = spt.DataFlow.arrays([svhn_test], 1)

    x_train_complexity, x_test_complexity = load_complexity(config.in_dataset, config.compressor)
    svhn_train_complexity, svhn_test_complexity = load_complexity(config.out_dataset, config.compressor)

    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True, skip_incomplete=True)
    mixed_array = get_mixed_array(config, x_train, x_test, svhn_train, svhn_test, normalized=False)
    mixed_test_flow = spt.DataFlow.arrays([mixed_array], config.batch_size,
                                          shuffle=False, skip_incomplete=False)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/cd/d5/02279d802d3a67e701f5',
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/1a/d5/02732c28dc8d842701f5',
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/bd/d5/02279d802d3aec4401f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/6d/d5/02c52d867e43834401f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/5d/d5/02c52d867e43d6b101f5',
            'fashion_mnist': '/mnt/mfs/mlstorage-experiments/cwx17/36/e5/02279d802d3a8bf522f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/d5/e5/02c52d867e43c3c622f5',
            'omniglot': '/mnt/mfs/mlstorage-experiments/cwx17/e5/e5/02c52d867e43e4c622f5',
            'mnist': '/mnt/mfs/mlstorage-experiments/cwx17/4c/d5/02812baa4f7012d622f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/7c/d5/02812baa4f70a83722f5',
            'kmnist': '/mnt/mfs/mlstorage-experiments/cwx17/f5/e5/02c52d867e43382822f5',
            'not_mnist': '/mnt/mfs/mlstorage-experiments/cwx17/66/e5/02279d802d3a7c8822f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/50/e5/02812baa4f700cc9d2f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/ef/e5/02c52d867e430cc9d2f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/c4/e5/02732c28dc8d0cc9d2f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/d4/e5/02732c28dc8d0cc9d2f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/ff/e5/02c52d867e43906cd2f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/f4/e5/02732c28dc8deb5cd2f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/f0/f5/02279d802d3a7b5cd2f5'
        }
        print(experiment_dict)
        if config.in_dataset in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset] + '/checkpoint'
            restore_checkpoint = os.path.join(
                restore_dir, 'checkpoint',
                'checkpoint.dat-{}'.format(config.max_epoch if config.self_ood else config.warm_up_start))
        else:
            restore_dir = results.system_path('checkpoint')
            restore_checkpoint = None

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['q_net', 'p_net', 'posterior_flow', 'G_theta', 'D_psi', 'G_omega', 'D_kappa'],
                           max_epoch=config.max_epoch + 1,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           restore_checkpoint=restore_checkpoint
                           ) as loop:

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()
            fast_end = False
            if config.use_transductive is False and config.out_dataset in experiment_dict and config.mixed_ratio == 1.0:
                fast_end = True

            epoch_iterator = loop.iter_epochs()
            # adversarial training
            for epoch in epoch_iterator:

                if epoch == config.max_epoch + 1 or fast_end:
                    cifar_train_nll, svhn_train_nll, cifar_test_nll, svhn_test_nll = make_diagram(
                        loop, ele_test_ll, [cifar_train_flow, svhn_train_flow, cifar_test_flow, svhn_test_flow],
                        input_x, names=[config.in_dataset + ' Train', config.out_dataset + ' Train',
                                        config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        fig_name='log_prob_histogram'
                    )

                    def stand(base, another_arrays=None):
                        mean, std = np.mean(base), np.std(base)
                        return_arrays = []
                        for array in another_arrays:
                            return_arrays.append(-np.abs((array - mean) / std) * config.stand_weight)
                        return return_arrays

                    [cifar_train_stand, cifar_test_stand, svhn_train_stand, svhn_test_stand] = stand(
                        cifar_train_nll, [cifar_train_nll, cifar_test_nll, svhn_train_nll, svhn_test_nll])

                    loop.collect_metrics(stand_histogram=plot_fig(
                        data_list=[cifar_test_stand, svhn_test_stand],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim',
                        fig_name='stand_histogram'))

                    if config.self_ood:

                        def t_perm(base, another_arrays=None):
                            base = sorted(base)
                            N = len(base)
                            return_arrays = []
                            for array in another_arrays:
                                return_arrays.append(-np.abs(np.searchsorted(base, array) - N // 2))
                            return return_arrays

                        [cifar_train_nll_t, cifar_test_nll_t, svhn_train_nll_t, svhn_test_nll_t] = t_perm(
                            cifar_train_nll, [cifar_train_nll, cifar_test_nll, svhn_train_nll, svhn_test_nll])

                        loop.collect_metrics(T_perm_histogram=plot_fig(
                            data_list=[cifar_test_nll_t, svhn_test_nll_t],
                            color_list=['red', 'green'],
                            label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                            x_label='bits/dim',
                            fig_name='T_perm_histogram'))
                        make_diagram(loop,
                                     ele_test_ll,
                                     [cifar_test_flow, svhn_test_flow],
                                     [input_x],
                                     names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                     fig_name='ll_with_complexity_histogram',
                                     addtion_data=[x_test_complexity, svhn_test_complexity]
                                     )

                    # make_diagram(loop, grad_x_norm,
                    #              [cifar_train_flow,
                    #               cifar_test_flow,
                    #               svhn_train_flow,
                    #               svhn_test_flow],
                    #              [input_x],
                    #              names=[config.in_dataset + ' Train', config.in_dataset + ' Test',
                    #                     config.out_dataset + ' Train', config.out_dataset + ' Test'],
                    #              fig_name='grad_norm_histogram'
                    #              )

                    # make_diagram(
                    #     loop, sum_counter, [cifar_single_train_flow, cifar_single_test_flow, svhn_single_train_flow,
                    #                         svhn_single_test_flow],
                    #     [input_x],
                    #     names=[config.in_dataset + ' Train', config.in_dataset + ' Test',
                    #            config.out_dataset + ' Train', config.out_dataset + ' Test'],
                    #     fig_name='grad_theta_norm_histogram'
                    # )

                    omega_op = ele_test_omega_ll

                    if fast_end:
                        restore_dir = experiment_dict[config.out_dataset] + '/checkpoint'
                        restore_checkpoint = os.path.join(
                            restore_dir, 'checkpoint', 'checkpoint.dat-{}'.format(config.warm_up_start))
                        loop._checkpoint_saver.restore(restore_checkpoint)
                        omega_op = ele_test_ll

                    make_diagram(loop,
                                 omega_op,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='log_prob_mixed_histogram'
                                 )

                    make_diagram(loop,
                                 -omega_op,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='kl_histogram', addtion_data=[cifar_test_nll, svhn_test_nll]
                                 )
                    make_diagram(loop,
                                 -omega_op,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='kl_with_stand_histogram',
                                 addtion_data=[cifar_test_nll + cifar_test_stand, svhn_test_nll + svhn_test_stand]
                                 )

                    loop.print_logs()
                    break

                if epoch == config.warm_up_start + 1:
                    mixed_test_kl = get_ele(ele_test_ll, mixed_test_flow, input_x)
                    mixed_test_flow = spt.DataFlow.arrays([mixed_array, mixed_test_kl],
                                                          config.batch_size, shuffle=True, skip_incomplete=True)

                    if config.pretrain:
                        session.run(copy_ops)

                if epoch <= config.warm_up_start:
                    for step, [x] in loop.iter_steps(train_flow):
                        _, batch_theta_loss = session.run([theta_train_op, theta_loss], feed_dict={
                            input_x: x
                        })
                        loop.collect_metrics(theta_loss=batch_theta_loss)
                else:
                    for step, [x, ll] in loop.iter_steps(mixed_test_flow):
                        if config.self_ood:
                            x = get_noise_array(config, x, normalized=False)
                        else:
                            if config.distill_ratio != 1.0 and config.use_transductive and epoch > config.distill_epoch:
                                ll_omega = session.run(ele_test_omega_ll, feed_dict={
                                    input_x: x
                                })
                                batch_index = np.argsort(ll - ll_omega)
                                batch_index = batch_index[:int(len(batch_index) * config.distill_ratio)]
                                x = x[batch_index]

                        _, batch_omega_loss = session.run([omega_train_op, omega_loss], feed_dict={
                            input_x: x
                        })
                        loop.collect_metrics(omega_loss=batch_omega_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.max_epoch or epoch == config.warm_up_start:
                    loop._checkpoint_saver.save(epoch)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
