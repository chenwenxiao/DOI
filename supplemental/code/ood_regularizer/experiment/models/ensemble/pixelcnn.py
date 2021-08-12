# -*- coding: utf-8 -*-
import functools
import sys
import os
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

from ood_regularizer.experiment.datasets.celeba import load_celeba
from ood_regularizer.experiment.datasets.overall import load_overall
from ood_regularizer.experiment.datasets.svhn import load_svhn
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import make_diagram, get_ele, plot_fig


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
    max_epoch = 200
    warm_up_start = 0
    initial_beta = -3.0
    uniform_scale = False
    use_transductive = True
    mixed_train = False
    self_ood = False
    mixed_ratio = 1.0
    mutation_rate = 0.1
    noise_type = "mutation"  # or unit
    in_dataset_test_ratio = 1.0

    in_dataset = 'cifar10'
    out_dataset = 'svhn'

    max_step = None
    batch_size = 128
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
    test_epoch_freq = 40
    plot_epoch_freq = 20
    distill_ratio = 1.0
    distill_epoch = 10

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    extra_stride = 2
    class_num = 10
    ensemble_times = 5
    count_experiment = False


config = ExpConfig()


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


@add_arg_scope
def dropout(inputs, training=False, scope=None):
    print(inputs, training)
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

    # prepare for training and testing data
    # It is important: the `x_shape` must have channel dimension, even it is 1! (i.e. (28, 28, 1) for MNIST)
    # And the value of images should not be normalized, ranged from 0 to 255.
    # prepare for training and testing data
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    (svhn_train, svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)
    config.x_shape = x_train.shape[1:]

    config.class_num = np.max(y_train) + 1
    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x
    if config.x_shape == (28, 28, 1):
        config.extra_stride = 1
    config.max_epoch = config.warm_up_start + config.test_epoch_freq * config.ensemble_times

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + config.x_shape, name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=(None,), name='input_y')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm, dropout], training=True):
        train_p_net = p_net(input_x)
        theta_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_x, logits=train_p_net),
            axis=np.arange(-len(config.x_shape), 0)
        )
        theta_loss = tf.reduce_mean(theta_loss)

        theta_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_p_net = p_net(input_x)
        ele_test_ll = -tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_x, logits=test_p_net),
            axis=np.arange(-len(config.x_shape), 0)
        ) / config.x_shape_multiple / np.log(2)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        theta_params = tf.trainable_variables('p_net')
        with tf.variable_scope('theta_optimizer'):
            theta_optimizer = tf.train.AdamOptimizer(learning_rate)
            theta_grads = theta_optimizer.compute_gradients(theta_loss, theta_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            theta_train_op = theta_optimizer.apply_gradients(theta_grads)

    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size)

    train_flow = spt.DataFlow.arrays([x_train, y_train], config.batch_size, shuffle=True,
                                     skip_incomplete=True)

    reconstruct_test_flow = spt.DataFlow.arrays([x_test], 100, shuffle=True, skip_incomplete=True)
    reconstruct_train_flow = spt.DataFlow.arrays([x_train], 100, shuffle=True, skip_incomplete=True)
    reconstruct_omega_test_flow = spt.DataFlow.arrays([svhn_test], 100, shuffle=True, skip_incomplete=True)
    reconstruct_omega_train_flow = spt.DataFlow.arrays([svhn_train], 100, shuffle=True, skip_incomplete=True)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
        }
        print(experiment_dict)
        if config.in_dataset in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset] + '/checkpoint'
            restore_checkpoint = os.path.join(
                restore_dir, 'checkpoint', 'checkpoint.dat-{}'.format(config.max_epoch))
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

            epoch_iterator = loop.iter_epochs()
            # adversarial training
            for epoch in epoch_iterator:

                if epoch == config.max_epoch + 1:
                    final_cifar_test_ll = []
                    final_svhn_test_ll = []
                    for current_class in range(0, config.ensemble_times):
                        pse_epoch = config.warm_up_start + (current_class + 1) * config.test_epoch_freq
                        loop._checkpoint_saver.restore(os.path.join(
                            restore_dir, 'checkpoint', 'checkpoint.dat-{}'.format(pse_epoch)))
                        final_cifar_test_ll.append(get_ele(ele_test_ll, cifar_test_flow, input_x))
                        final_svhn_test_ll.append(get_ele(ele_test_ll, svhn_test_flow, input_x))

                    def get_bpd_waic(arrays):
                        arrays = np.stack(arrays, axis=0)
                        waic = np.mean(arrays, axis=0) - np.var(arrays, axis=0)
                        return waic

                    def get_ll_waic(arrays):
                        arrays = np.stack(arrays, axis=0)
                        arrays = arrays * config.x_shape_multiple * np.log(2)
                        waic = np.mean(arrays, axis=0) - np.var(arrays, axis=0)
                        return waic

                    def get_mean(arrays):
                        arrays = np.stack(arrays, axis=0)
                        return np.mean(arrays, axis=0)

                    def get_var(arrays):
                        arrays = np.stack(arrays, axis=0)
                        return -np.var(arrays, axis=0)

                    loop.collect_metrics(bpd_waic_histogram=plot_fig(
                        data_list=[get_bpd_waic(final_cifar_test_ll), get_bpd_waic(final_svhn_test_ll)],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim', fig_name='bpd_waic_histogram'))

                    loop.collect_metrics(ll_waic_histogram=plot_fig(
                        data_list=[get_ll_waic(final_cifar_test_ll), get_ll_waic(final_svhn_test_ll)],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim', fig_name='ll_waic_histogram'))

                    loop.collect_metrics(mean_log_prob_histogram=plot_fig(
                        data_list=[get_mean(final_cifar_test_ll), get_mean(final_svhn_test_ll)],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim', fig_name='mean_log_prob_histogram'))

                    loop.collect_metrics(var_log_prob_histogram=plot_fig(
                        data_list=[get_var(final_cifar_test_ll), get_var(final_svhn_test_ll)],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim', fig_name='var_log_prob_histogram'))
                    loop.print_logs()
                    break

                if (epoch - config.warm_up_start) % config.test_epoch_freq == 1 and epoch > config.warm_up_start:
                    session.run(tf.global_variables_initializer())  # Initialize all variables

                if epoch > config.warm_up_start:
                    for step, [x, y] in loop.iter_steps(train_flow):
                        _, batch_theta_loss = session.run([theta_train_op, theta_loss], feed_dict={
                            input_x: x
                        })
                        loop.collect_metrics(theta_loss=batch_theta_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if (epoch - config.warm_up_start) % config.test_epoch_freq == 0:
                    loop._checkpoint_saver.save(epoch)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
