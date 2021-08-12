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

from tfsnippet.preprocessing import UniformNoiseSampler

from ood_regularizer.experiment.datasets.celeba import load_celeba
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

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 200
    warm_up_start = 100
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
    test_epoch_freq = 200
    plot_epoch_freq = 20
    distill_ratio = 1.0
    distill_epoch = 150
    mcmc_times = 5

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
    print(inputs, training)
    return spt.layers.dropout(inputs, rate=0.2, training=training, name=scope)


@add_arg_scope
@spt.global_reuse
def q_net(x, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg), ):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=config.extra_stride,
                                             scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean')
        z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd')

    # sample z ~ q(z|x)
    z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q)),
                n_samples=n_z, group_ndims=1)

    return net


@add_arg_scope
@spt.global_reuse
def p_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    z = net.add('z', normal, n_samples=n_z, group_ndims=1)

    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 128 * config.x_shape[0] // 4 * config.x_shape[
            1] // 4 // config.extra_stride // config.extra_stride, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4 // config.extra_stride, config.x_shape[1] // 4 // config.extra_stride, 128)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=config.extra_stride,
                                               scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
        x_mean = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_mean',
            kernel_initializer=tf.zeros_initializer(),  # activation_fn=tf.nn.tanh
        )
        x_logstd = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_logstd',
            kernel_initializer=tf.zeros_initializer(),
        )

    beta = tf.get_variable(name='beta', shape=(), initializer=tf.constant_initializer(config.initial_beta),
                           dtype=tf.float32, trainable=True)
    x = net.add('x', DiscretizedLogistic(
        mean=x_mean,
        log_scale=spt.ops.maybe_clip_value(beta if config.uniform_scale else x_logstd, min_val=config.epsilon),
        bin_size=2.0 / 256.0,
        min_val=-1.0 + 1.0 / 256.0,
        max_val=1.0 - 1.0 / 256.0,
        epsilon=1e-10
    ), group_ndims=3)
    return net


@add_arg_scope
@spt.global_reuse
def q_omega_net(x, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg), ):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=config.extra_stride,
                                             scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean')
        z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd')

    # sample z ~ q(z|x)
    z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q)),
                n_samples=n_z, group_ndims=1)

    return net


@add_arg_scope
@spt.global_reuse
def p_omega_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    z = net.add('z', normal, n_samples=n_z, group_ndims=1)

    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 128 * config.x_shape[0] // 4 * config.x_shape[
            1] // 4 // config.extra_stride // config.extra_stride, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4 // config.extra_stride, config.x_shape[1] // 4 // config.extra_stride, 128)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=config.extra_stride,
                                               scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
        x_mean = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_mean',
            kernel_initializer=tf.zeros_initializer(),  # activation_fn=tf.nn.tanh
        )
        x_logstd = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_logstd',
            kernel_initializer=tf.zeros_initializer(),
        )

    beta = tf.get_variable(name='beta', shape=(), initializer=tf.constant_initializer(config.initial_beta),
                           dtype=tf.float32, trainable=True)
    x = net.add('x', DiscretizedLogistic(
        mean=x_mean,
        log_scale=spt.ops.maybe_clip_value(beta if config.uniform_scale else x_logstd, min_val=config.epsilon),
        bin_size=2.0 / 256.0,
        min_val=-1.0 + 1.0 / 256.0,
        max_val=1.0 - 1.0 / 256.0,
        epsilon=1e-10
    ), group_ndims=3)
    return net


def get_all_loss(q_net, p_net):
    with tf.name_scope('adv_prior_loss'):
        train_recon = p_net['x'].log_prob()
        train_kl = tf.reduce_mean(
            -p_net['z'].log_prob() + q_net['z'].log_prob()
        )
        VAE_loss = -train_recon + train_kl
    return VAE_loss


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
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    (svhn_train, svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)

    def normalize(x):
        return [(x - 127.5) / 256.0 * 2]

    def double(x):
        return [x, x]

    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x
    if config.x_shape == (28, 28, 1):
        config.extra_stride = 1

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    input_complexity = tf.placeholder(
        dtype=tf.float32, shape=(None,), name='input_complexity')
    input_y = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_y')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_q_net = q_net(input_x, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z']},
                            n_z=config.train_n_qz)
        VAE_loss = get_all_loss(train_q_net, train_p_net)
        train_q_omega_net = q_omega_net(input_x, n_z=config.train_n_qz)
        train_p_omega_net = p_omega_net(observed={'x': input_x, 'z': train_q_omega_net['z']},
                                        n_z=config.train_n_qz)
        VAE_omega_loss = get_all_loss(train_q_omega_net, train_p_omega_net)

        VAE_loss += tf.losses.get_regularization_loss()
        VAE_omega_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_qz)
        test_chain = test_q_net.chain(p_net, observed={'x': input_y}, n_z=config.test_n_qz, latent_axis=0)
        ele_test_recon_sample = test_chain.model['x'].distribution.mean[0]
        ele_test_recon = tf.reduce_mean(test_chain.model['x'].log_prob(), axis=0) / config.x_shape_multiple / np.log(2)
        test_recon = tf.reduce_mean(
            ele_test_recon
        )
        ele_test_ll = test_chain.vi.evaluation.is_loglikelihood() / config.x_shape_multiple / np.log(2)
        test_nll = -tf.reduce_mean(
            ele_test_ll
        )
        ele_test_lb = test_chain.vi.lower_bound.elbo() / config.x_shape_multiple / np.log(2)
        print(ele_test_lb)
        test_lb = tf.reduce_mean(ele_test_lb)

        test_q_omega_net = q_omega_net(input_x, n_z=config.test_n_qz)
        test_omega_chain = test_q_omega_net.chain(p_omega_net, observed={'x': input_x}, n_z=config.test_n_qz,
                                                  latent_axis=0)
        test_omega_recon = tf.reduce_mean(
            test_omega_chain.model['x'].log_prob()
        )
        ele_test_omega_ll = test_omega_chain.vi.evaluation.is_loglikelihood() / config.x_shape_multiple / np.log(2)
        test_omega_nll = -tf.reduce_mean(
            ele_test_omega_ll
        )
        test_omega_lb = tf.reduce_mean(test_omega_chain.vi.lower_bound.elbo())

        ele_test_kl = ele_test_omega_ll - ele_test_ll

        eval_test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0)
        eval_test_ll = eval_test_chain.vi.evaluation.is_loglikelihood() / config.x_shape_multiple / np.log(2)
        grad_x = tf.gradients(eval_test_ll, [input_x])[0]
        print(grad_x)
        grad_x_norm = tf.sqrt(tf.reduce_sum((grad_x ** 2), axis=[-1, -2, -3]))

    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('p_net')
        VAE_omega_params = tf.trainable_variables('q_omega_net') + tf.trainable_variables('p_omega_net')
        with tf.variable_scope('theta_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)
        with tf.variable_scope('omega_optimizer'):
            VAE_omega_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_omega_grads = VAE_omega_optimizer.compute_gradients(VAE_omega_loss, VAE_omega_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)
            VAE_omega_train_op = VAE_omega_optimizer.apply_gradients(VAE_omega_grads)
        copy_ops = []
        for i in range(len(VAE_params)):
            copy_ops.append(tf.assign(VAE_omega_params[i], VAE_params[i]))
        copy_ops = tf.group(*copy_ops)

        print(eval_test_ll)
        grad_theta = tf.gradients(eval_test_ll, VAE_params)
        print(grad_theta)
        sum_counter = 0
        for grad in grad_theta:
            if grad is not None:
                sum_counter = sum_counter + tf.reduce_sum(grad ** 2)
        sum_counter = tf.sqrt(sum_counter)
        sum_counter = tf.expand_dims(sum_counter, axis=-1)
        print(sum_counter)

    # derive the plotting function
    with tf.name_scope('plotting'):
        plot_net = p_net(n_z=config.sample_n_z)
        vae_plots = tf.reshape(plot_net['x'].distribution.mean, (-1,) + config.x_shape)
        vae_plots = 256.0 * vae_plots / 2 + 127.5
        reconstruct_q_net = q_net(input_x)
        reconstruct_z = reconstruct_q_net['z']
        reconstruct_plots = 256.0 * tf.reshape(
            p_net(observed={'z': reconstruct_z})['x'].distribution.mean,
            (-1,) + config.x_shape
        ) / 2 + 127.5
        reconstruct_plots = tf.clip_by_value(reconstruct_plots, 0, 255)
        vae_plots = tf.clip_by_value(vae_plots, 0, 255)

        plot_omega_net = p_omega_net(n_z=config.sample_n_z)
        vae_omega_plots = tf.reshape(plot_omega_net['x'].distribution.mean, (-1,) + config.x_shape)
        vae_omega_plots = 256.0 * vae_omega_plots / 2 + 127.5
        reconstruct_q_omega_net = q_omega_net(input_x)
        reconstruct_omega_z = reconstruct_q_omega_net['z']
        reconstruct_omega_plots = 256.0 * tf.reshape(
            p_omega_net(observed={'z': reconstruct_omega_z})['x'].distribution.mean,
            (-1,) + config.x_shape
        ) / 2 + 127.5
        reconstruct_omega_plots = tf.clip_by_value(reconstruct_omega_plots, 0, 255)
        vae_omega_plots = tf.clip_by_value(vae_omega_plots, 0, 255)

    def plot_samples(loop, extra_index=None):
        if extra_index is None:
            extra_index = loop.epoch

        try:
            with loop.timeit('plot_time'):
                # plot reconstructs
                def plot_reconnstruct(flow, name, plots):
                    for [x] in flow:
                        x_samples = x
                        images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                        images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                        images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                        batch_reconstruct_plots = session.run(
                            plots, feed_dict={input_x: x_samples})
                        images[2::3, ...] = np.round(batch_reconstruct_plots)
                        # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                        save_images_collection(
                            images=images,
                            filename='plotting/{}-{}.png'.format(name, extra_index),
                            grid_size=(20, 15),
                            results=results,
                        )
                        break

                plot_reconnstruct(reconstruct_test_flow, 'test.reconstruct/theta', reconstruct_plots)
                plot_reconnstruct(reconstruct_train_flow, 'train.reconstruct/theta', reconstruct_plots)
                plot_reconnstruct(reconstruct_omega_test_flow, 'test.reconstruct/omega', reconstruct_omega_plots)
                plot_reconnstruct(reconstruct_omega_train_flow, 'train.reconstruct/omega', reconstruct_omega_plots)

                # plot samples
                images = session.run(vae_plots)
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}-{}.png'.format('theta', extra_index),
                    grid_size=(10, 10),
                    results=results,
                )
                images = session.run(vae_omega_plots)
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}-{}.png'.format('omega', extra_index),
                    grid_size=(10, 10),
                    results=results,
                )
        except Exception as e:
            print(e)

    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size).map(normalize).map(double)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size).map(normalize).map(double)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size).map(normalize).map(double)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size).map(normalize).map(double)

    cifar_single_train_flow = spt.DataFlow.arrays([x_train], 1).map(normalize).map(double)
    cifar_single_test_flow = spt.DataFlow.arrays([x_test], 1).map(normalize).map(double)
    svhn_single_train_flow = spt.DataFlow.arrays([svhn_train], 1).map(normalize).map(double)
    svhn_single_test_flow = spt.DataFlow.arrays([svhn_test], 1).map(normalize).map(double)

    x_train_complexity, x_test_complexity = load_complexity(config.in_dataset, config.compressor)
    svhn_train_complexity, svhn_test_complexity = load_complexity(config.out_dataset, config.compressor)

    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True,
                                     skip_incomplete=True).map(normalize)

    reconstruct_test_flow = spt.DataFlow.arrays([x_test], 100, shuffle=True, skip_incomplete=True).map(normalize)
    reconstruct_train_flow = spt.DataFlow.arrays([x_train], 100, shuffle=True, skip_incomplete=True).map(normalize)
    reconstruct_omega_test_flow = spt.DataFlow.arrays([svhn_test], 100, shuffle=True, skip_incomplete=True).map(
        normalize)
    reconstruct_omega_train_flow = spt.DataFlow.arrays([svhn_train], 100, shuffle=True, skip_incomplete=True).map(
        normalize)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
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

            epoch_iterator = loop.iter_epochs()
            # adversarial training
            for epoch in epoch_iterator:

                if epoch == config.warm_up_start + 1:
                    mixed_array = get_mixed_array(config, x_train, x_test, svhn_train, svhn_test)
                    print(mixed_array.shape)
                    mixed_test_flow = spt.DataFlow.arrays(
                        [mixed_array], config.batch_size, shuffle=False,
                        skip_incomplete=False).map(normalize).map(double)
                    mixed_test_kl = get_ele(ele_test_ll, mixed_test_flow, [input_x, input_y])
                    mixed_test_flow = spt.DataFlow.arrays(
                        [mixed_array, mixed_test_kl], config.batch_size, shuffle=True,
                        skip_incomplete=True)

                    if config.pretrain:
                        session.run(copy_ops)

                make_diagram(loop,
                             ele_test_ll - ele_test_omega_ll,
                             [cifar_test_flow, svhn_test_flow], [input_x, input_y],
                             names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                             fig_name='kl_histogram_{}'.format(epoch),
                             )

                if epoch <= config.warm_up_start:
                    for step, [x] in loop.iter_steps(train_flow):
                        _, batch_VAE_loss = session.run([VAE_train_op, VAE_loss], feed_dict={
                            input_x: x
                        })
                        loop.collect_metrics(VAE_loss=batch_VAE_loss)
                else:
                    for step, [x, ll] in loop.iter_steps(mixed_test_flow):
                        [x] = normalize(x)
                        if config.self_ood:
                            x = get_noise_array(config, x)
                        else:
                            if config.distill_ratio != 1.0 and config.use_transductive and epoch > config.distill_epoch:
                                ll_omega = session.run(ele_test_omega_ll, feed_dict={
                                    input_x: x
                                })
                                batch_index = np.argsort(ll - ll_omega)
                                batch_index = batch_index[:int(len(batch_index) * config.distill_ratio)]
                                x = x[batch_index]

                        _, batch_VAE_omega_loss = session.run([VAE_omega_train_op, VAE_omega_loss], feed_dict={
                            input_x: x
                        })
                        loop.collect_metrics(VAE_omega_loss=batch_VAE_omega_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.max_epoch or epoch == config.warm_up_start:
                    loop._checkpoint_saver.save(epoch)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
