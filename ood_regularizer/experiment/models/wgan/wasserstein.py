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
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np

from tfsnippet.preprocessing import UniformNoiseSampler

from ood_regularizer.experiment.datasets.celeba import load_celeba
from ood_regularizer.experiment.datasets.overall import load_overall
from ood_regularizer.experiment.datasets.svhn import load_svhn
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import make_diagram, get_ele
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
    max_epoch = 150
    warm_up_start = 100

    min_distance = 0.2
    use_transductive = True  # can model use the data in SVHN's and CIFAR's testing set
    mixed_train = False
    use_gan = False  # if use_gan == True, you should set warm_up_start to 1000 to ensure the pre-training for gan
    self_ood = False
    mixed_ratio = 1.0
    mutation_rate = 0.1
    in_dataset_test_ratio = 1.0
    pretrain = False

    in_dataset = 'cifar10'
    out_dataset = 'svhn'

    max_step = None
    batch_size = 128
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = []
    lr_anneal_step_freq = None

    gradient_penalty_algorithm = 'interpolate'  # both or interpolate
    gradient_penalty_weight = 2
    gradient_penalty_index = 6

    n_critical = 5  # TODO
    # evaluation parameters
    train_n_pz = 128
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 100
    plot_epoch_freq = 10
    distill_ratio = 1.0
    distill_epoch = 50

    sample_n_z = 100
    epsilon = -20
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
def G_omega(z):
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 128 * config.x_shape[0] // 4 * config.x_shape[1] // 4, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4, config.x_shape[1] // 4, 128)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
    x_mean = spt.layers.conv2d(
        h_z, config.x_shape[-1], (1, 1), padding='same', scope='feature_map_mean_to_pixel',
        kernel_initializer=tf.zeros_initializer(),  # activation_fn=tf.nn.tanh
    )
    return x_mean


@add_arg_scope
@spt.global_reuse
def D_psi(x, y=None):
    # if y is not None:
    #     return D_psi(y) + 0.1 * tf.sqrt(tf.reduce_sum((x - y) ** 2, axis=tf.range(-len(config.x_shape), 0)))
    # TODO
    normalizer_fn = None
    # x = tf.round(256.0 * x / 2 + 127.5)
    # x = (x - 127.5) / 256.0 * 2
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        h_x = spt.layers.dense(h_x, 64, scope='level_-2')
    # sample z ~ q(z|x)
    h_x = spt.layers.dense(h_x, 1, scope='level_-1')
    # h_x = tf.clip_by_value(h_x, -1000, 1000)
    return tf.squeeze(h_x, axis=-1)


@add_arg_scope
@spt.global_reuse
def p_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    z = net.add('z', normal, n_samples=n_z, group_ndims=1)
    x_mean = G_omega(z)
    x = net.add('x', spt.Normal(
        mean=x_mean, logstd=0.0
    ), group_ndims=3)
    return net


def get_all_loss(input_x, input_y):
    with tf.name_scope('adv_prior_loss'):
        x = input_x
        x_ = input_y
        energy_real = D_psi(x)
        energy_fake = D_psi(x_)

        gradient_penalty = 0.0

        if config.gradient_penalty_algorithm == 'interpolate':
            # Sample from interpolates
            batch_size = spt.utils.get_shape(input_x)[0]
            alpha = tf.random_uniform(
                tf.concat([[batch_size], [1] * len(config.x_shape)], axis=0),
                minval=0, maxval=1.0
            )
            x = tf.reshape(x, (-1,) + config.x_shape)
            x_ = tf.reshape(x_, (-1,) + config.x_shape)
            differences = x - x_
            interpolates = x_ + alpha * differences
            # print(interpolates)
            D_interpolates = D_psi(interpolates)
            # print(D_interpolates)
            gradient_penalty = tf.square(tf.gradients(D_interpolates, [interpolates])[0])
            gradient_penalty = tf.reduce_sum(gradient_penalty, tf.range(-len(config.x_shape), 0))
            gradient_penalty = tf.pow(gradient_penalty, config.gradient_penalty_index / 2.0)
            gradient_penalty = tf.reduce_mean(gradient_penalty) * config.gradient_penalty_weight

        if config.gradient_penalty_algorithm == 'both':
            # Sample from fake and real
            gradient_penalty_real = tf.square(tf.gradients(energy_real, [x.tensor if hasattr(x, 'tensor') else x])[0])
            gradient_penalty_real = tf.reduce_sum(gradient_penalty_real, tf.range(-len(config.x_shape), 0))
            gradient_penalty_real = tf.pow(gradient_penalty_real, config.gradient_penalty_index / 2.0)

            gradient_penalty_fake = tf.square(
                tf.gradients(energy_fake, [x_.tensor if hasattr(x_, 'tensor') else x_])[0])
            gradient_penalty_fake = tf.reduce_sum(gradient_penalty_fake, tf.range(-len(config.x_shape), 0))
            gradient_penalty_fake = tf.pow(gradient_penalty_fake, config.gradient_penalty_index / 2.0)

            gradient_penalty = (tf.reduce_mean(gradient_penalty_fake) + tf.reduce_mean(gradient_penalty_real)) \
                               * config.gradient_penalty_weight / 2.0

        if config.use_gan:
            energy_fake = tf.reshape(energy_fake, (-1,))
            energy_real = tf.reshape(energy_real, (-1,))
            adv_G_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(energy_fake),
                logits=energy_fake)
            adv_G_loss = tf.reduce_mean(adv_G_loss)
            adv_D_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(energy_real),
                logits=energy_real)
            adv_D_real = tf.reduce_mean(adv_D_loss)
            adv_D_loss = adv_D_real - adv_G_loss
        else:
            adv_D_real = tf.reduce_mean(energy_real)
            adv_G_loss = tf.reduce_mean(energy_fake)
            adv_D_loss = adv_D_real - adv_G_loss + gradient_penalty
    return adv_D_loss, adv_G_loss, adv_D_real


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
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    (svhn_train, svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)

    def normalize(x):
        return [(x - 127.5) / 256.0 * 2]

    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    input_y = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_y')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_pn_omega = p_net(n_z=config.train_n_pz)
        D_loss, G_loss, D_real = get_all_loss(input_y, input_x)
        train_D_loss, train_G_loss, train_D_real = get_all_loss(input_x, train_pn_omega['x'].distribution.mean)
        D_loss += tf.losses.get_regularization_loss()
        G_loss += tf.losses.get_regularization_loss()
        train_D_loss += tf.losses.get_regularization_loss()
        train_G_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        ele_test_energy = D_psi(input_x)
        ele_gradient = tf.square(tf.gradients(ele_test_energy, [input_x])[0])
        ele_gradient_norm = tf.sqrt(tf.reduce_sum(ele_gradient, tf.range(-len(config.x_shape), 0)))
        print(ele_gradient_norm)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        D_params = tf.trainable_variables('D_psi')
        G_params = tf.trainable_variables('G_omega')
        print("========D_params=========")
        print(D_params)
        print("========G_params=========")
        print(G_params)
        with tf.variable_scope('D_optimizer'):
            D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            D_grads = D_optimizer.compute_gradients(D_loss, D_params)
            train_D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            train_D_grads = train_D_optimizer.compute_gradients(train_D_loss, D_params)
        with tf.variable_scope('G_optimizer'):
            train_G_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            train_G_grads = train_G_optimizer.compute_gradients(train_G_loss, G_params)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_G_train_op = train_G_optimizer.apply_gradients(train_G_grads)
            D_train_op = D_optimizer.apply_gradients(D_grads)
            train_D_train_op = train_D_optimizer.apply_gradients(train_D_grads)

        # derive the plotting function
        with tf.name_scope('plotting'):
            sample_n_z = config.sample_n_z
            plot_net = p_net(n_z=sample_n_z)
            x_plots = 256.0 * tf.reshape(
                plot_net['x'].distribution.mean, (-1,) + config.x_shape) / 2 + 127.5
            x_plots = tf.clip_by_value(x_plots, 0, 255)

        def plot_samples(loop, extra_index=None):
            if extra_index is None:
                extra_index = loop.epoch
            with loop.timeit('plot_time'):
                # plot reconstructs
                # plot samples
                images = session.run(x_plots)

                try:
                    save_images_collection(
                        images=np.round(images),
                        filename='plotting/sample/{}.png'.format(extra_index),
                        grid_size=(10, 10),
                        results=results,
                    )
                except Exception as e:
                    print(e)

                return images

    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size).map(normalize)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size).map(normalize)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size).map(normalize)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size).map(normalize)

    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True,
                                     skip_incomplete=True).map(normalize)
    mixed_array = get_mixed_array(config, x_train, x_test, svhn_train, svhn_test)
    mixed_test_flow = spt.DataFlow.arrays([mixed_array], config.batch_size,
                                          shuffle=True, skip_incomplete=True).map(normalize)

    with spt.utils.create_session().as_default() as session, train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/de/d5/02732c28dc8d103d22f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/26/e5/02c52d867e43103d22f5',
            'fashion_mnist': '/mnt/mfs/mlstorage-experiments/cwx17/36/e5/02c52d867e43103d22f5',
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/46/e5/02c52d867e43103d22f5',
            'kmnist': '/mnt/mfs/mlstorage-experiments/cwx17/fe/d5/02732c28dc8df9c132f5',
            'omniglot': '/mnt/mfs/mlstorage-experiments/cwx17/b6/e5/02279d802d3a19a232f5',
            'not_mnist': '/mnt/mfs/mlstorage-experiments/cwx17/c6/e5/02279d802d3aede332f5',
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/86/e5/02c52d867e43e2e432f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/96/e5/02c52d867e43a22632f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/a6/e5/02c52d867e431f3732f5',
            'mnist': '/mnt/mfs/mlstorage-experiments/cwx17/d6/e5/02279d802d3a485932f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/b6/e5/02c52d867e4371ab32f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/80/f5/02279d802d3adcc9d2f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/60/e5/02812baa4f70dcc9d2f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/90/f5/02279d802d3adcc9d2f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/a0/f5/02279d802d3adcc9d2f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/b0/f5/02279d802d3adcc9d2f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/e4/e5/02732c28dc8d095ad2f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/c0/f5/02279d802d3a5a5ad2f5',
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
                           var_groups=['q_net', 'p_net', 'posterior_flow', 'G_theta', 'D_psi'],
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

            n_critical = config.n_critical
            # adversarial training
            for epoch in epoch_iterator:

                if epoch == config.warm_up_start + 1:
                    make_diagram(loop,
                                 ele_gradient_norm,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='origin_gradient_norm_histogram'
                                 )
                    make_diagram(loop,
                                 ele_test_energy,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='origin_log_prob_histogram'
                                 )
                    if not config.pretrain:
                        session.run(tf.global_variables_initializer())

                if epoch == config.max_epoch + 1:
                    make_diagram(loop,
                                 ele_gradient_norm,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='gradient_norm_histogram'
                                 )

                    make_diagram(loop,
                                 ele_test_energy,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='log_prob_histogram'
                                 )
                    loop.print_logs()
                    break

                if epoch < config.warm_up_start:
                    step_iterator = MyIterator(train_flow)
                    while step_iterator.has_next:
                        for step, [x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            # training generator
                            [_, batch_D_loss, batch_D_real] = session.run(
                                [train_D_train_op, train_D_loss, train_D_real], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)

                        # training discriminator
                        [_, batch_G_loss] = session.run(
                            [train_G_train_op, train_G_loss], feed_dict={
                                input_x: x
                            })
                        loop.collect_metrics(G_loss=batch_G_loss)

                else:
                    for step, [x] in loop.iter_steps(train_flow):
                        for [y] in mixed_test_flow:
                            if config.distill_ratio != 1.0 and epoch > config.distill_epoch and config.use_transductive:
                                batch_energy = session.run(ele_test_energy, feed_dict={
                                    input_x: y
                                })
                                batch_index = np.argsort(batch_energy)
                                batch_index = batch_index[:int(len(batch_index) * config.distill_ratio)]
                                y = y[batch_index]
                                x_index = np.arange(len(x))
                                np.random.shuffle(x_index)
                                x_index = x_index[:len(batch_index)]
                                x = x[x_index]

                            # spec-training discriminator
                            [_, batch_D_loss, batch_G_loss, batch_D_real] = session.run(
                                [D_train_op, D_loss, G_loss, D_real], feed_dict={
                                    input_x: x, input_y: y
                                })
                            loop.collect_metrics(G_loss=batch_G_loss)
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)
                            break

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
