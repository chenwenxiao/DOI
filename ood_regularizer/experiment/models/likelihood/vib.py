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
from ood_regularizer.experiment.models.utils import get_mixed_array
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
    max_beta = 0.01

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
    distill_epoch = 5000
    mcmc_times = 5

    epsilon = -20.0
    min_logstd_of_q = -10.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    extra_stride = 2
    class_num = 10
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
    shape = (1,) + config.x_shape[:-1] + (5,)
    print(shape)
    extend_x = tf.get_variable(name='extend_x', shape=shape, dtype=tf.float32,
                               trainable=True)
    print(extend_x)
    batch_size = spt.utils.get_shape(x)[0]
    extend_x = tf.tile(extend_x, tf.concat([[batch_size], [1] * len(config.x_shape)], axis=0))
    print(extend_x)
    x = tf.concat([x, extend_x], axis=-1)
    print(x)

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
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 500)
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)
        logits = spt.layers.dense(h_z, config.class_num)

    y = net.add('y', spt.Categorical(logits=logits))
    return net


def get_all_loss(q_net, p_net, warm_up):
    with tf.name_scope('vib_loss'):
        train_recon = p_net['y'].log_prob()
        train_kl = tf.reduce_mean(
            -p_net['z'].log_prob() + q_net['z'].log_prob()
        )
        VAE_loss = -train_recon + warm_up * train_kl
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

    if config.count_experiment:
        with open('/home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/count_experiments', 'a') as f:
            f.write(results.system_path("") + '\n')
            f.close()

    # prepare for training and testing data
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    (svhn_train, svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)

    def normalize(x, y):
        return (x - 127.5) / 256.0 * 2, y

    config.class_num = np.max(y_train) + 1
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
        dtype=tf.int32, shape=(None,), name='input_y')
    warm_up = tf.placeholder(
        dtype=tf.float32, shape=(), name='warm_up')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_q_net = q_net(input_x, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z'], 'y': input_y},
                            n_z=config.train_n_qz)
        train_recon = train_p_net['y'].log_prob()
        VAE_loss = get_all_loss(train_q_net, train_p_net, warm_up)

        VAE_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_qz)
        print(test_q_net['z'])
        test_chain = test_q_net.chain(p_net, observed={'y': input_y}, n_z=config.test_n_qz, latent_axis=0)
        print(test_chain.model['y'].log_prob())
        ele_test_recon = tf.reduce_mean(test_chain.model['y'].log_prob(), axis=0) / config.x_shape_multiple / np.log(2)
        print(ele_test_recon)
        ele_test_entropy = []
        for i in range(config.class_num):
            fake_y = tf.ones_like(input_y, dtype=tf.int32) * i
            ele_test_entropy.append(
                tf.reduce_mean(tf.exp(test_chain.model['y'].distribution.log_prob(given=fake_y)), axis=0))
        ele_test_entropy = tf.stack(ele_test_entropy, axis=-1)  # [batch_size, class_num]
        print(ele_test_entropy)
        ele_test_predict = tf.argmax(ele_test_entropy, axis=-1)
        ele_test_predict_value = tf.reduce_max(ele_test_entropy, axis=-1)
        ele_test_entropy = tf.reduce_sum(-tf.log(ele_test_entropy) * ele_test_entropy, axis=-1)

        test_recon = tf.reduce_mean(
            ele_test_recon
        )
        ele_test_ll = test_chain.vi.evaluation.is_loglikelihood() / config.x_shape_multiple / np.log(2)
        print(ele_test_ll)
        test_nll = -tf.reduce_mean(
            ele_test_ll
        )
        ele_test_lb = test_chain.vi.lower_bound.elbo() / config.x_shape_multiple / np.log(2)
        print(ele_test_lb)
        test_lb = tf.reduce_mean(ele_test_lb)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('p_net')
        with tf.variable_scope('theta_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)

    cifar_train_flow = spt.DataFlow.arrays([x_train, y_train], config.test_batch_size).map(normalize)
    cifar_test_flow = spt.DataFlow.arrays([x_test, y_test], config.test_batch_size).map(normalize)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train, svhn_train_y], config.test_batch_size).map(normalize)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test, svhn_test_y], config.test_batch_size).map(normalize)

    train_flow = spt.DataFlow.arrays([x_train, y_train], config.batch_size, shuffle=True,
                                     skip_incomplete=True).map(normalize)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
            'kmnist': '/mnt/mfs/mlstorage-experiments/cwx17/a8/e5/02279d802d3ada3a62f5',
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/00/e5/02732c28dc8de8d962f5',
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/ff/d5/02732c28dc8d02d962f5',
            'not_mnist': '/mnt/mfs/mlstorage-experiments/cwx17/98/e5/02279d802d3a8f4962f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/ef/d5/02732c28dc8d2ad862f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/df/d5/02732c28dc8d87d862f5',
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/08/e5/02c52d867e431fc862f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/f7/e5/02c52d867e43403862f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/dd/d5/02812baa4f7014b762f5',
            'fashion_mnist': '/mnt/mfs/mlstorage-experiments/cwx17/cf/d5/02732c28dc8d14b762f5',
            'mnist': '/mnt/mfs/mlstorage-experiments/cwx17/cd/d5/02812baa4f7014b762f5',
            'omniglot': '/mnt/mfs/mlstorage-experiments/cwx17/78/e5/02279d802d3a14b762f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/d0/f5/02279d802d3a3fdad2f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/e0/f5/02279d802d3ab75cd2f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/01/f5/02279d802d3acecdd2f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/11/f5/02279d802d3a6b7ed2f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/21/f5/02279d802d3a7c7ed2f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/31/f5/02279d802d3a71fed2f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/d0/f5/02c52d867e4350bdf2f5',
        }
        print(experiment_dict)
        if config.in_dataset in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset] + '/checkpoint'
            restore_checkpoint = os.path.join(
                restore_dir, 'checkpoint',
                'checkpoint.dat-{}'.format(config.max_epoch))
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
                    cifar_train_predict = get_ele(ele_test_predict, cifar_train_flow, [input_x, input_y])
                    cifar_test_predict = get_ele(ele_test_predict, cifar_test_flow, [input_x, input_y])
                    get_ele(ele_test_predict_value, cifar_train_flow, [input_x, input_y])
                    get_ele(ele_test_predict_value, cifar_test_flow, [input_x, input_y])
                    print('Correct number in cifar test is {}'.format(
                        np.sum(cifar_test_predict == y_test)))
                    print('Correct number in cifar train is {}'.format(
                        np.sum(cifar_train_predict == y_train)))

                    make_diagram(loop,
                                 ele_test_entropy,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='H_histogram'
                                 )

                    make_diagram(loop,
                                 ele_test_lb - ele_test_recon,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='R_histogram'
                                 )

                    make_diagram(loop,
                                 -ele_test_entropy,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='nH_histogram'
                                 )

                    make_diagram(loop,
                                 -ele_test_lb + ele_test_recon,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='nR_histogram'
                                 )

                    make_diagram(loop,
                                 ele_test_recon,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='recon_histogram'
                                 )

                    make_diagram(loop,
                                 ele_test_lb,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='elbo_histogram'
                                 )

                    make_diagram(loop,
                                 ele_test_ll,
                                 [cifar_test_flow, svhn_test_flow],
                                 [input_x, input_y],
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='log_prob_histogram'
                                 )

                    loop.print_logs()
                    break

                for step, [x, y] in loop.iter_steps(train_flow):
                    _, batch_VAE_loss, batch_recon = session.run([VAE_train_op, VAE_loss, train_recon], feed_dict={
                        input_x: x, input_y: y, warm_up: max(1.0 * epoch / config.warm_up_start, 1.0) * config.max_beta
                    })
                    loop.collect_metrics(VAE_loss=batch_VAE_loss)
                    loop.collect_metrics(recon=batch_recon)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.max_epoch:
                    loop._checkpoint_saver.save(epoch)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
