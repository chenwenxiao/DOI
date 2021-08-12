# -*- encoding: utf-8 -*-
from enum import Enum
from typing import *

import mltk
import numpy as np
import tensorkit as tk
from tensorkit import tensor as T, examples
from torch import autograd
from torchvision.models import ResNet

from flow_next.models.glow import Glow
from utils.data import *
from utils.evaluation import dequantized_bpd
import torch

__all__ = ['TrainConfig', 'train_model']


class OptimizerType(str, Enum):
    ADAM = 'adam'
    ADAMAX = 'adamax'


class TrainConfig(mltk.Config):
    # initialization parameters
    init_batch_size: int

    # train parameters
    l2_reg: float = 0.

    max_epoch: int
    batch_size: int

    optimizer: OptimizerType = OptimizerType.ADAM
    lr: float = 0.001
    warmup_epochs: Optional[int] = 10
    grad_global_clip_norm: Optional[float] = 100.0

    # evaluation parameters
    test_epoch_freq: int
    test_batch_size: int
    ensemble_epsilon = 0.02

    # plot parameters
    plot_epoch_freq: int = 1
    plot_temperatures: List[float] = [0., .25, .5, .6, .7, .8, .9, 1.]

    # whether or not to print debug information?
    debug: bool = False


class TrainLRScheduler(tk.optim.lr_scheduler.LRScheduler):
    _last_set_lr: Optional[float] = None
    lr: float
    warmup_epochs: Optional[int]

    def __init__(self,
                 optimizer: tk.optim.Optimizer,
                 lr: float,
                 warmup_epochs: Optional[int]):
        super().__init__(optimizer)
        self.lr = lr
        self.warmup_epochs = warmup_epochs

    def update_lr(self):
        lr = self.lr
        if self.warmup_epochs is not None:
            if self.loop.epoch <= self.warmup_epochs:
                if self.loop.max_batch is None:
                    discount = self.loop.epoch * 1. / self.warmup_epochs
                else:
                    discount = (
                        (self.loop.epoch - 1. +
                         self.loop.batch * 1. / self.loop.max_batch) /
                        self.warmup_epochs
                    )
                lr = self.lr * discount
        if lr != self._last_set_lr:
            self.optimizer.set_lr(lr)
            self.loop.add_metrics(lr=lr)
            self._last_set_lr = lr

    def _bind_events(self, loop: mltk.TrainLoop):
        loop.on_batch_begin.do(self.update_lr)

    def _unbind_events(self, loop: mltk.TrainLoop):
        loop.on_batch_begin.cancel_do(self.update_lr)


def build_optimizer(config: TrainConfig,
                    params: Iterable[T.Variable]) -> tk.optim.Optimizer:
    if config.optimizer == OptimizerType.ADAM:
        return tk.optim.Adam(params, lr=config.lr)
    elif config.optimizer == OptimizerType.ADAMAX:
        return tk.optim.Adamax(params, lr=config.lr)
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer}')


def train_model(exp: mltk.Experiment,
                model: Glow,
                train_dataset: DataSet,
                test_dataset: DataSet,
                data_generator=None):
    train_config: TrainConfig = exp.config.train

    # print information of the data
    print('Train dataset information\n'
          '*************************')
    print_dataset_info(train_dataset)
    print('')

    print('Test dataset information\n'
          '*************************')
    if test_dataset is not None:
        print_dataset_info(test_dataset)
    print('')

    # prepare for the data streams
    train_stream = train_dataset.get_stream(
        'train', ['x'], batch_size=train_config.batch_size,
        shuffle=True, skip_incomplete=True,
    )
    if test_dataset is not None:
        test_stream = test_dataset.get_stream(
            'test', ['x'], batch_size=train_config.test_batch_size).to_arrays_stream()

    # print experiment and data information
    # examples.utils.print_experiment_summary(
    #     exp, train_data=train_stream, test_data=test_stream)

    # make tensor streams
    train_stream = tk.utils.as_tensor_stream(train_stream, prefetch=3)
    if test_dataset is not None:
        test_stream = tk.utils.as_tensor_stream(test_stream, prefetch=3)

    # inspect the model
    params, param_names = examples.utils.get_params_and_names(model)
    examples.utils.print_parameters_summary(params, param_names)
    print('')

    # initialize the network with a few portion of the train data
    [init_x] = train_dataset.sample('train', ['x'], train_config.init_batch_size)

    images = image_array_to_rgb(init_x, train_dataset.slots['x'])
    images = make_images_grid(images, n_cols=10)
    save_image_to_file(images,
                       exp.make_parent(f'plotting/samples/initialization.jpg'))  # save the initialization samples

    with T.no_grad():
        [init_ll, init_outputs] = model.initialize(init_x)
    init_mean, init_var = map(T.to_numpy, T.calculate_mean_and_var(init_ll))
    init_std = np.sqrt(init_var)
    mltk.print_with_time(f'Network initialized, ll: {init_mean:.6g} ± {init_std:.6g}')

    # define the train and evaluate functions
    def get_batch_output(x):
        ll, outputs = model(x)
        ll = T.reduce_mean(ll)
        bpd = dequantized_bpd(ll, train_dataset.slots['x'])
        ret = {'ll': ll, 'bpd': bpd}
        # for i, output in enumerate(outputs):
        #     ret[f'z_ll{i}'] = output.right_log_prob
        #     if train_config.debug:
        #         z_mu, z_var = T.calculate_mean_and_var(output.right)
        #         ret[f'z_mu{i}'] = z_mu
        #         ret[f'z_var{i}'] = z_var
        return ret

    def train_step(x):
        outputs = get_batch_output(x)
        loss = -outputs['ll']
        if train_config.l2_reg > 0.:
            outputs['l2_term'] = train_config.l2_reg * T.nn.l2_regularization(params)
            loss = loss + outputs['l2_term']
        # if train_config.z_mean_var_reg:
        #     # the square loss s too weak
        #     # outputs['z_topo_term'] = train_config.z_mean_var_reg * (
        #     #     outputs['z_mu'] ** 2 + (outputs['z_var'] - 1.) ** 2)
        #     topo_term = outputs['z_mu'] ** 2 + T.log(outputs['z_var']) ** 2
        #     outputs['z_topo_term'] = (train_config.z_mean_var_reg *
        #                               topo_term * np.prod(dataset.slots['x'].shape))
        #     loss = loss + outputs['z_topo_term']
        outputs['loss'] = loss
        return outputs

    def eval_step(x):
        with tk.layers.scoped_eval_mode(model), T.no_grad():
            return get_batch_output(x)

    def plot_samples(epoch=None):
        epoch = epoch or loop.epoch

        def f(n_samples, temperatures, suffix):
            with tk.layers.scoped_eval_mode(model), T.no_grad():
                images = model.sample(n_samples, temperatures=temperatures)
            images = T.to_numpy(images)
            images = image_array_to_rgb(images, train_dataset.slots['x'])
            images = make_images_grid(images, n_cols=10)
            save_image_to_file(images, exp.make_parent(f'plotting/samples/{epoch}{suffix}.jpg'))

        f(100, None, '')
        f(10, train_config.plot_temperatures, '_t')

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=train_config.max_epoch)
    loop.add_callback(mltk.callbacks.StopOnNaN())
    optimizer = build_optimizer(train_config, tk.layers.iter_parameters(model))
    lr_scheduler = TrainLRScheduler(
        optimizer=optimizer,
        lr=train_config.lr,
        warmup_epochs=train_config.warmup_epochs,
        # ratio=train_config.lr_anneal_ratio,
        # epochs=train_config.lr_anneal_epochs
    )
    lr_scheduler.bind(loop)
    if test_dataset is not None:
        loop.run_after_every(
            lambda: loop.test().run(eval_step, test_stream),
            epochs=train_config.test_epoch_freq
        )
    # loop.run_after_every(plot_samples, epochs=train_config.plot_epoch_freq)

    # train the model
    tk.layers.set_train_mode(model, True)
    examples.utils.fit_model(
        loop=loop,
        optimizer=optimizer,
        fn=train_step,
        stream=train_stream if data_generator is None else data_generator,
        global_clip_norm=train_config.grad_global_clip_norm,
    )

    # do the final test
    # plot_samples('final')
    if test_dataset is not None:
        results = mltk.TestLoop().run(eval_step, test_stream)
        print('')
        print(mltk.format_key_values(results, title='Results'))


def train_classifier(exp: mltk.Experiment,
                     model: ResNet,
                     train_dataset: DataSet,
                     test_dataset: DataSet,
                     data_generator=None):
    train_config: TrainConfig = exp.config.classifier_train

    # print information of the data
    print('Train dataset information\n'
          '*************************')
    print_dataset_info(train_dataset)
    print('')

    print('Test dataset information\n'
          '*************************')
    if test_dataset is not None:
        print_dataset_info(test_dataset)
        print('')

    # prepare for the data streams
    train_stream = train_dataset.get_stream(
        'train', ['x', 'y'], batch_size=train_config.batch_size,
        shuffle=True, skip_incomplete=True,
    )
    if test_dataset is not None:
        test_stream = test_dataset.get_stream(
            'test', ['x', 'y'], batch_size=train_config.test_batch_size).to_arrays_stream()

    # print experiment and data information
    examples.utils.print_experiment_summary(
        exp, train_data=train_stream, test_data=test_stream)

    # make tensor streams
    train_stream = tk.utils.as_tensor_stream(train_stream, prefetch=3)

    if test_dataset is not None:
        test_stream = tk.utils.as_tensor_stream(test_stream, prefetch=3)

    # inspect the model
    params, param_names = examples.utils.get_params_and_names(model)
    examples.utils.print_parameters_summary(params, param_names)
    print('')

    # define the train and evaluate functions
    def get_batch_output(x, y):
        pred = model(x)
        criterion = torch.nn.CrossEntropyLoss()
        cross_entropy = criterion(pred + 1e-8, y.long())
        ret = {'cross_entropy': cross_entropy}
        return ret

    def train_step(x, y):
        outputs = get_batch_output(x, y)
        loss = outputs['cross_entropy']
        if train_config.l2_reg > 0.:
            outputs['l2_term'] = train_config.l2_reg * T.nn.l2_regularization(params)
            loss = loss + outputs['l2_term']
        outputs['loss'] = loss
        return outputs

    def eval_step(x, y):
        with tk.layers.scoped_eval_mode(model), T.no_grad():
            return get_batch_output(x, y)

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=train_config.max_epoch)
    loop.add_callback(mltk.callbacks.StopOnNaN())
    optimizer = build_optimizer(train_config, tk.layers.iter_parameters(model))
    lr_scheduler = TrainLRScheduler(
        optimizer=optimizer,
        lr=train_config.lr,
        warmup_epochs=train_config.warmup_epochs,
        # ratio=train_config.lr_anneal_ratio,
        # epochs=train_config.lr_anneal_epochs
    )
    lr_scheduler.bind(loop)
    loop.run_after_every(
        lambda: loop.test().run(eval_step, test_stream),
        epochs=train_config.test_epoch_freq
    )

    # train the model
    tk.layers.set_train_mode(model, True)
    examples.utils.fit_model(
        loop=loop,
        optimizer=optimizer,
        fn=train_step,
        stream=train_stream if data_generator is None else data_generator,
        global_clip_norm=train_config.grad_global_clip_norm,
    )

    # do the final test
    if test_dataset is not None:
        results = mltk.TestLoop().run(eval_step, test_stream)
        print('')
        print(mltk.format_key_values(results, title='Results'))


def train_classifier_ensemble(exp: mltk.Experiment,
                              model: ResNet,
                              train_dataset: DataSet,
                              test_dataset: DataSet,
                              data_generator=None):
    train_config: TrainConfig = exp.config.classifier_train

    # print information of the data
    print('Train dataset information\n'
          '*************************')
    print_dataset_info(train_dataset)
    print('')

    print('Test dataset information\n'
          '*************************')
    if test_dataset is not None:
        print_dataset_info(test_dataset)
        print('')

    # prepare for the data streams
    train_stream = train_dataset.get_stream(
        'train', ['x', 'y'], batch_size=train_config.batch_size,
        shuffle=True, skip_incomplete=True,
    )
    if test_dataset is not None:
        test_stream = test_dataset.get_stream(
            'test', ['x', 'y'], batch_size=train_config.test_batch_size).to_arrays_stream()

    # print experiment and data information
    examples.utils.print_experiment_summary(
        exp, train_data=train_stream, test_data=test_stream)

    # make tensor streams
    train_stream = tk.utils.as_tensor_stream(train_stream, prefetch=3)

    if test_dataset is not None:
        test_stream = tk.utils.as_tensor_stream(test_stream, prefetch=3)

    # inspect the model
    params, param_names = examples.utils.get_params_and_names(model)
    examples.utils.print_parameters_summary(params, param_names)
    print('')

    # define the train and evaluate functions
    def get_batch_output(x, y):
        x.requires_grad = True
        pred = model(x)
        criterion = torch.nn.CrossEntropyLoss()
        cross_entropy = criterion(pred, y.long())

        gradients = autograd.grad(cross_entropy, x, grad_outputs=torch.ones(cross_entropy.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]
        sign = torch.sign(gradients)
        x_hat = x + train_config.ensemble_epsilon * sign

        pred_hat = model(x_hat)
        cross_entropy_hat = criterion(pred_hat, y.long())

        ret = {'cross_entropy': cross_entropy + cross_entropy_hat}
        return ret

    def train_step(x, y):
        outputs = get_batch_output(x, y)
        loss = outputs['cross_entropy']
        if train_config.l2_reg > 0.:
            outputs['l2_term'] = train_config.l2_reg * T.nn.l2_regularization(params)
            loss = loss + outputs['l2_term']
        outputs['loss'] = loss
        return outputs

    def eval_step(x, y):
        with tk.layers.scoped_eval_mode(model):
            return get_batch_output(x, y)

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=train_config.max_epoch)
    loop.add_callback(mltk.callbacks.StopOnNaN())
    optimizer = build_optimizer(train_config, tk.layers.iter_parameters(model))
    lr_scheduler = TrainLRScheduler(
        optimizer=optimizer,
        lr=train_config.lr,
        warmup_epochs=train_config.warmup_epochs,
        # ratio=train_config.lr_anneal_ratio,
        # epochs=train_config.lr_anneal_epochs
    )
    lr_scheduler.bind(loop)
    loop.run_after_every(
        lambda: loop.test().run(eval_step, test_stream),
        epochs=train_config.test_epoch_freq
    )

    # train the model
    tk.layers.set_train_mode(model, True)
    examples.utils.fit_model(
        loop=loop,
        optimizer=optimizer,
        fn=train_step,
        stream=train_stream if data_generator is None else data_generator,
        global_clip_norm=train_config.grad_global_clip_norm,
    )

    # do the final test
    if test_dataset is not None:
        results = mltk.TestLoop().run(eval_step, test_stream)
        print('')
        print(mltk.format_key_values(results, title='Results'))
