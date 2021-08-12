# -*- coding: utf-8 -*-
import mltk
from sklearn.covariance import EmpiricalCovariance
from mltk.data import ArraysDataStream, DataStream
import tensorkit as tk
from tensorkit import tensor as T
import sys
import torch
import numpy as np

from flow_next.common import TrainConfig, DataSetConfig, make_dataset, train_model, ImageAugmentationMapper, get_mapper
from flow_next.common.train_utils import train_classifier
from flow_next.models.glow import GlowConfig, Glow
from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import plot_fig, make_diagram_torch, get_ele_torch

from utils.evaluation import dequantized_bpd
import torch.autograd as autograd
import torchvision.models as models
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
    class_num = 10
    features_nums = 5

    odin_T = 1000
    odin_epsilon = 0.0012 * 2  # multiple 2 for the normalization [-1, 1] instead of [0, 1] in ODIN

    classifier_train = TrainConfig(
        optimizer='adamax',
        init_batch_size=128,
        batch_size=64,
        test_batch_size=64,
        test_epoch_freq=10,
        max_epoch=200,
        grad_global_clip_norm=None,
        # grad_global_clip_norm=100.0,
        debug=True
    )
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
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/0a/66/02c52d867e43a2d825f5',
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/5a/66/02c52d867e43b46925f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/d1/e5/02279d802d3a5a9d51f5',
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/50/e5/02c52d867e437d2851f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/c1/e5/02279d802d3ab6c751f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/f4/e5/02279d802d3a98e102f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/b4/e5/02279d802d3aa7d002f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/1a/66/02c52d867e43a2d825f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/3c/d5/02732c28dc8db6c751f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/d9/d5/02812baa4f70b68951f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/e1/e5/02279d802d3ad38f51f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/60/e5/02c52d867e4360ae51f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/a2/f5/02279d802d3aa24a03f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/92/f5/02279d802d3a3c3a03f5',
        }
        print(experiment_dict)
        if config.in_dataset.name in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset.name]
        else:
            restore_dir = None
        print('restore model from {}'.format(restore_dir))

        # load the dataset
        cifar_train_dataset, cifar_test_dataset, cifar_dataset = make_dataset(config.in_dataset)
        print('CIFAR DataSet loaded.')
        svhn_train_dataset, svhn_test_dataset, svhn_dataset = make_dataset(config.out_dataset)
        print('SVHN DataSet loaded.')

        cifar_train_flow = cifar_test_dataset.get_stream('train', 'x', config.batch_size)
        cifar_test_flow = cifar_test_dataset.get_stream('test', 'x', config.batch_size)
        svhn_train_flow = svhn_test_dataset.get_stream('train', 'x', config.batch_size)
        svhn_test_flow = svhn_test_dataset.get_stream('test', 'x', config.batch_size)

        x_train = cifar_dataset.get_array('train', 'x')
        y_train = cifar_dataset.get_array('train', 'y')
        x_test = cifar_dataset.get_array('test', 'x')
        y_test = cifar_dataset.get_array('test', 'y')
        svhn_train = svhn_dataset.get_array('train', 'x')
        svhn_test = svhn_dataset.get_array('test', 'x')
        config.class_num = np.max(y_train) + 1

        if restore_dir is None:
            classifier = models.resnet34(num_classes=config.class_num).cuda()
            train_classifier(exp, classifier, cifar_test_dataset, cifar_test_dataset)

            @torch.no_grad()
            def eval_predict(x):
                x = T.from_numpy(x)
                predict = classifier(x)
                predict = T.argmax(predict, axis=-1)
                return T.to_numpy(predict)

            cifar_test_predict = get_ele_torch(eval_predict, cifar_test_flow)
            print('Correct number in cifar test is {}'.format(
                np.sum(cifar_test_predict == y_test)))
            cifar_train_predict = get_ele_torch(eval_predict, cifar_train_flow)
            print('Correct number in cifar train is {}'.format(
                np.sum(cifar_train_predict == y_train)))

            torch.save(classifier, 'classifier.pkl')

            for current_class in range(config.class_num):
                # construct the model
                model = Glow(cifar_train_dataset.slots['x'], exp.config.model)
                print('Model constructed.')
                current_class_stream = ArraysDataStream([x_train[y_train == current_class]], config.batch_size,
                                                        shuffle=True, skip_incomplete=True)
                mapper = get_mapper(config.in_dataset, training=True)
                mapper.fit(cifar_dataset.slots['x'])
                current_class_stream = current_class_stream.map(lambda x: mapper.transform(x))
                current_class_stream = tk.utils.as_tensor_stream(current_class_stream, prefetch=3)
                # train the model
                train_model(exp, model, cifar_train_dataset, cifar_test_dataset, current_class_stream)
                torch.save(model, 'model_{}.pkl'.format(current_class))
        else:
            classifier = torch.load(restore_dir + '/classifier.pkl')

        with mltk.TestLoop() as loop:

            @torch.no_grad()
            def eval_predict(x):
                x = T.from_numpy(x)
                predict = classifier(x)
                odin = T.reduce_max(torch.softmax(predict, dim=-1), axis=[-1])
                print(T.reduce_mean(odin))
                predict = T.argmax(predict, axis=-1)
                return T.to_numpy(predict)

            cifar_test_predict = get_ele_torch(eval_predict, cifar_test_flow)
            svhn_test_predict = get_ele_torch(eval_predict, svhn_test_flow)

            @torch.no_grad()
            def eval_ll(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                bpd = -dequantized_bpd(ll, cifar_train_dataset.slots['x'])
                return T.to_numpy(bpd)

            final_cifar_test_ll = np.zeros(len(x_test))
            final_svhn_test_ll = np.zeros(len(svhn_test))
            for current_class in range(0, config.class_num):
                cifar_mask = cifar_test_predict == current_class
                svhn_mask = svhn_test_predict == current_class
                pse_epoch = config.warm_up_start + (current_class + 1) * config.test_epoch_freq

                if restore_dir is None:
                    model = torch.load('model_{}.pkl'.format(current_class))
                else:
                    model = torch.load(restore_dir + '/model_{}.pkl'.format(current_class))

                test_mapper = get_mapper(config.in_dataset, training=False)
                test_mapper.fit(cifar_dataset.slots['x'])
                if np.sum(cifar_mask) > 0:
                    cifar_test_ll = get_ele_torch(eval_ll, ArraysDataStream([
                        x_test[cifar_mask]
                    ], config.test_batch_size, False, False).map(lambda x: test_mapper.transform(x)))
                    final_cifar_test_ll[cifar_mask] = cifar_test_ll

                if np.sum(svhn_mask) > 0:
                    svhn_test_ll = get_ele_torch(eval_ll, ArraysDataStream([
                        svhn_test[svhn_mask]
                    ], config.test_batch_size, False, False).map(lambda x: test_mapper.transform(x)))
                    final_svhn_test_ll[svhn_mask] = svhn_test_ll

            loop.add_metrics(log_prob_histogram=plot_fig(
                [final_cifar_test_ll, final_svhn_test_ll],
                color_list=['red', 'green'],
                label_list=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                x_label='log(bit/dims)',
                fig_name='log_prob_histogram', auc_pair=(0, 1)
            ))


if __name__ == '__main__':
    main()
