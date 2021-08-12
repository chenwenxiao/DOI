# -*- coding: utf-8 -*-

from matplotlib import pyplot
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import tensorflow as tf


def get_ele(op, flow, inputs, default_feed_dict={}):
    if not isinstance(inputs, list):
        inputs = [inputs]
    packs = []
    session = tf.get_default_session()
    for batch_x in flow:
        feed_dict = default_feed_dict
        for i in range(len(inputs)):
            feed_dict[inputs[i]] = batch_x[i]
        pack = session.run(
            op, feed_dict=feed_dict)  # [batch_size]
        pack = np.asarray(pack)
        # print(pack.shape)
        packs.append(pack)
    packs = np.concatenate(packs, axis=0)  # [len_of_flow]
    print(packs.shape, np.mean(packs), np.std(packs))
    return packs


def get_ele_torch(fn, flow):
    packs = []
    for batch_x in flow:
        pack = fn(*batch_x)
        pack = np.asarray(pack)
        # print(pack.shape)
        packs.append(pack)
    packs = np.concatenate(packs, axis=0)  # [len_of_flow]
    print(packs.shape, np.mean(packs), np.std(packs))
    return packs


def draw_curve(cifar_test, svhn_test, fig_name):
    cifar_test = cifar_test[~np.isnan(cifar_test)]
    cifar_test = cifar_test[~np.isinf(cifar_test)]
    svhn_test = svhn_test[~np.isnan(svhn_test)]
    svhn_test = svhn_test[~np.isinf(svhn_test)]
    label = np.concatenate(([1] * len(cifar_test), [-1] * len(svhn_test)))
    score = np.concatenate((cifar_test, svhn_test))

    fpr, tpr, thresholds = roc_curve(label, score)
    precision, recall, thresholds = precision_recall_curve(label, score)
    pyplot.plot(recall, precision)
    pyplot.plot(fpr, tpr)
    out_precision, out_recall, out_thresholds = precision_recall_curve(-label, -score)
    print('%s auc: %4f, in_aupr: %4f, out_aupr: %4f, ap: %4f, FPR@TPR95: %4f' % (
        fig_name, auc(fpr, tpr), auc(recall, precision), auc(out_recall, out_precision),
        average_precision_score(label, score),
        np.min(fpr[tpr > 0.95])))
    return auc(fpr, tpr)


def draw_metric(metric, color, label, fig_name):
    np.save(fig_name + label, metric)
    metric = metric[~np.isnan(metric)]
    metric = metric[~np.isinf(metric)]
    mean, std = np.mean(metric), np.std(metric)
    print('{} done. Value is {:.4f}±{:.4f}'.format(label, mean, std))
    metric[metric > mean + std * 10] = mean + std * 10
    metric[metric < mean - std * 10] = mean - std * 10
    metric = list(metric)
    n, bins, patches = pyplot.hist(metric, 40, normed=True, facecolor=color, alpha=0.4, label=label)

    index = []
    for i in range(len(bins) - 1):
        index.append((bins[i] + bins[i + 1]) / 2)

    def smooth(c, N=5):
        weights = np.hanning(N)
        return np.convolve(weights / weights.sum(), c)[N - 1:-N + 1]

    n[2:-2] = smooth(n)
    pyplot.plot(index, n, color=color)
    pyplot.legend()


def plot_fig(data_list, color_list, label_list, x_label, fig_name, auc_pair=(-2, -1)):
    tmp = 0.0
    try:
        pyplot.cla()
        pyplot.plot()
        pyplot.grid(c='silver', ls='--')
        pyplot.xlabel(x_label)
        spines = pyplot.gca().spines
        for sp in spines:
            spines[sp].set_color('silver')

        for i in range(len(data_list)):
            draw_metric(data_list[i], color_list[i], label_list[i], fig_name)
        pyplot.savefig('plotting/%s.jpg' % fig_name)

        pyplot.cla()
        pyplot.plot()
        tmp = draw_curve(data_list[auc_pair[0]], data_list[auc_pair[1]], fig_name)
        pyplot.savefig('plotting/%s_curve.jpg' % fig_name)
    except Exception as e:
        print(e)
    return tmp


def make_diagram(loop, op, flows, input_x, colors=['red', 'green', 'salmon', 'lightgreen'],
                 names=['CIFAR-10 Train', 'SVHN Train', 'CIFAR-10 Test', 'SVHN Test'],
                 x_label='log(bit/dims)', fig_name='log_pro_histogram', addtion_data=None, *args, **kwargs):
    try:
        packs = [get_ele(op, flow, input_x, *args, **kwargs) for flow in flows]
        if addtion_data is not None:
            if len(packs) == len(addtion_data):
                packs = [packs[i] + addtion_data[i] for i in range(len(packs))]
        _dict = {}
        _dict[fig_name] = plot_fig(packs, colors, names, x_label, fig_name)
        loop.collect_metrics(_dict)
        return packs
    except Exception as e:
        print(e)


def make_diagram_torch(loop, fn, flows, colors=['red', 'green', 'salmon', 'lightgreen'],
                       names=['CIFAR-10 Train', 'SVHN Train', 'CIFAR-10 Test', 'SVHN Test'],
                       x_label='log(bit/dims)', fig_name='log_pro_histogram', addtion_data=None):
    try:
        packs = [get_ele_torch(fn, flow) for flow in flows]
        if addtion_data is not None:
            if len(packs) == len(addtion_data):
                packs = [packs[i] + addtion_data[i] for i in range(len(packs))]
        _dict = {}
        _dict[fig_name] = plot_fig(packs, colors, names, x_label, fig_name)
        loop.add_metrics(_dict)
        return packs
    except Exception as e:
        print(e)


if __name__ == '__main__':
    pass
