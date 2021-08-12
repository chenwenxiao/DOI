# -*- encoding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import *

import click
import matplotlib.pyplot as plt
import pandas as pd
import requests


@dataclass
class MetricValue(object):
    __slots__ = ('mean', 'std')

    mean: Union[float, str]
    std: Optional[Union[float, str]]


@dataclass
class BatchLog(object):
    __slots__ = ('index', 'metrics')

    index: int
    metrics: Dict[str, MetricValue]


@dataclass
class EpochLog(object):
    __slots__ = ('index', 'batches', 'n_batches', 'metrics')

    index: int
    batches: List[BatchLog]
    metrics: Dict[str, MetricValue]
    n_batches: Optional[int]


@dataclass
class LoopLog(object):
    __slots__ = ('type', 'epochs', 'metrics',)

    type: str
    epochs: List[EpochLog]
    metrics: Dict[str, MetricValue]


def parse_log(log: str) -> List[LoopLog]:
    train_start_pattern = re.compile(r'.*Train started')
    epoch_start_pattern = re.compile(r'^Epoch (\d+)(?:/\d+)?$')
    batch_pattern = re.compile(r'^\s*(\d+)(?:/\d+)?(?: - eta [^-]+)?((?: - [^:]+: .*?)*)\s*$')
    metric_pattern = re.compile(r'\s*(?P<name>[^:]+): (?P<mean>[^( ]+)( \(±(?P<std>[^)]+)\))?\s*')
    epoch_end_pattern = re.compile(r'^(\d+) iters in (?:[^-]+)(?: - eta [^-]+)?((?: - [^:]+: .*?)*)\s*$')

    def parse_metrics(source, target):
        for s in source.split(' - '):
            m2 = metric_pattern.match(s)
            if m2:
                m2 = m2.groupdict()

                if 'std' in m2 and m2['std'] is not None:
                    mv = MetricValue(mean=m2['mean'], std=m2['std'])
                else:
                    mv = MetricValue(mean=m2['mean'], std=None)

                for attr in ('mean', 'std'):
                    try:
                        setattr(mv, attr, float(getattr(mv, attr)))
                    except Exception:
                        pass

                target[m2['name']] = mv

    loops = []
    loop: Optional[LoopLog] = None
    epoch: Optional[EpochLog] = None

    for line in log.split('\n'):
        line = line.rstrip()

        # match loop start
        if train_start_pattern.match(line):
            loop = LoopLog(type='train', epochs=[], metrics={})
            loops.append(loop)
            continue

        # skip this line if no active loop
        if loop is None:
            continue

        # if epoch start
        m = epoch_start_pattern.match(line)
        if m:
            epoch = EpochLog(index=int(m.group(1)), batches=[], n_batches=None,
                             metrics={})
            loop.epochs.append(epoch)

        # skip this line if no active epoch
        if epoch is None:
            continue

        # if batch log
        m = batch_pattern.match(line)
        if m:
            batch = BatchLog(index=int(m.group(1)), metrics={})
            parse_metrics(m.group(2), batch.metrics)
            epoch.batches.append(batch)

        # if epoch end log
        m = epoch_end_pattern.match(line)
        if m:
            epoch.n_batches = int(m.group(1))
            parse_metrics(m.group(2), epoch.metrics)
            epoch = None

    return loops


def parse_log_from_server(uri: str) -> List[LoopLog]:
    # parse the id of the input
    patterns = [
        re.compile(r'^(https?://.*?)/v1/_getfile/([a-z0-9]{24})/console\.log$'),
        re.compile(r'^(https?://.*?)/([a-z0-9]{24})(/(((reports|figures|console|browse)(/.*)?)|))?$')
    ]
    for pattern in patterns:
        m = pattern.match(uri)
        if m:
            uri = f'{m.group(1)}/v1/_getfile/{m.group(2)}/console.log'

    # download the logs
    logs = requests.get(uri).content.decode('utf-8')

    # parse the logs
    return parse_log(logs)


def plot_epochs(log: LoopLog, metrics: List[str]):
    metric_vals = {k: ([], [], []) for k in metrics}
    metric_has_std = {k: True for k in metrics}

    for epoch in log.epochs:
        for k in metrics:
            v = epoch.metrics.get(k)
            if v is not None:
                if v.std is None:
                    metric_has_std[k] = False
                metric_vals[k][0].append(epoch.index)
                metric_vals[k][1].append(v.mean)
                metric_vals[k][2].append(v.std)

    metric_means: Dict[str, pd.Series] = {
        k: pd.Series(index=v[0], data=v[1])
        for k, v in metric_vals.items()}
    metric_stds: Dict[str, pd.Series] = {
        k: pd.Series(index=v[0], data=v[2])
        for k, v in metric_vals.items()
        if metric_has_std[k]
    }

    fig = plt.figure(figsize=(8, 5))
    for k in metrics:
        seq = metric_means[k]
        seq.plot(label=k)
        if metric_has_std[k]:
            std = metric_stds[k]
            plt.fill_between(seq.index, seq - std, seq + std, alpha=0.2)

    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.close(fig)


@click.command()
@click.option('--loop-index', default=None, type=int)
@click.option('-m', '--metric', 'metrics', type=str, multiple=True, required=True)
@click.argument('uri', required=True)
def main(uri, loop_index, metrics):
    loop_logs = parse_log_from_server(uri)
    if not loop_logs:
        raise ValueError(f'Log is not recognized.')

    # print the information of this log
    for i, loop in enumerate(loop_logs, 1):
        n_batches = [e.n_batches for e in loop.epochs]
        if len(n_batches) > 1:
            n_batches = n_batches[:-1]
        avg_n_batches = sum(n_batches) / len(n_batches)
        if abs(int(avg_n_batches) - avg_n_batches) < 1e-5:
            avg_n_batches = str(int(avg_n_batches))
        else:
            avg_n_batches = f'{avg_n_batches:.2f}'

        epoch_metrics = set()
        batch_metrics = set()
        for epoch in loop.epochs:
            epoch_metrics.update(epoch.metrics.keys())
            for batch in epoch.batches:
                batch_metrics.update(batch.metrics.keys())

        print(f'Loop #{i}: {loop.type}, {len(loop.epochs)} epochs, '
              f'{avg_n_batches} batches per epoch.')
        print(f'  Epoch metrics: {sorted(epoch_metrics)}')
        print(f'  Batch metrics: {sorted(batch_metrics)}')

    # select the requested loop log
    if loop_index is None:
        if sum((int(loop.type == 'train') for loop in loop_logs)) != 1:
            raise ValueError(f'`loop_index` is required when there are multiple '
                             f'train loops.')
        for i, loop in enumerate(loop_logs):
            if loop.type == 'train':
                loop_index = i
                break

    log = loop_logs[loop_index]

    # plot the metrics
    plot_epochs(log, list(metrics))


if __name__ == '__main__':
    main()
