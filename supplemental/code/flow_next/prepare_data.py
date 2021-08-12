import time

import click
import numpy as np

from utils.data import *
from utils.data.mappers import *


def init_random(seed=None):
    if seed is None:
        seed = int(time.time())
    print(f'Random seed: {seed}')
    print('')
    np.random.seed(seed)


@click.group()
def main():
    """Prepare the datasets for Flow-Next experiments."""


@main.command('cifar10')
@click.option('-o', '--output-dir', required=True, type=str)
@click.option('-S', '--split', 'splits', required=False, default=None, multiple=True)
@click.option('-s', '--slot', 'slots', required=False, default=None, multiple=True)
@click.option('--seed', required=False, default=None, type=int)
@click.option('--dequantize', is_flag=True, required=False, default=False)
def cifar10(output_dir, splits, slots, seed, dequantize):
    init_random(seed)
    dataset = Cifar10()
    if dequantize:
        dataset = dataset.apply_mappers(x=Dequantize())
    print_dataset_info(dataset, splits, slots)
    MMapDataSet.populate_dir(output_dir, dataset, splits, slots, name='cifar10')


if __name__ == '__main__':
    main()
