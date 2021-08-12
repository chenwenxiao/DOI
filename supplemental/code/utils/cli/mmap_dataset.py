"""
Usage::

    python -m utils.cli.mmap_dataset sample-images -s train -S x -n 100 -i ~/Downloads/cifar10
    python -m utils.cli.mmap_dataset make-celebA -i ~/Downloads/CelebA/celebA -o ~/Downloads/CelebA/mmap

"""


import os

import click
import mltk

from utils.data import *
from utils.data.mappers import *


@click.group()
def main():
    """Populate MMap datasets."""


@main.command('sample-images')
@click.option('-s', '--split', type=str, required=True)
@click.option('-S', '--slot', type=str, required=True)
@click.option('-n', type=int, required=True, default=100)
@click.option('--with-replacement', is_flag=True, default=False)
@click.option('-o', '--output-file', required=False, default=None)
@click.option('--open', 'should_open', is_flag=True, default=False)
@click.option('-i', '--input-dir', required=True)
def sample_images(split, slot, n, with_replacement, output_file, should_open,
                  input_dir):
    dataset = MMapDataSet(input_dir)
    print_dataset_info(dataset)

    slots = [slot]
    if slot != 'id' and 'id' in dataset.slots:
        slots.append('id')
    arrays = dataset.sample(split, slots, n, with_replacement)

    for slot, arr in zip(slots, arrays):
        if slot == 'id':
            print(format_labels_grid(arr))

    if slots[0] != 'id':
        images = image_array_to_rgb(arrays[0], info=dataset.slots[slots[0]])
        grid = make_images_grid(images)
        saved_file = save_image_to_file(grid, output_file=output_file)
        if output_file is None or should_open:
            click.launch(saved_file)


def make_colored_images_mmap_dataset_command(name,
                                             dataset_cls,
                                             default_bit_depth: int = 8,
                                             require_input_dir: bool = False):
    @click.option('--force', '-F', is_flag=True, default=False)
    @click.option('--flatten', is_flag=True, default=False)
    @click.option('--channel-first', is_flag=True, default=False)
    @click.option('--dequantize', is_flag=True, default=False)
    @click.option('--bit-depth', type=int, default=default_bit_depth)
    @click.option('--val-split', type=float, default=None, required=False)
    @click.option('--min-val', type=float, default=None, required=False)
    @click.option('--max-val', type=float, default=None, required=False)
    @click.option('-o', '--output-dir', type=str, required=True)
    def make_dataset(force, flatten, channel_first, dequantize, bit_depth,
                     val_split, min_val, max_val, output_dir, input_dir=None):
        if require_input_dir and input_dir is None:
            raise ValueError('`--input-dir` is required.')
        if os.path.exists(output_dir) and not force:
            raise IOError(f'Output directory already exists: {output_dir}')
        if (min_val is None) != (max_val is None):
            raise ValueError(f'`min_val` and `max_val` should be both specified '
                             f'or neither specified.')

        dataset_args = []
        if require_input_dir:
            dataset_args.append(input_dir)
        dataset = dataset_cls(*dataset_args, val_split=val_split)

        mappers = []
        if channel_first:
            mappers.append(ChannelLastToFirst())
        if flatten:
            mappers.append(Flatten())
        if bit_depth != 8:
            mappers.append(ReduceToBitDepth(bit_depth))
        if dequantize:
            mappers.append(Dequantize())
        if min_val is not None:
            mappers.append(ScaleToRange(min_val, max_val))

        dataset = dataset.apply_mappers(x=mappers)
        MMapDataSet.populate_dir(output_dir, dataset)

    if require_input_dir:
        make_dataset = click.option('-i', '--input-dir', type=str, required=True)(make_dataset)

    make_dataset = main.command(f'make-{name}')(make_dataset)
    return make_dataset


make_colored_images_mmap_dataset_command('cifar10', Cifar10)
make_colored_images_mmap_dataset_command('cifar100', Cifar100)
make_colored_images_mmap_dataset_command('celebA', CelebA, require_input_dir=True)


if __name__ == '__main__':
    main()
