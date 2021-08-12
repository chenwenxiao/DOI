"""
Read images from Glow .tfr files, and save the images to a directory.

Usage::

     PYTHONPATH=. python ./3rdparty/glow/tfr_to_mmap.py \
        --input-dir /Volumes/ipwx-data/DataSets/CelebA-Glow/origin \
        --output-dir /Volumes/ipwx-data/DataSets/CelebA-Glow/mmap
"""
import codecs
import os
import re

import click
import mltk
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from utils.data import *

tf.enable_eager_execution()


@click.command()
@click.option('--input-dir', type=str, required=True)
@click.option('--output-dir', type=str, required=True)
@click.option('--bit-depth', type=int, default=8)
@click.option('--val-split', type=float, default=None, required=False)
def main(input_dir, output_dir, bit_depth, val_split):
    slots = {}
    splits = {}

    for split in ['train', 'test']:
        src_split = {'train': 'train', 'test': 'validation'}[split]

        # get all tfr files
        split_src = os.path.join(input_dir, src_split)
        pattern = re.compile(rf'{src_split}-r{bit_depth:02d}-s-(\d+)-of-(\d+)\.tfrecords')
        total = None
        file_list = {}

        for name in os.listdir(split_src):
            m = pattern.match(name)
            if m:
                idx, m_total = int(m.group(1)), int(m.group(2))
                if total is None:
                    total = m_total
                elif total != m_total:
                    raise ValueError(f'`total` inconsistent.')

                if idx in file_list:
                    raise ValueError(f'Duplicated: {idx}')
                file_list[idx] = name

        if not file_list:
            raise IOError(f'Empty dir: {split_src}')
        if len(file_list) != total:
            raise ValueError(f'Some data file is missing.')

        file_list = [s for _, s in sorted(file_list.items())]

        # load the data
        # see: https://github.com/openai/glow/blob/master/data_loaders/get_data.py
        def inspect_tfrecord_tf(record):
            features = tf.parse_single_example(record, features={
                'shape': tf.FixedLenFeature([3], tf.int64),
                'label': tf.FixedLenFeature([1], tf.int64)})
            label, shape = features['label'], features['shape']
            label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
            return shape, label

        def parse_tfrecord_tf(record, res, rnd_crop):
            features = tf.parse_single_example(record, features={
                'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)})
            # label is always 0 if uncondtional
            # to get CelebA attr, add 'attr': tf.FixedLenFeature([40], tf.int64)
            data, shape = features['data'], features['shape']
            img = tf.decode_raw(data, tf.uint8)
            if rnd_crop:
                # For LSUN Realnvp only - random crop
                img = tf.reshape(img, shape)
                img = tf.random_crop(img, [res, res, 3])
            img = tf.reshape(img, [res, res, 3])
            return img,  # to get CelebA attr, also return attr

        files = tf.data.Dataset.list_files(
            os.path.join(split_src, f'{src_split}-r{bit_depth:02d}-s-*-of-*.tfrecords'),
            shuffle=False
        )
        dset = files.apply(tf.data.TFRecordDataset)

        # inspect the dataset
        dset2 = dset.map(inspect_tfrecord_tf)
        shape = None
        labels = []

        for shape0, label in tqdm(dset2, desc=f'Inspect {split}'):
            if shape is None:
                shape = tuple(shape0.numpy().tolist())
            elif shape != tuple(shape0.numpy().tolist()):
                raise ValueError(f'Shape inconsistent.')
            labels.append(label)

        labels = np.asarray(labels, dtype=np.int32)
        train_idx = np.arange(len(labels), dtype=np.int32)
        print(f'Discovered {len(labels)} images.')

        if split == 'train' and val_split is not None and val_split > 0:
            train_idx, val_idx = mltk.utils.split_numpy_array(train_idx, portion=val_split)
            val_idx = np.asarray(sorted(val_idx))
        else:
            val_idx = None
        train_idx = np.asarray(sorted(train_idx))

        splits[split] = SplitInfo(data_count=len(train_idx))
        if val_idx is not None:
            splits['val'] = SplitInfo(data_count=len(val_idx))

        slots['x'] = ArrayInfo(
            dtype='uint8',
            shape=list(shape),
            is_discrete=True,
            min_val=0,
            max_val=255,
            n_discrete_vals=2 ** bit_depth,
            bit_depth=bit_depth,
        )

        # now load the images
        dset2 = dset.map(lambda x: parse_tfrecord_tf(
            x, res=2 ** bit_depth, rnd_crop=False))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        train_x = np.memmap(
            filename=os.path.join(output_dir, f'{split}__x.dat'),
            dtype=np.uint8,
            mode='w+',
            shape=(len(train_idx),) + shape,
        )
        if val_idx is not None:
            val_x = np.memmap(
                filename=os.path.join(output_dir, f'val__x.dat'),
                dtype=np.uint8,
                mode='w+',
                shape=(len(val_idx),) + shape,
            )
        else:
            val_x = None

        train_i = 0
        val_i = 0

        for i, (img,) in tqdm(enumerate(dset2), total=len(labels),
                              desc=f'Load {split}'):
            if i == train_idx[train_i]:
                train_x[train_i, ...] = img
                train_i += 1
            else:
                val_x[val_i, ...] = img
                val_i += 1

        if train_i != len(train_x) or (val_x is not None and val_i != len(val_x)):
            raise RuntimeError(f'`train_i` or `val_i` does not match array size.')

        train_x.flush()
        del train_x

        if val_x is not None:
            val_x.flush()
            del val_x

    # generate the meta file
    def make_dict_array(d):
        ret = []
        for key, val in d.items():
            v = {'name': key}
            v.update(val.__dict__)
            ret.append(v)
        return ret

    yml_obj = {
        'slots': make_dict_array(slots),
        'splits': make_dict_array(splits),
    }
    yml_path = os.path.join(output_dir, 'meta.yml')
    with codecs.open(yml_path, 'wb', 'utf-8') as f:
        yaml.dump(yml_obj, f)


if __name__ == '__main__':
    main()
