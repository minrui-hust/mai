import argparse
from functools import partial
import os

import numpy as np
import yaml

from mai.data.datasets.base_dataset import ConcatDataset

from mai.utils import FI
from mai.utils import io
from mai.utils import GCFG

r'''
View dataset
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('config', help='training config file')
    parser.add_argument('--split', default='eval',
                        choices=['train', 'eval', 'export'], help='which split to view')
    parser.add_argument('--plot_args', type=partial(yaml.load,
                        Loader=yaml.FullLoader), default='{}', help='show model prediction')
    parser.add_argument('--dataset_root', type=str, nargs='+',
                        help='the dataset root folder, this will override config')
    parser.add_argument('--shuffle', action='store_true',
                        help='random shuffle')
    return parser.parse_args()


def main(args):

    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')

    if args.dataset_root:
        print(f'INFO: override dataset_root to {args.dataset_root}')
        if len(args.dataset_root) == 1:
            GCFG['dataset_root'] = args.dataset_root[0]  # backward compatance
        else:
            GCFG['dataset_root'] = args.dataset_root

    config = io.load(args.config)
    ds_cfg = config['data'][args.split]['dataset']
    if isinstance(ds_cfg, list):
        dataset = ConcatDataset(ds_cfg)
    else:
        dataset = FI.create(ds_cfg)

    indices = np.arange(len(dataset))
    if args.shuffle:
        np.random.shuffle(indices)

    for i in indices:
        sample = dataset[i]
        dataset.plot(sample, **args.plot_args)


if __name__ == '__main__':
    main(parse_args())
