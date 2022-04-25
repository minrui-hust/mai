import argparse
from functools import partial
import os

import yaml

from mai.utils import FI
from mai.utils import io
from mai.utils import GCFG

r'''
View dataset
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('dataset_type', help='sample dataset type')
    parser.add_argument('sample_path', help='sample path')
    parser.add_argument('--plot_args', type=partial(yaml.load,
                        Loader=yaml.FullLoader), default='{}', help='plot args')
    return parser.parse_args()


def main(args):
    dataset = FI.get(args.dataset_type)
    sample = io.load(args.sample_path)
    dataset.plot(sample, **args.plot_args)


if __name__ == '__main__':
    main(parse_args())
