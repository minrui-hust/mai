import argparse
import os
import os.path as osp

from mai.utils import FI
from mai.utils import io

r'''
View model predictions
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('config', help='training config file')
    parser.add_argument('--split', default='val', help='which split to view')
    parser.add_argument('--pred', help='prediction result folder')
    parser.add_argument('--show_gt', action='store_true',
                        help='prediction result folder')
    return parser.parse_args()


def main(args):

    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')

    # create dataset
    config = io.load(args.config)
    dataset = FI.create(config['data'][args.split]['dataset'])

    pred_folder = args.pred

    for sample in dataset:
        # attach prediction result
        if pred_folder is not None:
            sample_name = sample['meta']['sample_name']
            pred_path = osp.join(pred_folder, f'{sample_name}.pkl')
            pred = io.load(pred_path)
            sample['pred'] = pred

        dataset.plot(sample)


if __name__ == '__main__':
    main(parse_args())
