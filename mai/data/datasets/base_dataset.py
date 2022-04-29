from torch.utils.data import Dataset as TorchDataset
import numpy as np

from mai.data.datasets.sample import Sample
from mai.utils import FI
from mai.utils import io


class BaseDataset(TorchDataset):
    r'''
    Base class of all dataset
    '''

    def __init__(self, info_path, filters=[], transforms=[], codec=None):
        super().__init__()

        self.info_path = info_path

        self.filters = [FI.create(cfg) for cfg in filters]

        self.transforms = [FI.create(cfg) for cfg in transforms]

        self.codec = FI.create(codec)

        print(f'Loading info from {self.info_path}')
        self.sample_infos = io.load(self.info_path, format='pkl')
        for filter in self.filters:
            self.sample_infos = filter(self.sample_infos)
        print(f'Load info done!')

    def __len__(self):
        return len(self.sample_infos)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.__get_one_item(i) for i in idx]
        else:
            return self._get_one_item(idx)

    def _get_one_item(self, idx):
        sample = Sample()
        self.process(sample, self.sample_infos[idx])
        return sample

    def process(self, sample, info):
        self.load(sample, info)
        self.transform(sample, info)
        self.encode(sample, info)

    def transform(self, sample, info):
        if self.transforms:
            for t in self.transforms:
                t(sample, info, self)

    def encode(self, sample, info):
        r'''
        encode standard sample format to task specified
        '''
        if self.codec:
            self.codec.encode(sample, info)

    def load(self, sample, info):
        raise NotImplementedError

    @classmethod
    def plot(cls, sample, **kwargs):
        raise NotImplementedError

    @classmethod
    def format(cls, result, pred_path=None, gt_path=None):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, predict_path, gt_path=None):
        raise NotImplementedError


@FI.register
class ConcatDataset(TorchDataset):
    r'''
    Concat a list of datasets(of same type) to be a new one
    '''

    def __init__(self, datasets=[], codec=None):
        super().__init__()
        self.datasets = [FI.create(ds) for ds in datasets]
        self.codec = FI.create(codec)

        offset = 0
        self.datasets_range = []
        for ds in self.datasets:
            self.datasets_range.append([offset, offset+len(ds)])
            offset += len(ds)
        self.datasets_range = np.array(self.datasets_range, dtype=np.int32)

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._get_one_item(i) for i in idx]
        else:
            return self._get_one_item(idx)

    def _get_one_item(self, idx):
        ds_idx = np.searchsorted(self.datasets_range[:, 1], idx)
        return self.datasets[ds_idx][idx - self.datasets_range[ds_idx, 0]]

    def plot(self, sample, **kwargs):
        self.datasets[0].plot(sample, **kwargs)

    @classmethod
    def format(cls, result, pred_path=None, gt_path=None):
        return pred_path, gt_path

    @classmethod
    def evaluate(cls, predict_path, gt_path=None):
        return {}

    @property
    def codec(self):
        return self._codec

    @codec.setter
    def codec(self, codec):
        self._codec = codec
        for ds in self.datasets:
            ds.codec = self._codec


@FI.register
class RepeatedDataset(TorchDataset):
    r'''
    Repeat a dataset to be a larger new one
    '''

    def __init__(self, dataset=None, repeat_times=2, codec=None):
        super().__init__()
        self.repeat_times = repeat_times
        self.dataset = FI.create(dataset)
        self.codec = FI.create(codec)

    def __len__(self):
        return self.repeat_times * len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._get_one_item(i) for i in idx]
        else:
            return self._get_one_item(idx)

    def _get_one_item(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def plot(self, sample, **kwargs):
        self.dataset.plot(sample, **kwargs)

    @classmethod
    def format(cls, result, pred_path=None, gt_path=None):
        return pred_path, gt_path

    @classmethod
    def evaluate(cls, predict_path, gt_path=None):
        return {}

    @property
    def codec(self):
        return self._codec

    @codec.setter
    def codec(self, codec):
        self._codec = codec
        self.dataset.codec = self._codec


@FI.register
class RandomDownsampleDataset(TorchDataset):
    r'''
    random downsample of a dataset, each get item will return a random sample
    '''

    def __init__(self, dataset=None, factor=0.5, codec=None):
        super().__init__()
        self.factor = factor
        self.dataset = FI.create(dataset)
        self.codec = FI.create(codec)

        self.random_indice = np.random.shuffle(np.arange(len(self.dataset)))
        self.inner_idx = 0

    def __len__(self):
        return int(self.factor * len(self.dataset))

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._get_one_item(i) for i in idx]
        else:
            return self._get_one_item(idx)

    def _get_one_item(self, idx):
        sample = self.dataset[self.random_indice[self.inner_idx]]

        self.inner_idx += 1
        if self.inner_idx == len(self.dataset):
            self.random_indice = np.random.shuffle(self.random_indice)

        return sample

    def plot(self, sample, **kwargs):
        self.dataset.plot(sample, **kwargs)

    @classmethod
    def format(cls, result, pred_path=None, gt_path=None):
        return pred_path, gt_path

    @classmethod
    def evaluate(cls, predict_path, gt_path=None):
        return {}

    @property
    def codec(self):
        return self._codec

    @codec.setter
    def codec(self, codec):
        self._codec = codec
        self.dataset.codec = self._codec
