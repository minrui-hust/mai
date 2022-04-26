from torch.utils.data import Dataset as TorchDataset

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
            return [self.__getitem__(id) for id in idx]
        else:
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
