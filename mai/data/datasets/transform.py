r'''
Base class of dataset transform
'''


class DatasetTransform(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, info, ds=None):
        pass
