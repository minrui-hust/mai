import torch
import torch.nn as nn


class BaseModule(nn.Module):
    r'''
    Extention of torch.nn.Module, support different mode('train', 'val', 'test', 'infer')
    '''

    def __init__(self):
        super().__init__()
        self.mode = 'train'

    def set_train(self):
        self.set_mode('train')

    def set_eval(self):
        self.set_mode('eval')

    def set_export(self):
        self.set_mode('export')

    def set_mode(self, mode):
        assert(mode in ['train', 'eval', 'export'])
        if mode in ['train']:
            self.train()
        if mode in ['eval', 'export']:
            self.eval()
        self.mode = mode

        for m in self.children():
            if isinstance(m, BaseModule):
                m.set_mode(mode)

    def freeze(self):
        self.do_freeze(self)

    def do_freeze(self, module):
        # freeze parameter of this module
        for _, param in module._parameters.items():
            if param is not None:
                param.requires_grad = False

        # freeze parameter of submodule
        for _, submodule in module._modules.items():
            self.do_freeze(submodule)

    def forward(self, *args, **kwargs):
        if self.mode == 'train':
            return self.forward_train(*args, **kwargs)
        elif self.mode == 'eval':
            return self.forward_eval(*args, **kwargs)
        elif self.mode == 'export':
            return self.forward_export(*args, **kwargs)
        else:
            raise NotImplementedError

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    @torch.no_grad()
    def forward_export(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
