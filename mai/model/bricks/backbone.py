import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

from mai.utils import FI
from mai.model import BaseModule


@FI.register
class ModelIntermediateLayers(BaseModule):
    r'''
    Warning: model should meet the requirement of IntermediateLayerGetter, see
    https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py#L15
    '''

    def __init__(self, model, interm_layers):
        r'''
        Args:
            model: the model config used to create model by FI
            interm_layers: a dict specify which intermediate layers to output
        '''
        super().__init__()
        model = FI.create(model)
        self.model = IntermediateLayerGetter(model, interm_layers)

    def forward_train(self, x):
        out_dict = self.model(x)

        # make output a list
        out_list = [(name, value) for name, value in out_dict.items()]
        out_list.sort(key=lambda x: int(x[0]))
        out_list = [value for _, value in out_list]

        return out_list
