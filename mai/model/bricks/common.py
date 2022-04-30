import torch
import torch.nn as nn
import torch.nn.functional as F

from mai.utils import FI
from mai.model import BaseModule


@FI.register
class MLP(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels=None,
                 linear_cfg=dict(type='Linear'),
                 norm_cfg=dict(type='LayerNorm'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        if out_channels is None:
            out_channels = hidden_channels

        self.lin0 = FI.create(linear_cfg, in_channels, hidden_channels)
        self.norm = FI.create(norm_cfg, hidden_channels)
        self.act = FI.create(act_cfg)
        self.lin1 = FI.create(linear_cfg, hidden_channels, out_channels)

    def forward_train(self, x):
        lin0_out = self.lin0(x)
        if self.norm is not None:
            lin0_out = self.norm(lin0_out)
        if self.act is not None:
            lin0_out = self.act(lin0_out)
        out = self.lin1(lin0_out)
        return out


@FI.register
class ResBlock(BaseModule):
    def __init__(self, residual_cfg,
                 norm_cfg=dict(type='LayerNorm', normalized_shape=64),
                 act_cfg=dict(type='ReLU', inplace=True),
                 sampler=None):
        super().__init__()

        self.residual = FI.create(residual_cfg)
        self.norm = FI.create(norm_cfg)
        self.act = FI.create(act_cfg)
        self.sampler = sampler

    def forward_train(self, x, *args, **kwargs):
        identity = x
        if self.sampler:
            identity = self.sampler(identity)

        out = self.norm(identity + self.residual(x, *args, **kwargs))

        if self.act:
            out = self.act(out)

        return out


@FI.register
class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 atten_cfg=dict(type='MultiHeadSelfAtten'),
                 ff_cfg=dict(type='MLP'),
                 norm_cfg=dict(type='LayerNorm')):
        super().__init__()

        self.atten_res_block = ResBlock(atten_cfg, norm_cfg, act_cfg=None)
        self.ff_res_block = ResBlock(ff_cfg, norm_cfg, act_cfg=None)

    def forward_train(self, x, **args):
        att_out = self.atten_res_block(x, **args)
        ff_out = self.ff_res_block(att_out)
        return ff_out


@FI.register
class TransformerDecoderLayer(BaseModule):
    def __init__(self, self_atten_cfg=dict(type='MultiHeadSelfAtten'),
                 cross_atten_cfg=dict(type='MultiHeadAtten'),
                 ff_cfg=dict(type='MLP'),
                 norm_cfg=dict(type='LayerNorm')):
        super().__init__()

        self.self_atten_res_block = ResBlock(
            self_atten_cfg, norm_cfg, act_cfg=None)
        self.cross_atten_res_block = ResBlock(
            cross_atten_cfg, norm_cfg, act_cfg=None)
        self.ff_res_block = ResBlock(ff_cfg, norm_cfg, act_cfg=None)

    def forward_train(self, x, y, **args):
        self_atten_out = self.self_atten_res_block(x)
        cross_atten_out = self.cross_atten_res_block(self_atten_out, y, **args)
        ff_out = self.ff_res_block(cross_atten_out)
        return ff_out


@FI.register
class TransformerEncoder(BaseModule):
    def __init__(self, layer_cfg, layer_num=1):
        super().__init__()

        self.layer_list = nn.ModuleList(
            [FI.create(layer_cfg) for _ in range(layer_num)])

    def forward_train(self, x, **args):
        for layer in self.layer_list:
            x = layer(x, **args)
        return x


@FI.register
class TransformerDecoder(BaseModule):
    def __init__(self, layer_cfg, layer_num=1):
        super().__init__()

        self.layer_list = nn.ModuleList(
            [FI.create(layer_cfg) for _ in range(layer_num)])

    def forward_train(self, x, y, **args):
        for layer in self.layer_list:
            x = layer(x, y, **args)
        return x


@FI.register
class PointnetLayer(BaseModule):
    def __init__(self, in_channels, hidden_channels, linear_cfg, norm_cfg, act_cfg, feature_first=False):
        r'''
        Args:
            dim: indicate which dim is feature dim
        '''
        super().__init__()
        self.lin = FI.create(linear_cfg, in_channels, hidden_channels)
        self.norm = FI.create(norm_cfg, hidden_channels)
        self.act = FI.create(act_cfg)

        self.norm_type = norm_cfg['type']
        self.feature_dim = -2 if feature_first else -1
        self.spatial_dim = -1 if feature_first else -2

    def forward_train(self, x, mask=None):
        x = self.lin(x)
        if self.norm is not None:
            if self.norm_type == 'LayerNorm' and mask is not None:
                x = x.masked_fill(mask, 0)
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if mask is not None:
            x = x.masked_fill(mask, float('-inf'))

        x_max = torch.max(x, dim=self.spatial_dim, keepdim=True)[0]
        x = torch.cat([x, x_max.expand_as(x)], dim=self.feature_dim)

        return x, x_max.squeeze(self.spatial_dim)


@FI.register
class PointNet(BaseModule):
    def __init__(self, in_channels, hidden_channels, num_layers, linear_cfg, norm_cfg, act_cfg, feature_first=False, ret_point_wise=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_cfg = dict(
                type='PointnetLayer',
                in_channels=in_channels if i == 0 else 2*hidden_channels,
                hidden_channels=hidden_channels,
                feature_first=feature_first,
                linear_cfg=linear_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.layers.append(FI.create(layer_cfg))

        self.ret_point_wise = ret_point_wise

    def forward_train(self, x, mask=None):
        for layer in self.layers:
            x, x_max = layer(x, mask)

        if self.ret_point_wise:
            return x
        else:
            return x_max
