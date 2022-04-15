import torch
import torch.nn as nn
import torch.nn.functional as F

from mai.utils import FI


@FI.register
class MultiHeadSelfAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()

        self.atten = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout,
                                           batch_first=batch_first)

    def forward(self, x, mask=None):
        if mask is None:
            return self.atten(x, x, x, key_padding_mask=mask, need_weights=False)[0]

        valid_batch_mask = torch.any(torch.logical_not(mask), dim=1)

        if torch.sum(valid_batch_mask) == len(x):  # all sample in batch is valid
            return self.atten(x, x, x, key_padding_mask=mask, need_weights=False)[0]
        else:
            x_valid = x[valid_batch_mask]
            mask_valid = mask[valid_batch_mask]
            out = x.clone()
            out[valid_batch_mask] = self.atten(
                x_valid, x_valid, x_valid, key_padding_mask=mask_valid, need_weights=False)[0]
            return out


@FI.register
class MultiHeadCrossAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()

        self.atten = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout,
                                           batch_first=batch_first)

    def forward(self, x, y, mask=None):
        if mask is None:
            return self.atten(x, y, y, key_padding_mask=mask, need_weights=False)[0]

        valid_batch_mask = torch.any(torch.logical_not(mask), dim=1)
        if torch.sum(valid_batch_mask) == len(y):
            return self.atten(x, y, y, key_padding_mask=mask, need_weights=False)[0]
        else:
            x_valid = x[valid_batch_mask]
            y_valid = y[valid_batch_mask]
            mask_valid = mask[valid_batch_mask]
            out = x.clone()
            out[valid_batch_mask] = self.atten(
                x_valid, y_valid, y_valid, key_padding_mask=mask_valid, need_weights=False)[0]
            return out
