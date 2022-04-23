import math
import torch


def SinPositionalEncoding1D(length, d_model):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))

    # in range [0, 2pi)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(
        1) * (2*math.pi / length)

    # in range [1, 10000)
    scale_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float32) *
                           (math.log(10000.0) / d_model))

    pe = torch.empty(length, d_model)
    pe[:, 0::2] = torch.cos(position * scale_term)
    pe[:, 1::2] = torch.sin(position * scale_term)

    return pe


def SinPositionalEncoding2D(height, width, d_model):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))

    pe = torch.empty(d_model, height, width)

    # Each dimension use half of d_model
    d_model = int(d_model / 2)

    pos_w = torch.arange(0, width, dtype=torch.float32).unsqueeze(
        1) * (2 * math.pi / width)
    pos_h = torch.arange(0, height, dtype=torch.float32).unsqueeze(
        1) * (2 * math.pi / height)

    scale_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float32) *
                           (math.log(10000.0) / d_model))

    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * scale_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * scale_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)

    pe[d_model::2, :, :] = torch.sin(
        pos_h * scale_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * scale_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
