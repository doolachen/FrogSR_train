import torch

qtype = [torch.quint8, torch.qint8, torch.qint32, torch.quint4x2, torch.quint2x4]

itype = [torch.uint8, torch.int8, torch.int32]


def quantize_to_uint(x):
    return torch.int_repr(x)


def uint_to_quantize(quant_layer, x):
    x = (x - quant_layer.zero_point).float() * quant_layer.scale
    return quant_layer(x)


def forward_split_q(forward_split, quants, x, x_branch=(None, None, None, None), *args, **kwargs):
    x_branch = [
        uint_to_quantize(quant, xb) if isinstance(xb, torch.Tensor) and xb.dtype in itype else xb
        for xb, quant in zip(x_branch, quants)
    ]
    outputs = forward_split(x=x, x_branch=x_branch, *args, **kwargs)
    x_final, x_branch = outputs[0], outputs[1:]
    x_branch = [
        quantize_to_uint(xb) if isinstance(xb, torch.Tensor) and xb.dtype in qtype else xb
        for xb, quant in zip(x_branch, quants)
    ]
    return x_final, *x_branch
