import torch.nn as nn


class ConvCompress(nn.Sequential):
    def __init__(self, in_dims, compressed_dims=48, compress_kernel_size=7):
        super().__init__(
            nn.Conv2d(in_channels=in_dims, out_channels=compressed_dims,
                      kernel_size=compress_kernel_size, stride=1, padding=(compress_kernel_size - 1) // 2),
            nn.ReLU(inplace=False))
        self.compressed_dims = compressed_dims

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        b, d, c, h, w = x.shape
        x = x.reshape(b * d, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, d, self.compressed_dims, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class ConvExtract(nn.Sequential):
    def __init__(self, out_dims, compressed_dims=48, compress_kernel_size=7):
        super().__init__(
            nn.Conv2d(in_channels=compressed_dims, out_channels=out_dims,
                      kernel_size=compress_kernel_size, stride=1, padding=(compress_kernel_size - 1) // 2),
            nn.ReLU(inplace=False))
        self.out_dims = out_dims

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        b, d, c, h, w = x.shape
        x = x.reshape(b * d, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, d, self.out_dims, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        return x
