import logging
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from .base import CompressAlgo

logger = logging.getLogger('PCA')


def transform(x: torch.Tensor, dim: int):
    assert dim < len(x.shape)
    shape_in = x.shape
    n_features_in = shape_in[dim]
    x = x.permute(*list(range(0, dim)), *list(range(dim + 1, len(shape_in))), dim)
    x = x.reshape(-1, n_features_in)
    return x


def reverse(x: torch.Tensor, dim: int, shape_in):
    n_features_out = x.shape[1]
    x = x.reshape(*shape_in[0:dim], *shape_in[dim + 1:], n_features_out)
    x = x.permute(*list(range(0, dim)), len(shape_in) - 1, *list(range(dim, len(shape_in) - 1)))
    return x


def forward(op, x: torch.Tensor, dim: int):
    shape_in = x.shape
    x = transform(x, dim)
    x = op(x)
    x = reverse(x, dim, shape_in=shape_in)
    return x


class PCACompressor(nn.Linear):
    def __init__(self, compress_dim, original_dims, compressed_dims, components_=None, mean_=None):
        super().__init__(in_features=original_dims, out_features=compressed_dims, bias=True)
        self.compress_dim = compress_dim
        self.compressed_dims = compressed_dims
        if components_ is not None:
            self.weight = nn.Parameter(components_)
            if mean_ is not None:
                self.bias = nn.Parameter(-components_ @ mean_.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward(super().forward, x, self.compress_dim)


class PCAExractor(nn.Linear):
    def __init__(self, compress_dim, original_dims, compressed_dims, components_, mean_):
        super().__init__(in_features=compressed_dims, out_features=original_dims, bias=True)
        self.compress_dim = compress_dim
        self.compressed_dims = compressed_dims
        if components_ is not None:
            self.weight = nn.Parameter(components_.T)
            if mean_ is not None:
                self.bias = nn.Parameter(mean_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward(super().forward, x, self.compress_dim)


class PCA(CompressAlgo):
    def __init__(self, compress_dim, original_channel, compressed_channel=48):
        super().__init__()
        self.compress_dim = compress_dim
        self.original_dims = original_channel
        self.compressed_dims = compressed_channel
        self.PCA = IncrementalPCA(n_components=compressed_channel)
        self.components_ = None
        self.mean_ = None

    def calibrate_impl(self, x: torch.Tensor):
        assert x.shape[self.compress_dim] == self.original_dims
        self.PCA.partial_fit(transform(x, self.compress_dim).cpu())
        self.components_ = torch.tensor(self.PCA.components_, device=x.device, dtype=x.dtype)
        self.mean_ = torch.tensor(self.PCA.mean_, device=x.device, dtype=x.dtype)

    def compress_impl(self, x: torch.Tensor) -> torch.Tensor:
        return forward(lambda x: (x - self.mean_) @ self.components_.T, x, self.compress_dim)

    def extract_impl(self, x: torch.Tensor) -> torch.Tensor:
        return forward(lambda x: x @ self.components_ + self.mean_, x, self.compress_dim)

    def convert_to_compressor(self) -> nn.Module:
        return PCACompressor(self.compress_dim, self.original_dims, self.compressed_dims, self.components_, self.mean_)

    def convert_to_extractor(self) -> nn.Module:
        return PCAExractor(self.compress_dim, self.original_dims, self.compressed_dims, self.components_, self.mean_)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cpu")
    x = torch.rand((2, 120, 7, 256, 448), device=device)
    model = PCA(compress_dim=1, original_channel=120)

    model.calibrate_impl(x)
    xq = model.compress_impl(x)
    xq_gt = forward(lambda x: torch.tensor(model.PCA.transform(x.cpu()), device=device), x, 1).to(device)
    diff = xq - xq_gt
    print(diff.reshape(-1).abs().sum() / len(diff.reshape(-1)))
    compressor = model.convert_to_compressor().to(device)
    with torch.no_grad():
        xq_linear = compressor(x)
        diff = xq_linear - xq_gt
        print(diff.reshape(-1).abs().sum() / len(diff.reshape(-1)))

    xr = model.extract_impl(xq)
    xr_gt = forward(lambda x: torch.tensor(model.PCA.inverse_transform(x.cpu()), device=device), xq, 1).to(device)
    diff = xr - xr_gt
    print(diff.reshape(-1).abs().sum() / len(diff.reshape(-1)))
    extractor = model.convert_to_extractor().to(device)
    with torch.no_grad():
        xr_linear = extractor(xq_linear)
        diff = xr_linear - xr_gt
        print(diff.reshape(-1).abs().sum() / len(diff.reshape(-1)))

    diff = xr - x
    max_diff = diff.max()
    min_diff = diff.min()
    avg_diff = diff.reshape(-1).abs().sum() / len(diff.reshape(-1))
    print(avg_diff)
    print(max_diff)
    print(min_diff)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), layout='constrained')
    ax.hist(diff.reshape(-1), bins=100, density=True, alpha=0.2)
    plt.show()
