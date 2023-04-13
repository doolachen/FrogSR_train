import logging
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from fogsr.compress.base import CompressAlgo

logger = logging.getLogger('KMeans')


# TODO: 科研层面思路缺陷：为什么需要KMeans量化？其他的量化手段有什么缺陷？KMeans量化有什么优势？
# TODO: 到底是为了优化什么而得来的KMeans量化？
# TODO: 不能回答上面的问题做了也是白做

def euclidean_similarity(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.T - (a ** 2).sum(axis=1)[..., :, None] - (b ** 2).sum(axis=1)[..., None, :]


def max_similarity(a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    values, indices = euclidean_similarity(a, b).max(dim=-1)
    return values, indices


class KMeansCompressor(nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.centers = centers

    def forward(self, x):
        return (x.unsqueeze(1) - self.centers.unsqueeze(0)).abs().argmin(dim=1)


class KMeansExtractor(nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.centers = centers

    def forward(self, x):
        return torch.gather(self.centers, 0, x)


class KMeans(CompressAlgo):
    def __init__(self, n_channels=48, bit=8, **kwargs):
        super().__init__()
        self.KMeansList = [MiniBatchKMeans(n_clusters=2 ** bit, **kwargs) for _ in range(n_channels)]
        self.bit = bit
        self.centers = None

    def calibrate_impl(self, x):
        assert x.shape[1] == len(self.KMeansList), "n_features of x must equal to the specified n_channels!"
        centers = []
        for i, KMeans in enumerate(self.KMeansList):
            KMeans.partial_fit(x[:, i:i + 1])
            centers.append(torch.tensor(KMeans.cluster_centers_, device=x.device, dtype=x.dtype))
        self.centers, _ = torch.sort(torch.cat(centers, dim=1), dim=0)

    def compress_impl(self, x: torch.Tensor) -> torch.Tensor:
        return (x.unsqueeze(1) - self.centers.unsqueeze(0)).abs().argmin(dim=1)

    def extract_impl(self, x: torch.Tensor) -> torch.Tensor:
        return torch.gather(self.centers, 0, x)

    # TODO: 写到这就能看出问题了：48个uint8所能表示的类不是2的8次方乘48，而是2的8次方的48次方！
    def convert_to_compressor(self) -> nn.Module:
        return KMeansCompressor(self.centers)

    def convert_to_extractor(self) -> nn.Module:
        return KMeansExtractor(self.centers)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.rand((256 * 448, 48))
    model = KMeans(max_iter=1000, tol=1e-3)
    model.calibrate_impl(x)
    xq = model.compress_impl(x)
    compressor = model.convert_to_compressor()
    with torch.no_grad():
        xq_linear = compressor(x)

    xr = model.extract_impl(xq)
    extractor = model.convert_to_extractor()
    with torch.no_grad():
        xr_linear = extractor(xq_linear)
        diff_xr = xr_linear - xr
        print(diff_xr.abs().sum() / len(diff_xr))

    diff = xr - x
    max_diff = diff.max()
    min_diff = diff.min()
    sum_diff = diff.abs().sum()
    avg_diff = sum_diff / (diff.shape[0] * diff.shape[1])
    print(avg_diff)
    print(max_diff)
    print(min_diff)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), layout='constrained')
    ax.hist(diff.reshape(-1), bins=100, density=True, alpha=0.2)
    plt.show()
