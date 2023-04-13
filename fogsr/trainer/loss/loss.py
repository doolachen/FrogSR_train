import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio

from fogsr.datasets import restore_images
from .metrics import structural_similarity


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


def quality(outputs, targets):
    output_images = restore_images(outputs)
    target_images = restore_images(targets)
    psnr_all, ssim_all, n = 0, 0, 0
    for output, target in zip(output_images, target_images):
        psnr = peak_signal_noise_ratio(output, target)
        ssim = structural_similarity(output, target)
        psnr_all += psnr
        ssim_all += ssim
        n += 1
    psnr = psnr_all / n
    ssim = ssim_all / n
    return psnr, ssim
