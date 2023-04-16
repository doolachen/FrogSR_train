import os

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

from fogsr.datasets import restore_images
from fogsr.trainer.loss import structural_similarity
from fogsr.trainer.vrt_util import test_vrt

path = os.path.join("lightning_logs", "vrt_test")
os.makedirs(path, exist_ok=True)

def model_large():
    from fogsr.models.vrt import VRT_Dv3
    from fogsr.models.vrt import VRTDv3_videosr_bi_Vimeo_7frames
    test_args = VRTDv3_videosr_bi_Vimeo_7frames['wrapper']['test_args']
    return VRT_Dv3(**VRTDv3_videosr_bi_Vimeo_7frames['model']), test_args

def dataloader_ugc():
    pass

def dataloader_reds():
    from config import REDS_dataloader
    return REDS_dataloader()

def main():
    model, test_args = model_large()
    test_loader = dataloader_reds()

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model to {device}")
    model = model.to(device)

    for idx, batch in enumerate(test_loader):
        lq = batch['lq'].to(device)
        gt = batch['gt']
        with torch.inference_mode():
            output = test_vrt(lq, model, **test_args)
            print(output.shape)
            input_images = restore_images(lq)
            output_images = restore_images(output)
            target_images = restore_images(gt)
            psnr_all, ssim_all, n = 0, 0, 0
            for j, (lr, output, target) in enumerate(zip(input_images, output_images, target_images)):
                psnr = peak_signal_noise_ratio(output, target)
                ssim = structural_similarity(output, target)
                psnr_all += psnr
                ssim_all += ssim
                n += 1
                cv2.imwrite(os.path.join(path, "vid4_%d_%02d_LR.png" % (idx, j)), lr)
                cv2.imwrite(os.path.join(path, "vid4_%d_%02d_HR.png" % (idx, j)), output)
                cv2.imwrite(os.path.join(path, "vid4_%d_%02d_GT.png" % (idx, j)), target)
            print('PSNR / SSIM', psnr_all / n, '/', ssim_all / n)


if __name__ == '__main__':
    main()
