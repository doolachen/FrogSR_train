import os

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

from fogsr.datasets import restore_images
from fogsr.trainer.loss import structural_similarity
from fogsr.trainer.vrt_util import test_vrt

path = "vrt_test"
os.makedirs(path, exist_ok=True)


def main():
    config = parse_config()
    model = parse_model(config)
    model, test_loader = model.model, model.val_dataloader()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model to {device}")
    model = model.to(device)

    for idx, batch in enumerate(test_loader):
        lq = batch['LRs'].to(device)
        gt = batch['HRs']
        # lq, gt = crop(lq, gt)
        f_lr = open(os.path.join(path, "vid4_lr_%d.yuv" % idx), 'wb')
        f_hr = open(os.path.join(path, "vid4_hr_%d.yuv" % idx), 'wb')
        f_gt = open(os.path.join(path, "vid4_gt_%d.yuv" % idx), 'wb')
        # inference
        with torch.inference_mode():
            output = test_vrt(lq, model, **config['wrapper']['test_args'])
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
                lr_yuv = cv2.cvtColor(lr, cv2.COLOR_BGR2YUV_I420)
                hr_yuv = cv2.cvtColor(output, cv2.COLOR_BGR2YUV_I420)
                gt_yuv = cv2.cvtColor(target, cv2.COLOR_BGR2YUV_I420)
                f_lr.write(lr_yuv.astype(np.uint8).tobytes())
                f_hr.write(hr_yuv.astype(np.uint8).tobytes())
                f_gt.write(gt_yuv.astype(np.uint8).tobytes())
            print('PSNR / SSIM', psnr_all / n, '/', ssim_all / n)
            f_lr.close()
            f_hr.close()
            f_gt.close()


if __name__ == '__main__':
    main()
