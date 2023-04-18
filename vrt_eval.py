import os

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

from fogsr.datasets import restore_images
from fogsr.trainer.loss import structural_similarity
from fogsr.trainer.vrt_util import test_vrt
from fogsr.data.ugc_dataset import convert_color, img2tensor
import torchvision.utils
import torch
import torch.nn as nn


class Wrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model


def model_small():
    from fogsr.models.vrt import VRT_Dv3
    from fogsr.models.vrt import VRTDv3UltimateSmall_videosr_bi_Vimeo_7frames
    model = VRT_Dv3(**VRTDv3UltimateSmall_videosr_bi_Vimeo_7frames['model'])
    test_args = VRTDv3UltimateSmall_videosr_bi_Vimeo_7frames['wrapper']['test_args']
    ckpt = torch.load(os.path.expanduser("~/FrogSR_train/lightning_logs/version_0/checkpoints/epoch=2-step=35998.ckpt"))
    Wrapper(model).load_state_dict(ckpt["state_dict"])
    return model, test_args


lq_root = os.path.expanduser('~/dataset/ugc-dataset-image/original_videos_h264_x4lossless')
hr_root = os.path.join("tmp", "vrt_test")


def main(n=7):
    model, test_args = model_small()

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model to {device}")
    model = model.to(device)
    
    for video in os.listdir(lq_root):
        frames = sorted(os.listdir(os.path.join(lq_root, video)))
        for i in range(0, len(frames)-n):
            batch_frames = frames[i:i+n]
            batch_images = []
            for batch_frame in batch_frames:
                image = cv2.imread(os.path.join(lq_root, video, batch_frame), cv2.IMREAD_UNCHANGED)
                image = convert_color(image)
                batch_images.append(image)
            batch_images = img2tensor(batch_images)
            lq = torch.stack([torch.from_numpy(np.stack(batch_images))]).to(device)
            output = test_vrt(lq, model, device=device, **test_args)
            hr_frame = output[:,n//2,...]
            os.makedirs(os.path.join(hr_root, str(n), video), exist_ok=True)
            print(hr_frame.shape)
            torchvision.utils.save_image(
                hr_frame, os.path.join(hr_root, str(n), video, batch_frames[n//2]), nrow=1, normalize=False)


if __name__ == '__main__':
    with torch.no_grad():
        main()
