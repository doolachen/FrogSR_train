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
import multiprocessing as mp


class Wrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

q_batch = mp.Queue(16)
q_frames = mp.Queue(16)

def model_small():
    from fogsr.models.vrt import VRT_Dv3
    from fogsr.models.vrt import VRTDv3UltimateSmall_videosr_bi_Vimeo_7frames
    model = VRT_Dv3(**VRTDv3UltimateSmall_videosr_bi_Vimeo_7frames['model'])
    test_args = VRTDv3UltimateSmall_videosr_bi_Vimeo_7frames['wrapper']['test_args']
    ckpt = torch.load(os.path.expanduser("~/FrogSR_train/lightning_logs/version_0/checkpoints/epoch=3-step=65997.ckpt"))
    Wrapper(model).load_state_dict(ckpt["state_dict"])
    return model, test_args


lq_root = os.path.expanduser('~/dataset/ugc-dataset-image/original_videos_h264_x4lossless')
hr_root = os.path.join("tmp", "vrt_test")

def batch_path_loader(n, lq_root):
    for video in os.listdir(lq_root):
        frames = sorted(os.listdir(os.path.join(lq_root, video)))
        for i in range(0, len(frames)-n):
            q_frames.put((lq_root, video, frames[i:i+n]))
            
    
def batch_lq_loader():
    while True:
        lq_root, video, batch_frames = q_frames.get()
        batch_images = []
        for batch_frame in batch_frames:
            image = cv2.imread(os.path.join(lq_root, video, batch_frame), cv2.IMREAD_UNCHANGED)
            image = convert_color(image)
            batch_images.append(image)
        batch_images = img2tensor(batch_images)
        lq = torch.stack([torch.from_numpy(np.stack(batch_images))])
        q_batch.put((video, batch_frames, lq))

def main(n=3):
    
    path_loader = mp.Process(target=batch_path_loader, args=(n, lq_root))
    path_loader.start()
    
    lq_loader = mp.Pool(16)
    for _ in range(16):
        lq_loader.apply_async(func=batch_lq_loader)
        
    while True:
        video, batch_frames, lq = q_batch.get()
        print(video, batch_frames, lq.shape)


if __name__ == '__main__':
    with torch.no_grad():
        main()
