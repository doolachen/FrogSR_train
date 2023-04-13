import torch
import numpy as np

def restore_images(outputs: torch.Tensor):
    b, n, c, h, w = outputs.size()
    outputs = outputs.reshape(-1, c, h, w).clamp(0, 1).cpu().numpy()
    images = []
    for i in range(outputs.shape[0]):
        img = (outputs[i].squeeze().transpose((1, 2, 0)) * 255.0).astype(np.uint8)[:, :, [2, 1, 0]]
        images.append(img)
    return images
