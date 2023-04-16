
import torch
import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    

from fogsr.models.vrt import VRT_Dv3
from fogsr.models.vrt import VRTDv3SmallDepthsTail_videosr_bi_Vimeo_7frames
model = VRT_Dv3(**VRTDv3SmallDepthsTail_videosr_bi_Vimeo_7frames['model'])

ckpt = torch.load("/home/cbj/models/VRTDv3SmallDepthsTail_videosr_bi_Vimeo_7frames.ckpt")
Wrapper(model).load_state_dict(ckpt["state_dict"])
torch.save(model.state_dict(), "/home/cbj/models/VRTDv3SmallDepthsTail_videosr_bi_Vimeo_7frames.pth")