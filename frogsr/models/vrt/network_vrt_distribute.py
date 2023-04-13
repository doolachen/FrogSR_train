import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .network_vrt import VRT


def norm(in_dim, dim):
    return nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                         nn.LayerNorm(dim),
                         Rearrange('n d h w c -> n c d h w'))


def downsample(in_dim, dim):
    return nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                         nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim),
                         Rearrange('n d h w c -> n c d h w'))


class VRT_D(VRT):
    def __init__(self, embed_dims, **kwargs):
        super().__init__(embed_dims=embed_dims, **kwargs)
        # stage 1,2,3
        for i in range(0, 3):
            in_dim = embed_dims[i - 1]
            dim = embed_dims[i]
            setattr(
                self, f'reshape{i + 1}',
                norm(in_dim=in_dim, dim=dim) if i == 0 else
                nn.Sequential(*(downsample(in_dim=in_dim, dim=dim) for _ in range(i)))
            )

    def forward_features(self, x, flows_backward, flows_forward,
                         early_exit_layer_idx_list: [(int, int)] = [(None, None)] * 7 + [None]):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4], early_exit_layer_idx_list[0])
        x1p = self.reshape1(x)
        x2 = self.stage2(x1p, flows_backward[1::4], flows_forward[1::4], early_exit_layer_idx_list[1])
        x2p = self.reshape2(x)
        x3 = self.stage3(x2p, flows_backward[2::4], flows_forward[2::4], early_exit_layer_idx_list[2])
        x3p = self.reshape3(x)
        x4 = self.stage4(x3p, flows_backward[3::4], flows_forward[3::4], early_exit_layer_idx_list[3])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4], early_exit_layer_idx_list[4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4], early_exit_layer_idx_list[5])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4], early_exit_layer_idx_list[6])
        x = x + x1

        for layer in self.stage8[0:early_exit_layer_idx_list[7]]:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x
