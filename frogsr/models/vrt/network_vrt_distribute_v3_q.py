import torch.quantization as qnn

from .network_vrt_distribute_v3 import VRT_Dv3
from .quant import forward_split_q


# 把Stage也缩小

class VRT_Dv3Q(VRT_Dv3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i in range(4):
            setattr(self, f"quant{i + 1}", qnn.QuantStub())
            setattr(self, f"dquant{i + 1}", qnn.DeQuantStub())

    def _forward_split(self, x, x_branch=(None, None, None, None),
                       early_exit_layer_idx_list: [(int, int)] = [(None, None)] * 7 + [None],
                       blocks: [str] = ['branch1', 'branch2', 'branch3', 'branch4', 'gather']):
        x, x_lq, flows_backward, flows_forward = self.forward_before(x)
        x1, x2, x3, x4 = x_branch
        x_final = None
        if 'branch1' in blocks:
            x1 = self.forward_features_branch1(x, flows_backward, flows_forward, early_exit_layer_idx_list)
            x1 = self.quant1(x1)
        if 'branch2' in blocks:
            x2 = self.forward_features_branch2(x, flows_backward, flows_forward, early_exit_layer_idx_list)
            x2 = self.quant2(x2)
        if 'branch3' in blocks:
            x3 = self.forward_features_branch3(x, flows_backward, flows_forward, early_exit_layer_idx_list)
            x3 = self.quant3(x3)
        if 'branch4' in blocks:
            x4 = self.forward_features_branch4(x, flows_backward, flows_forward,
                                               early_exit_layer_idx_list)
            x4 = self.quant4(x4)
        if 'gather' in blocks:
            features = self.forward_features_gather(
                self.dquant1(x1), self.dquant2(x2), self.dquant3(x3), self.dquant4(x4),
                flows_backward, flows_forward, early_exit_layer_idx_list)
            features = features.transpose(1, 4)
            x = self.forward_after(features, x, x_lq)
            x_final = x
        return x_final, x1, x2, x3, x4

    def forward_split(self, *args, **kwargs):
        return forward_split_q(
            self._forward_split,
            (self.quant1, self.quant2, self.quant3, self.quant4),
            *args, **kwargs
        )
