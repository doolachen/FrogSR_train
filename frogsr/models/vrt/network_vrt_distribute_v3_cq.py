from fogsr.compress import stub_pair, PCA, ConvCompress, ConvExtract
from .network_vrt_distribute_v3_q import VRT_Dv3Q


# 把Stage也缩小

class VRT_Dv3CQ(VRT_Dv3Q):
    def __init__(self, compress_on_dim=1, compressed_dims=48,
                 embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                 **kwargs):
        super().__init__(**kwargs)
        for i in range(4):
            c, e = stub_pair(PCA(compress_on_dim, original_channel=embed_dims[i], compressed_channel=compressed_dims))
            setattr(self, f"compress{i + 1}", c)
            setattr(self, f"extract{i + 1}", e)

    def _forward_split(self, x, x_branch=(None, None, None, None),
                       early_exit_layer_idx_list: [(int, int)] = [(None, None)] * 7 + [None],
                       blocks: [str] = ['branch1', 'branch2', 'branch3', 'branch4', 'gather']):
        x, x_lq, flows_backward, flows_forward = self.forward_before(x)
        x1, x2, x3, x4 = x_branch
        x_final = None
        if 'branch1' in blocks:
            x1 = self.forward_features_branch1(x, flows_backward, flows_forward, early_exit_layer_idx_list)
            x1 = self.compress1(x1)
            x1 = self.quant1(x1)
        if 'branch2' in blocks:
            x2 = self.forward_features_branch2(x, flows_backward, flows_forward, early_exit_layer_idx_list)
            x2 = self.compress2(x2)
            x2 = self.quant2(x2)
        if 'branch3' in blocks:
            x3 = self.forward_features_branch3(x, flows_backward, flows_forward, early_exit_layer_idx_list)
            x3 = self.compress3(x3)
            x3 = self.quant3(x3)
        if 'branch4' in blocks:
            x4 = self.forward_features_branch4(x, flows_backward, flows_forward,
                                               early_exit_layer_idx_list)
            x4 = self.compress4(x4)
            x4 = self.quant4(x4)
        if 'gather' in blocks:
            features = self.forward_features_gather(
                self.extract1(self.dquant1(x1)),
                self.extract2(self.dquant2(x2)),
                self.extract3(self.dquant3(x3)),
                self.extract4(self.dquant4(x4)),
                flows_backward, flows_forward, early_exit_layer_idx_list)
            features = features.transpose(1, 4)
            x = self.forward_after(features, x, x_lq)
            x_final = x
        return x_final, x1, x2, x3, x4


class VRT_Dv3DCQ(VRT_Dv3CQ):
    def __init__(self,
                 embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                 compressed_dims=48, compress_kernel_size=7,
                 **kwargs):
        super().__init__(embed_dims=embed_dims, **kwargs)
        for i in range(4):
            setattr(self, f"compress{i + 1}", ConvCompress(
                in_dims=embed_dims[i], compressed_dims=compressed_dims, compress_kernel_size=compress_kernel_size))
            setattr(self, f"extract{i + 1}", ConvExtract(
                out_dims=embed_dims[i], compressed_dims=compressed_dims, compress_kernel_size=compress_kernel_size))
