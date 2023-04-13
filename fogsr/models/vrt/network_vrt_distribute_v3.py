import torch
import torch.nn as nn

from .network_vrt import VRT, Stage


# 把Stage也缩小

class VRT_Dv3(VRT):
    def __init__(self,
                 window_size=[6, 8, 8],
                 depths_v3=[8, [4, 4], [3, 3, 2], [2, 2, 2, 2], 8, 8, 8, 4, 4, 4, 4, 4, 4],
                 mul_attn_ratio_v3=[[0.5, 0.5], [0.34, 0.34, 0.5], [0.5, 0.5, 0.5, 0.5]],
                 embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                 num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 pa_frames=2,
                 deformable_groups=16,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 **kwargs):
        depths = [sum(depth) if isinstance(depth, list) else depth for depth in depths_v3]
        super().__init__(
            window_size=window_size,
            depths=depths,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pa_frames=pa_frames,
            deformable_groups=deformable_groups,
            use_checkpoint_attn=use_checkpoint_attn,
            use_checkpoint_ffn=use_checkpoint_ffn,
            no_checkpoint_attn_blocks=no_checkpoint_attn_blocks,
            no_checkpoint_ffn_blocks=no_checkpoint_ffn_blocks,
            **kwargs)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        scales = [1, 2, 4, 8, 4, 2, 1]
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in
                                range(len(depths))]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in
                               range(len(depths))]

        # pstage1里面是stage1, pstage2里面是stage1和stage2, pstage3里面是stage1和stage2和stage3
        for i in range(0, 3):
            setattr(self, f'pstage{i + 1}', nn.ModuleList([None] * (i + 1)))
        for i in range(0, 3):  # 共有3种stage需要处理
            # pstage1里面是stage1, pstage2里面是stage1和stage2, pstage3里面是stage1和stage2和stage3
            for j in range(i, 3):  # stage1在pstage123里都有, stage2在pstage23里有, stage3在pstage3里有
                # 于是stage1就被放进了pstage123里, stage2被放进了pstage23, stage3被放进了pstage3
                getattr(self, f'pstage{j + 1}')[i] = Stage(
                    in_dim=embed_dims[i - 1],
                    dim=embed_dims[i],
                    depth=depths_v3[j + 1][i],
                    num_heads=num_heads[i],
                    mul_attn_ratio=mul_attn_ratio_v3[j][i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    pa_frames=pa_frames,
                    deformable_groups=deformable_groups,
                    reshape=reshapes[i],
                    max_residue_magnitude=10 / scales[i],
                    use_checkpoint_attn=use_checkpoint_attns[i],
                    use_checkpoint_ffn=use_checkpoint_ffns[i],
                )
        # pstage1后面是stage2, pstage2后面是stage3, pstage3后面是stage4
        for i in range(1, 4):
            setattr(self, f'stage{i + 1}',
                    Stage(
                        in_dim=embed_dims[i - 1],
                        dim=embed_dims[i],
                        depth=depths_v3[i][-1],
                        num_heads=num_heads[i],
                        mul_attn_ratio=mul_attn_ratio_v3[i - 1][-1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                        norm_layer=norm_layer,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],
                        max_residue_magnitude=10 / scales[i],
                        use_checkpoint_attn=use_checkpoint_attns[i],
                        use_checkpoint_ffn=use_checkpoint_ffns[i],
                    )
                    )

    def forward_split(self, x, x_branch=(None, None, None, None),
                      early_exit_layer_idx_list: [(int, int)] = [(None, None)] * 7 + [None],
                      blocks: [str] = ['branch1', 'branch2', 'branch3', 'branch4', 'gather']):
        x, x_lq, flows_backward, flows_forward = self.forward_before(x)
        x1, x2, x3, x4 = x_branch
        x_final = None
        if 'branch1' in blocks:
            x1 = self.forward_features_branch1(x, flows_backward, flows_forward, early_exit_layer_idx_list)
        if 'branch2' in blocks:
            x2 = self.forward_features_branch2(x, flows_backward, flows_forward, early_exit_layer_idx_list)
        if 'branch3' in blocks:
            x3 = self.forward_features_branch3(x, flows_backward, flows_forward, early_exit_layer_idx_list)
        if 'branch4' in blocks:
            x4 = self.forward_features_branch4(x, flows_backward, flows_forward, early_exit_layer_idx_list)
        if 'gather' in blocks:
            features = self.forward_features_gather(x1, x2, x3, x4, flows_backward, flows_forward,
                                                    early_exit_layer_idx_list)
            features = features.transpose(1, 4)
            x = self.forward_after(features, x, x_lq)
            x_final = x
        return x_final, x1, x2, x3, x4

    def forward(self, x, early_exit_layer_idx_list: [(int, int)] = [(None, None)] * 7 + [None]):
        x, x1, x2, x3, x4 = self.forward_split(x, early_exit_layer_idx_list=early_exit_layer_idx_list)
        return x

    def forward_pstage(self, pstage, xp, flows_backward, flows_forward):
        i = 0
        for m in pstage:
            xp = m(xp, flows_backward[i::4], flows_forward[i::4])
            i += 1
        return xp

    def forward_features_branch1(self, x, flows_backward, flows_forward, early_exit_layer_idx_list):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4], early_exit_layer_idx_list[0])
        return x1

    def forward_features_branch2(self, x, flows_backward, flows_forward, early_exit_layer_idx_list):
        '''Main network for feature extraction.'''

        x1p = self.forward_pstage(self.pstage1, x, flows_backward, flows_forward)
        x2 = self.stage2(x1p, flows_backward[1::4], flows_forward[1::4], early_exit_layer_idx_list[1])
        return x2

    def forward_features_branch3(self, x, flows_backward, flows_forward, early_exit_layer_idx_list):
        '''Main network for feature extraction.'''

        x2p = self.forward_pstage(self.pstage2, x, flows_backward, flows_forward)
        x3 = self.stage3(x2p, flows_backward[2::4], flows_forward[2::4], early_exit_layer_idx_list[2])
        return x3

    def forward_features_branch4(self, x, flows_backward, flows_forward, early_exit_layer_idx_list):
        '''Main network for feature extraction.'''

        x3p = self.forward_pstage(self.pstage3, x, flows_backward, flows_forward)
        x4 = self.stage4(x3p, flows_backward[3::4], flows_forward[3::4], early_exit_layer_idx_list[3])
        return x4
