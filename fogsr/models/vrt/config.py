import json
import re

config = {}

VRT_videosr_6frames = dict(
    upscale=4,
    num_frames=6,
    window_size=[6, 8, 8],
    depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
    indep_reconsts=[11, 12],
    embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    pa_frames=2,
    deformable_groups=12
)
VRT_videosr_bi_REDS_6frames = dict(
    model=dict(
        **VRT_videosr_6frames,
        pretrained_url='https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth'
    ),
    wrapper=dict(
        name='fogsr.trainer.VRTLightningWrapper',
        batch_size=1,
        test_args=dict(
            scale=4,
            lq_clip=[6, 64, 64],
            scale_branch=[1, 2, 4, 8],
            channel_branch=[120, 120, 120, 120],
            max_batch_size=16
        )
    ),
)

VRT_videosr_16frames = dict(
    upscale=4,
    num_frames=16,
    window_size=[8, 8, 8],
    depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
    indep_reconsts=[11, 12],
    embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    pa_frames=6,
    deformable_groups=24
)
VRT_videosr_bi_REDS_16frames = dict(
    model=dict(
        **VRT_videosr_16frames,
        pretrained_url='https://github.com/JingyunLiang/VRT/releases/download/v0.0/002_VRT_videosr_bi_REDS_16frames.pth'
    ),
    wrapper=dict(
        name='fogsr.trainer.VRTLightningWrapper',
        batch_size=1,
        test_args=dict(
            scale=4,
            lq_clip=[16, 64, 64],
            scale_branch=[1, 2, 4, 8],
            channel_branch=[120, 120, 120, 120],
            max_batch_size=16
        )
    ),
)

config = dict(
    **config,
    VRT_videosr_bi_REDS_6frames=VRT_videosr_bi_REDS_6frames,
    VRT_videosr_bi_REDS_16frames=VRT_videosr_bi_REDS_16frames,
)


def to_vid4_bdx4(conf):
    conf = json.loads(json.dumps(conf))
    conf['wrapper']['test_args'] = {**conf['wrapper']['test_args'], **dict(
        lq_clip=[7, 64, 64],
    )}
    return conf


def to_vid4_bix4(conf):
    conf = json.loads(json.dumps(conf))
    conf['wrapper']['test_args'] = {**conf['wrapper']['test_args'], **dict(
        lq_clip=[7, 64, 64],
    )}
    return conf


VRT_videosr_7frames = dict(
    upscale=4,
    num_frames=8,
    window_size=[8, 8, 8],
    depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
    indep_reconsts=[11, 12],
    embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    pa_frames=4,
    deformable_groups=16
)
VRT_videosr_bi_Vimeo_7frames = dict(
    model=dict(
        **VRT_videosr_7frames,
        pretrained_url='https://github.com/JingyunLiang/VRT/releases/download/v0.0/003_VRT_videosr_bi_Vimeo_7frames.pth'
    ),
    wrapper=dict(
        name='fogsr.trainer.VRTLightningWrapper',
        batch_size=1,
        test_args=dict(
            scale=4,
            lq_clip=[7, 64, 64],
            scale_branch=[1, 2, 4, 8],
            channel_branch=[120, 120, 120, 120],
            max_batch_size=16
        )
    ),
)
VRT_videosr_bi_Vimeo_7frames_vid4 = to_vid4_bix4(VRT_videosr_bi_Vimeo_7frames)

VRT_videosr_bd_Vimeo_7frames = dict(
    model=dict(
        **VRT_videosr_7frames,
        pretrained_url='https://github.com/JingyunLiang/VRT/releases/download/v0.0/004_VRT_videosr_bd_Vimeo_7frames.pth'
    ),
    wrapper=dict(
        name='fogsr.trainer.VRTLightningWrapper',
        batch_size=1,
        test_args=dict(
            scale=4,
            lq_clip=[7, 64, 64],
            scale_branch=[1, 2, 4, 8],
            channel_branch=[120, 120, 120, 120],
            max_batch_size=16
        )
    ),
)
VRT_videosr_bd_Vimeo_7frames_vid4 = to_vid4_bdx4(VRT_videosr_bd_Vimeo_7frames)

config = dict(
    **config,
    VRT_videosr_bi_Vimeo_7frames=VRT_videosr_bi_Vimeo_7frames,
    VRT_videosr_bi_Vimeo_7frames_vid4=VRT_videosr_bi_Vimeo_7frames_vid4,
    VRT_videosr_bd_Vimeo_7frames=VRT_videosr_bd_Vimeo_7frames,
    VRT_videosr_bd_Vimeo_7frames_vid4=VRT_videosr_bd_Vimeo_7frames_vid4,
)


def to_small_depths(conf):
    conf = json.loads(json.dumps(conf))
    conf['model']['depths'] = [d // 2 if isinstance(d, int) else d for d in conf['model']['depths']]
    conf['model']['strict'] = False
    return conf


def to_small_tail(conf):
    conf = json.loads(json.dumps(conf))
    conf['model']['depths'] = conf['model']['depths'][0:-5]
    conf['model']['strict'] = False
    return conf


def to_small_dims(conf):
    conf = json.loads(json.dumps(conf))
    conf['model']['embed_dims'] = [d // 5 for d in conf['model']['embed_dims']]
    conf['model']['pretrained_url'] = None
    conf['wrapper']['test_args']['channel_branch'] = [d // 5 for d in conf['wrapper']['test_args']['channel_branch']]
    return conf


def to_medium_dims(conf):
    conf = json.loads(json.dumps(conf))
    conf['model']['embed_dims'] = [d * 2 // 5 for d in conf['model']['embed_dims']]
    conf['model']['pretrained_url'] = None
    conf['wrapper']['test_args']['channel_branch'] = [d * 2 // 5 for d in
                                                      conf['wrapper']['test_args']['channel_branch']]
    return conf


def to_small_depths_tail(conf):
    conf = to_small_depths(conf)
    conf = to_small_tail(conf)
    return conf


def to_small_dims_tail(conf):
    conf = to_small_dims(conf)
    conf = to_small_tail(conf)
    return conf


def to_small_depths_dims_tail(conf):
    conf = to_small_depths(conf)
    conf = to_small_dims_tail(conf)
    return conf


def to_medium_dims_tail(conf):
    conf = to_medium_dims(conf)
    conf = to_small_tail(conf)
    return conf


def to_small_depths_medium_dims_tail(conf):
    conf = to_small_depths(conf)
    conf = to_medium_dims_tail(conf)
    return conf


def to_small_dims_heads(conf):
    conf = to_small_dims(conf)
    conf['model']['num_heads'] = [d // 2 for d in conf['model']['num_heads']]
    conf['model']['pretrained_url'] = None
    return conf


def to_small_ultimate(conf):
    conf = to_small_dims_heads(conf)
    conf = to_small_depths(conf)
    conf['model']['depths'] = conf['model']['depths'][0:-3]
    conf['model']['strict'] = False
    return conf


def to_small_ultimate_tail(conf):
    conf = to_small_dims_heads(conf)
    conf = to_small_depths_tail(conf)
    return conf


def add_config(to, base_name, pattern, repl, base_conf, method):
    name_small = re.sub(pattern, repl, base_name)
    conf_small = method(base_conf)
    to[name_small] = conf_small


def small_adder(to, pattern, repl_fmt):
    def small_add(base_name, base_conf):
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepths', base_conf, to_small_depths)
        add_config(to, base_name, pattern, repl_fmt % 'SmallTail', base_conf, to_small_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDims', base_conf, to_small_dims)
        add_config(to, base_name, pattern, repl_fmt % 'MediumDims', base_conf, to_medium_dims)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsTail', base_conf, to_small_depths_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDimsTail', base_conf, to_small_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'MediumDimsTail', base_conf, to_medium_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsDimsTail', base_conf, to_small_depths_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsMediumDimsTail', base_conf,
                   to_small_depths_medium_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'UltimateSmall', base_conf, to_small_ultimate)
        add_config(to, base_name, pattern, repl_fmt % 'UltimateSmallTail', base_conf, to_small_ultimate_tail)

    return small_add


def small_adder_for_cq(to, pattern, repl_fmt):
    def small_add(base_name, base_conf):
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepths', base_conf, to_small_depths)
        add_config(to, base_name, pattern, repl_fmt % 'SmallTail', base_conf, to_small_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsTail', base_conf, to_small_depths_tail)

    return small_add


config_small = {}
for base_name, base_conf in list(config.items()):
    small_adder(config_small, 'VRT_', 'VRT%s_')(base_name, base_conf)


def to_distribution(conf):
    conf = json.loads(json.dumps(conf))
    return conf


def to_distribution_v2(conf):
    conf = json.loads(json.dumps(conf))
    conf['wrapper']['name'] = 'fogsr.trainer.VRTDLightningWrapper'
    conf['wrapper']['blocks'] = dict(
        branch=['branch1', 'branch2', 'branch3', 'branch4'],
        gather=['gather']
    )
    return conf


def to_distribution_v2q(conf):
    conf = to_distribution_v2(conf)
    return conf


def to_distribution_v2cq(conf):
    conf = to_distribution_v2q(conf)
    conf['model']['compress_on_dim'] = 1
    conf['model']['compressed_dims'] = 48
    return conf


def to_distribution_v2dcq(conf):
    conf = to_distribution_v2q(conf)
    conf['model']['compressed_dims'] = 48
    conf['model']['compress_kernel_size'] = 3
    return conf


def distribution_adder(to, pattern, repl_fmt):
    def distribution_add(base_name, base_conf):
        add_config(to, base_name, pattern, repl_fmt % 'D', base_conf, to_distribution)
        small_adder(to, pattern, repl_fmt % 'D%s')(base_name, to_distribution(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv2', base_conf, to_distribution_v2)
        small_adder(to, pattern, repl_fmt % 'Dv2%s')(base_name, to_distribution_v2(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv2Q', base_conf, to_distribution_v2q)
        small_adder(to, pattern, repl_fmt % 'Dv2Q%s')(base_name, to_distribution_v2q(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv2CQ', base_conf, to_distribution_v2cq)
        small_adder_for_cq(to, pattern, repl_fmt % 'Dv2CQ%s')(base_name, to_distribution_v2cq(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv2DCQ', base_conf, to_distribution_v2dcq)
        small_adder_for_cq(to, pattern, repl_fmt % 'Dv2DCQ%s')(base_name, to_distribution_v2dcq(base_conf))

    return distribution_add


config_distribution = {}
for base_name, base_conf in list(config.items()):
    distribution_adder(config_distribution, r"VRT_", "VRT%s_")(base_name, base_conf)


def to_distribution_v3(conf):
    conf = json.loads(json.dumps(conf))
    conf['wrapper']['name'] = 'fogsr.trainer.VRTDLightningWrapper'
    del conf['model']['depths']
    conf['model']['depths_v3'] = [8, [4, 4], [3, 3, 2], [2, 2, 2, 2], 8, 8, 8, 4, 4, 4, 4, 4, 4]
    conf['model']['mul_attn_ratio_v3'] = [[0.5, 0.5], [0.34, 0.34, 0.5], [0.5, 0.5, 0.5, 0.5]]
    conf['wrapper']['blocks'] = dict(
        branch=['branch1', 'branch2', 'branch3', 'branch4'],
        gather=['gather']
    )
    return conf


def to_distribution_v3q(conf):
    conf = to_distribution_v3(conf)
    return conf


def to_distribution_v3cq(conf):
    conf = to_distribution_v3q(conf)
    conf['model']['compress_on_dim'] = 1
    conf['model']['compressed_dims'] = 48
    return conf


def to_distribution_v3dcq(conf):
    conf = to_distribution_v3q(conf)
    conf['model']['compressed_dims'] = 48
    conf['model']['compress_kernel_size'] = 3
    return conf


def to_distribution_v3_small_depths(conf):
    conf = json.loads(json.dumps(conf))
    conf['model']['depths_v3'] = [4, [2, 2], [2, 2, 2], [2, 2, 2, 2], 4, 4, 4, 2, 2, 2, 2, 2, 2]
    conf['model']['mul_attn_ratio_v3'] = [[0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    conf['model']['strict'] = False
    return conf


def to_distribution_v3_small_tail(conf):
    conf = json.loads(json.dumps(conf))
    conf['model']['depths_v3'] = conf['model']['depths_v3'][0:-5]
    conf['model']['strict'] = False
    return conf


def to_distribution_v3_small_depths_tail(conf):
    conf = to_distribution_v3_small_depths(conf)
    conf = to_distribution_v3_small_tail(conf)
    return conf


def to_distribution_v3_small_dims_tail(conf):
    conf = to_small_dims(conf)
    conf = to_distribution_v3_small_tail(conf)
    return conf


def to_distribution_v3_medium_dims_tail(conf):
    conf = to_medium_dims(conf)
    conf = to_distribution_v3_small_tail(conf)
    return conf


def to_distribution_v3_small_depths_dims_tail(conf):
    conf = to_distribution_v3_small_depths(conf)
    conf = to_distribution_v3_small_dims_tail(conf)
    return conf


def to_distribution_v3_small_depths_medium_dims_tail(conf):
    conf = to_distribution_v3_small_depths(conf)
    conf = to_distribution_v3_medium_dims_tail(conf)
    return conf


def to_small_ultimate_v3(conf):
    conf = to_small_dims_heads(conf)
    conf = to_distribution_v3_small_depths(conf)
    conf['model']['depths_v3'] = conf['model']['depths_v3'][0:-3]
    return conf


def to_small_ultimate_v3_tail(conf):
    conf = to_small_dims_heads(conf)
    conf = to_distribution_v3_small_depths_tail(conf)
    return conf


def small_v3_adder(to, pattern, repl_fmt):
    def small_v3_add(base_name, base_conf):
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepths', base_conf, to_distribution_v3_small_depths)
        add_config(to, base_name, pattern, repl_fmt % 'SmallTail', base_conf, to_distribution_v3_small_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDims', base_conf, to_small_dims)
        add_config(to, base_name, pattern, repl_fmt % 'MediumDims', base_conf, to_medium_dims)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsTail', base_conf,
                   to_distribution_v3_small_depths_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDimsTail', base_conf, to_distribution_v3_small_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'MediumDimsTail', base_conf, to_distribution_v3_medium_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsDimsTail', base_conf,
                   to_distribution_v3_small_depths_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsMediumDimsTail', base_conf,
                   to_distribution_v3_small_depths_medium_dims_tail)
        add_config(to, base_name, pattern, repl_fmt % 'UltimateSmall', base_conf, to_small_ultimate_v3)
        add_config(to, base_name, pattern, repl_fmt % 'UltimateSmallTail', base_conf, to_small_ultimate_v3_tail)

    return small_v3_add


def small_v3_adder_for_cq(to, pattern, repl_fmt):
    def small_v3_add(base_name, base_conf):
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepths', base_conf, to_distribution_v3_small_depths)
        add_config(to, base_name, pattern, repl_fmt % 'SmallTail', base_conf, to_distribution_v3_small_tail)
        add_config(to, base_name, pattern, repl_fmt % 'SmallDepthsTail', base_conf,
                   to_distribution_v3_small_depths_tail)

    return small_v3_add


def distribution_adder_v3(to, pattern, repl_fmt):
    def distribution_add_v3(base_name, base_conf):
        add_config(to, base_name, pattern, repl_fmt % 'Dv3', base_conf, to_distribution_v3)
        small_v3_adder(to, pattern, repl_fmt % 'Dv3%s')(base_name, to_distribution_v3(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv3Q', base_conf, to_distribution_v3q)
        small_v3_adder(to, pattern, repl_fmt % 'Dv3Q%s')(base_name, to_distribution_v3q(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv3CQ', base_conf, to_distribution_v3cq)
        small_v3_adder_for_cq(to, pattern, repl_fmt % 'Dv3CQ%s')(base_name, to_distribution_v3cq(base_conf))
        add_config(to, base_name, pattern, repl_fmt % 'Dv3DCQ', base_conf, to_distribution_v3dcq)
        small_v3_adder_for_cq(to, pattern, repl_fmt % 'Dv3DCQ%s')(base_name, to_distribution_v3dcq(base_conf))

    return distribution_add_v3


config_distribution_v3 = {}
for base_name, base_conf in list(config.items()):
    distribution_adder_v3(config_distribution_v3, r"VRT_", "VRT%s_")(base_name, base_conf)

config = dict(**config, **config_small, **config_distribution, **config_distribution_v3)
for name, conf in config.items():
    exec(f"{name} = conf")
__all__ = list(config.keys())

if __name__ == "__main__":
    from pprint import pprint

    pprint(config)
