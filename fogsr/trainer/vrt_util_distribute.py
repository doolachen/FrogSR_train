import torch
from itertools import product
from .overlap import calculate_overlap_list, calculate_bigger_overlap_list, calculate_smaller_overlap_list


def test_vrt(video, video_branch, model,
             blocks: [str] = ['branch1', 'branch2', 'branch3', 'branch4', 'gather'], **kwargs):
    return test_video_cut_overlap(video, video_branch, model, blocks=blocks, **kwargs)


def new_tensor_like(tensor: torch.Tensor, shape):
    return tensor.reshape(-1)[0].repeat(*shape)


def cat(tensors, dim=0):
    # 由于QuantizedCUDA无法cat，遂用赋值代替之
    shape = list(tensors[0].shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    c = new_tensor_like(tensors[0], shape)
    idx = 0
    slices = [slice(0, None)] * len(shape)
    for tensor in tensors:
        slices[dim] = slice(idx, idx + tensor.shape[dim])
        c[slices] = tensor
        idx += tensor.shape[dim]
    return c


def test_clip(clip_in, clip_in_branch, model, blocks, max_batch_size=16):
    output_clips = []
    for idx in range(0, clip_in.shape[0], max_batch_size):
        clip_i = clip_in[idx: min(idx + max_batch_size, clip_in.shape[0]), ...]
        clip_ib = [b[idx: min(idx + max_batch_size, clip_in.shape[0]), ...] if isinstance(b, torch.Tensor) else b
                   for b in clip_in_branch]
        o_clip = model.forward_split(clip_i, clip_ib, blocks=blocks)
        output_clips.append(o_clip)
    outputs = []
    for o_clip in output_clips:
        outputs.extend([None] * max(0, len(o_clip) - len(outputs)))
        for i, o in enumerate(o_clip):
            if isinstance(o, torch.Tensor):
                if outputs[i] is None:
                    outputs[i] = o
                else:
                    outputs[i] = cat((outputs[i], o), dim=0)
            else:
                outputs[i] = o
    return outputs


def test_video_cut_overlap(video, video_branch, model,
                           blocks: [str] = ['branch1', 'branch2', 'branch3', 'branch4', 'gather'],
                           scale=4,
                           scale_branch=[1, 2, 4, 8],
                           channel_branch=[120, 120, 120, 120],
                           lq_clip=[7, 128, 128], max_batch_size=16, **_):
    b, d, c, h, w = video.size()
    d_in, h_in, w_in = lq_clip
    d_in = min(d_in, d)
    assert h_in <= h and w_in <= w, f"input size {(d, h, w)} not enough large than {(d_in, h_in, w_in)}"
    d_in_idx_list = list(range(0, d - d_in, d_in)) + [max(0, d - d_in)]
    h_in_idx_list = list(range(0, h - h_in, h_in)) + [max(0, h - h_in)]
    w_in_idx_list = list(range(0, w - w_in, w_in)) + [max(0, w - w_in)]
    in_idx_list = product(d_in_idx_list, h_in_idx_list, w_in_idx_list)

    d_out_overlap_idx, d_out_overlap_shape = calculate_overlap_list(d_in_idx_list, d_in)
    h_out_overlap_idx, h_out_overlap_shape = calculate_bigger_overlap_list(h_in_idx_list, h_in, scale)
    w_out_overlap_idx, w_out_overlap_shape = calculate_bigger_overlap_list(w_in_idx_list, w_in, scale)
    out_overlap_idx = product(d_out_overlap_idx, h_out_overlap_idx, w_out_overlap_idx)
    out_overlap_shape = product(d_out_overlap_shape, h_out_overlap_shape, w_out_overlap_shape)
    out_overlap_idx_list = [list(out_overlap_idx)]
    out_overlap_shape_list = [list(out_overlap_shape)]
    for sf in scale_branch:
        h_out_overlap_idx, h_out_overlap_shape = calculate_smaller_overlap_list(h_in_idx_list, h_in, sf)
        w_out_overlap_idx, w_out_overlap_shape = calculate_smaller_overlap_list(w_in_idx_list, w_in, sf)
        out_overlap_idx = product(d_out_overlap_idx, h_out_overlap_idx, w_out_overlap_idx)
        out_overlap_shape = product(d_out_overlap_shape, h_out_overlap_shape, w_out_overlap_shape)
        out_overlap_idx_list.append(list(out_overlap_idx))
        out_overlap_shape_list.append(list(out_overlap_shape))

    E = [torch.zeros(b, d, c, h * scale, w * scale, device=video.device)] + [
        torch.zeros(b, d, ch, h // sf, w // sf, device=video.device) for sf, ch in zip(scale_branch, channel_branch)]

    if video_branch is None:
        video_branch = [None] * len(scale_branch)

    in_idx_list = list(in_idx_list)
    clips_in, clips_in_branch = [], [[] for _ in range(len(scale_branch))]
    for i, (d_in_idx, h_in_idx, w_in_idx) in enumerate(in_idx_list):  # 那么最重要的就是这个idx_list了
        clip_in = video[:,
                  d_in_idx:d_in_idx + d_in, :,
                  h_in_idx:h_in_idx + h_in,
                  w_in_idx:w_in_idx + w_in]
        clip_in_branch = [(
            v[:,
            d_in_idx:d_in_idx + d_in, :,
            h_in_idx // sf:(h_in_idx + h_in) // sf,
            w_in_idx // sf:(w_in_idx + w_in) // sf].transpose(1, 2)
            if isinstance(v, torch.Tensor) else v
        ) for v, sf in zip(video_branch, scale_branch)]
        clips_in.append(clip_in)
        for j, (branch, branches) in enumerate(zip(clip_in_branch, clips_in_branch)):
            if branch is None:
                clips_in_branch[j] = None
            else:
                branches.append(branch)
    clip_in = torch.cat(clips_in, dim=0)
    clip_in_branch = [None] * len(clips_in_branch)
    for i, branch in enumerate(clips_in_branch):
        if branch is None:
            continue
        clip_in_branch[i] = cat(branch, dim=0)
    outputs = test_clip(clip_in, clip_in_branch, model, blocks=blocks, max_batch_size=max_batch_size)
    for i in range(len(in_idx_list)):
        out = [(output[i * b:(i + 1) * b, ...] if isinstance(output, torch.Tensor) else output) for output in outputs]
        for j in range(len(scale_branch) + 1):
            if out[j] is None:
                E[j] = None
                continue
            o = out[j]
            if j > 0:
                o = out[j].transpose(1, 2)  # 这里后面几个分支上的输出是b c d h w, 需要修改
            b_o, d_o, c_o, h_o, w_o = o.shape
            b_e, d_e, c_e, h_e, w_e = E[j].shape
            if E[j].dtype != o.dtype or c_o != c_e:
                E[j] = new_tensor_like(o, (b_e, d_e, c_o, h_e, w_e))
            d_out_overlap_idx, h_out_overlap_idx, w_out_overlap_idx = out_overlap_idx_list[j][i]
            d_out_overlap_shape, h_out_overlap_shape, w_out_overlap_shape = out_overlap_shape_list[j][i]
            E[j][:,
            d_out_overlap_idx[0]:d_out_overlap_idx[1], :,
            h_out_overlap_idx[0]:h_out_overlap_idx[1],
            w_out_overlap_idx[0]:w_out_overlap_idx[1]] = o[:,
                                                         d_out_overlap_shape[0]:d_out_overlap_shape[1], :,
                                                         h_out_overlap_shape[0]:h_out_overlap_shape[1],
                                                         w_out_overlap_shape[0]:w_out_overlap_shape[1]]
    return E
