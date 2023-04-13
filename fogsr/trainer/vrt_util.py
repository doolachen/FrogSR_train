import torch
from itertools import product
from .overlap import calculate_overlap_list, calculate_bigger_overlap_list


def test_vrt(video, model, **kwargs):
    return test_video_cut_overlap(video, model, **kwargs)


def test_video_cut_overlap(video, model, scale=4, lq_clip=[7, 128, 128], max_batch_size=16, **_):
    b, d, c, h, w = video.size()
    d_in, h_in, w_in = lq_clip
    d_in = min(d_in, d)
    assert h_in <= h and w_in <= w, f"input size {(d, h, w)} not enough large than {(d_in, h_in, w_in)}"
    d_in_idx_list = list(range(0, d - d_in, d_in)) + [max(0, d - d_in)]
    h_in_idx_list = list(range(0, h - h_in, h_in)) + [max(0, h - h_in)]
    w_in_idx_list = list(range(0, w - w_in, w_in)) + [max(0, w - w_in)]
    in_idx_list = list(product(d_in_idx_list, h_in_idx_list, w_in_idx_list))

    d_out_overlap_idx, d_out_overlap_shape = calculate_overlap_list(d_in_idx_list, d_in)
    h_out_overlap_idx, h_out_overlap_shape = calculate_bigger_overlap_list(h_in_idx_list, h_in, scale)
    w_out_overlap_idx, w_out_overlap_shape = calculate_bigger_overlap_list(w_in_idx_list, w_in, scale)
    out_overlap_idx = list(product(d_out_overlap_idx, h_out_overlap_idx, w_out_overlap_idx))
    out_overlap_shape = list(product(d_out_overlap_shape, h_out_overlap_shape, w_out_overlap_shape))

    E = torch.zeros(b, d, c, h * scale, w * scale, device=video.device)

    clips_in = []
    for i, (d_in_idx, h_in_idx, w_in_idx) in enumerate(in_idx_list):  # 那么最重要的就是这个idx_list了
        clip_in = video[:,
                  d_in_idx:d_in_idx + d_in, :,
                  h_in_idx:h_in_idx + h_in,
                  w_in_idx:w_in_idx + w_in]
        clips_in.append(clip_in)
    clip_in = torch.cat(clips_in, dim=0)
    output = []
    for idx in range(0, clip_in.shape[0], max_batch_size):
        output.append(model(clip_in[idx: min(idx + max_batch_size, clip_in.shape[0]), ...]))
    output = torch.cat(output, dim=0)
    for i in range(len(in_idx_list)):
        out = output[i * b:(i + 1) * b, ...]
        d_out_overlap_idx, h_out_overlap_idx, w_out_overlap_idx = out_overlap_idx[i]
        d_out_overlap_shape, h_out_overlap_shape, w_out_overlap_shape = out_overlap_shape[i]
        E[:,
        d_out_overlap_idx[0]:d_out_overlap_idx[1], :,
        h_out_overlap_idx[0]:h_out_overlap_idx[1],
        w_out_overlap_idx[0]:w_out_overlap_idx[1]] = out[:,
                                                     d_out_overlap_shape[0]:d_out_overlap_shape[1], :,
                                                     h_out_overlap_shape[0]:h_out_overlap_shape[1],
                                                     w_out_overlap_shape[0]:w_out_overlap_shape[1]]
    return E
