def calculate_overlap_list(idx_list, size):
    range_list = [[idx, (idx + size)] for idx in idx_list]
    overlap_idx = [[idx, (idx + size)] for idx in idx_list]
    overlap_shape = [[0, size] for _ in idx_list]
    for i, ((l_idx_l, l_idx_r), (r_idx_l, r_idx_r)) in enumerate(zip(range_list[0:-1], range_list[1:])):
        overlap_l, overlap_r = l_idx_r, r_idx_l
        if l_idx_r > r_idx_l:
            overlap_l = overlap_r = (l_idx_r + r_idx_l) // 2
        overlap_idx[i][1] = overlap_l
        overlap_idx[i + 1][0] = overlap_r
        overlap_shape[i][1] -= l_idx_r - overlap_l
        overlap_shape[i + 1][0] += overlap_r - r_idx_l
    return overlap_idx, overlap_shape


def calculate_bigger_overlap_list(idx_list, size, scale=1):
    return calculate_overlap_list([idx * scale for idx in idx_list], size * scale)


def calculate_smaller_overlap_list(idx_list, size, scale=1):
    return calculate_overlap_list([idx // scale for idx in idx_list], size // scale)


if __name__ == "__main__":
    idx_list = list(range(0, 100, 10))
    size = 14
    print([[idx, idx + size] for idx in idx_list])
    print([[0, size] for _ in idx_list])
    overlap_idx, overlap_shape = calculate_overlap_list(idx_list, 14)
    print(overlap_idx)
    print(overlap_shape)
    overlap_idx, overlap_shape = calculate_bigger_overlap_list(idx_list, 14, 2)
    print(overlap_idx)
    print(overlap_shape)
    overlap_idx, overlap_shape = calculate_smaller_overlap_list(idx_list, 14, 2)
    print(overlap_idx)
    print(overlap_shape)
