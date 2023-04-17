import math
import os
import torchvision.utils

from fogsr.data import build_dataloader, build_dataset


def ugc_loader(mode='folder',test=False):
    """Test UGC dataset.
    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'UGC'
    opt['type'] = 'UGCDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = '/home/cbj/dataset/ugc-dataset-image/original_videos_h264'
        opt['dataroot_lq'] = '/home/cbj/dataset/ugc-dataset-image/original_videos_h264_x4lossless'
        opt['dataroot_flow'] = None
        opt['meta_info_file'] = 'fogsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = '/home/cbj/dataset/ugc-dataset-lmdb/original_videos_h264/train_GT.lmdb'
        opt['dataroot_lq'] = '/home/cbj/dataset/ugc-dataset-lmdb/original_videos_h264/train_LR.lmdb'
        opt['dataroot_flow'] = None
        opt['meta_info_file'] = 'fogsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='lmdb')

    opt['val_partition'] = 'UGC4'
    opt['num_frame'] = 7
    opt['gt_size'] = 256
    opt['interval_list'] = [1]
    opt['random_reverse'] = True
    opt['use_hflip'] = True
    opt['use_rot'] = True
    opt['test_mode'] = test

    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 1
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    # os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    return data_loader