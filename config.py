
from fogsr.data import build_dataloader, build_dataset


def REDS_opt(mode='folder'):
    """Test reds dataset.
    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'REDS'
    opt['type'] = 'REDSRecurrentDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = '/home/cbj/dataset/REDS/train/train_sharp'
        opt['dataroot_lq'] = '/home/cbj/dataset/REDS/train/train_sharp_bicubic'
        opt['dataroot_flow'] = None
        opt['meta_info_file'] = 'fogsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = '/home/cbj/dataset/REDS/train/train_sharp_with_val.lmdb'
        opt['dataroot_lq'] = '/home/cbj/dataset/REDS/train/train_sharp_bicubic_with_val.lmdb'
        opt['dataroot_flow'] = None
        opt['meta_info_file'] = 'fogsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='lmdb')

    opt['val_partition'] = 'official'
    opt['num_frame'] = 7
    opt['gt_size'] = 256
    opt['interval_list'] = [1]
    opt['random_reverse'] = True
    opt['use_hflip'] = True
    opt['use_rot'] = True
    opt['test_mode'] = False
    
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 2
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1
    
    return opt

def REDS_dataloader(mode='folder'):
    opt = REDS_opt(mode=mode)
    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)
    return data_loader
