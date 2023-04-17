from os import path as osp

from fogsr.datasets.lmdb_util import make_lmdb_from_imgs
from fogsr.datasets.ugc.config import (
    large_train_list as train_list,
    large_gt_folder_path as gt_folder_path,
    large_gt_lmdb_path as gt_lmdb_path,
    large_lq_folder_path as lq_folder_path,
    large_lq_lmdb_path as lq_lmdb_path,
)


def create_lmdb_for_ugc():
    """Create lmdb files for YTB-UGC dataset.
        Remember to modify opt configurations according to your settings.
    """
    # # GT
    # folder_path = gt_folder_path
    # lmdb_path = gt_lmdb_path
    # img_path_list, keys = prepare_keys_ugc(folder_path, train_list,
    #                                        'gt')
    # makedirs(osp.dirname(lmdb_path), exist_ok=True)
    # make_lmdb_from_imgs(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # # LQ
    # folder_path = lq_folder_path
    # lmdb_path = lq_lmdb_path
    # img_path_list, keys = prepare_keys_ugc(folder_path, train_list,
    #                                        'lq')
    # makedirs(osp.dirname(lmdb_path), exist_ok=True)
    # make_lmdb_from_imgs(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)