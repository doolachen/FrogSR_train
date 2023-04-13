from os import listdir, makedirs
from os import path as osp

from fogsr.datasets.lmdb_util import make_lmdb_from_imgs
from fogsr.datasets.ugc.config import train_list

def create_lmdb_for_ugc():
    """Create lmdb files for YTB-UGC dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # GT
    folder_path = osp.expanduser('~/dataset/ugc-dataset-image/vp9_compressed_videos')
    lmdb_path = osp.expanduser('~/dataset/ugc-dataset-lmdb/vp9_compressed_videos/train_GT.lmdb')
    img_path_list, keys = prepare_keys_ugc(folder_path, train_list,
                                                'gt')
    makedirs(osp.dirname(lmdb_path), exist_ok=True)
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # LQ
    folder_path = osp.expanduser('~/dataset/ugc-dataset-image/vp9_compressed_videos_x4lossless')
    lmdb_path = osp.expanduser('~/dataset/ugc-dataset-lmdb/vp9_compressed_videos/train_LR.lmdb')
    img_path_list, keys = prepare_keys_ugc(folder_path, train_list,
                                                'lq')
    makedirs(osp.dirname(lmdb_path), exist_ok=True)
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_ugc(folder_path, train_list, mode):
    """Prepare image path list and keys for Vimeo90K dataset.

    Args:
        folder_path (str): Folder path.
        train_list (str): Path to the train list.
        mode (str): One of 'gt' or 'lq'.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = []
    keys = []
    for video_folder in train_list:
        img_path_list.extend([osp.join(video_folder, name) for name in listdir(osp.join(folder_path, video_folder))])
    keys = [v.split('.png')[0] for v in img_path_list]
    return img_path_list, keys


if __name__ == '__main__':
    create_lmdb_for_ugc()
