# YTB-UGC dataset

[Dataset Page](https://media.withyoutube.com/)

## Prepare & Preprocess Dataset

See [yindaheng98/nemoplayer](https://github.com/yindaheng98/nemoplayer#downsample-video)

## Create lmdb

```sh
PYTHONPATH=$(pwd) python fogsr/datasets/ugc/create_lmdb.py 
```