from importlib import import_module


def extract_cls(name: str):
    # 动态导入
    cls = name.split('.')[-1]
    pkg = name[0:-len(cls) - 1]
    init = getattr(import_module(pkg), cls)  # 动态导入
    return init


def extract_cls_conf(data: dict):
    # 动态导入
    if 'name' not in data:
        return None, None
    name = data['name']
    conf = data
    del conf['name']
    init = extract_cls(name)  # 动态导入
    return init, conf


if __name__ == "__main__":
    data = {'name': 'torch.optim.lr_scheduler.CosineAnnealingLR', "a": 'b'}
    print(extract_cls_conf(data))