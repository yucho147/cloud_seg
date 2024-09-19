#!/usr/bin/env python3

from attrdict import AttrDict
from datetime import datetime, timezone, timedelta
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
from typing import Optional
import numpy as np
import os
import random
import shutil
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_timestamp():
    timestamp = datetime.now(tz=timezone(timedelta(hours=9), "JST")).strftime(
        "%Y%m%d_%H%M%S"
    )
    return timestamp


def get_module(groups: list, name: Optional[str]):
    if name:
        for group in groups:
            if hasattr(group, name):
                return getattr(group, name)
        raise RuntimeError("Module not found:", name)
    else:
        def return_none(**args):
            return None
        return return_none


def set_module(groups: list, config: dict, key: str, **kwargs):
    conf = config[key]
    name = conf['name']
    params = conf.get('params', {})
    params.update(kwargs)
    return get_module(groups, name)(**params)


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def load_config(config_path: str) -> AttrDict:
    """config(yaml)ファイルを読み込む

    Parameters
    ----------
    config_path : string
        config fileのパスを指定する

    Returns
    -------
    config : attrdict.AttrDict
        configを読み込んでattrdictにしたもの
    """
    with open(config_path, 'r', encoding='utf-8') as fi_:
        return AttrDict(yaml.load(fi_, Loader=yaml.SafeLoader))


def output_config(config_path, output_dir, y=False, prefix=None):
    """configファイルを保存する
    Parameters
    ----------
    config_path : string
        config fileのパスを指定する
    output_dir : string
        出力するディレクトリ
    """
    if prefix:
        output_dir = output_dir + prefix
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = os.listdir(output_dir)
    files = [f for f in files if os.path.isfile(os.path.join(output_dir, f)) and f[0] != '.']
    if files:
        if y:
            inp = 'y'
        else:
            inp = input(f'{output_dir}には既に{files}がありますが、続けますか?[y/n] : ')
    else:
        inp = 'y'
    if inp == 'y':
        shutil.copy(config_path, output_dir)
    else:
        raise ValueError('ファイルがあるので終了した')


def check_device():
    """pytorchが認識しているデバイスを返す関数
    Returns
    -------
    device : str
        cudaを使用する場合 `'cuda'` 、cpuで計算する場合は `'cpu'`
    """
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    return DEVICE


def setup_logger(log_file, modname=__name__):
    """loggerの設定

    Parameters
    ----------
    log_file : string
        出力logファイル

    Returns
    -------
    logger
        logger
    """
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter(('%(asctime)s - '
                           '%(name)s - '
                           '%(levelname)s - '
                           '%(message)s'))
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_file)  # fh = file handler
    fh.setLevel(DEBUG)
    fh_formatter = Formatter(('%(asctime)s - '
                              '%(filename)s - '
                              '%(name)s - '
                              '%(lineno)d - '
                              '%(levelname)s - '
                              '%(message)s'))
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
