#!/usr/bin/env python3

from glob import glob
import os

from catalyst.utils import (
    load_checkpoint,
    unpack_checkpoint,
)
from PIL import Image
from catalyst.contrib import nn as catnn
from pytorch_toolbelt import losses
from torch import nn as tornn
from torch import optim
from src import models
from src.util import (
    get_module,
    get_timestamp,
    load_config,
    output_config,
    set_seed,
    setup_logger,
)
from src.models import (
    Large_Fig_TTA,
    fragment_transform,
    normalize_fig,
    normalize_simple_fig,
)
import click
import numpy as np
import torch


@click.command()
@click.option('--conf', '-c', help="config file path")
@click.option('--log', '-l', default='../logs/{0}/output.log'.format(get_timestamp()),
              help="log file path(It is recommend to use the default values.)")
@click.option("--checkpoint", "-cp", help="model check point path")
@click.option("--test_figs_dir", "-f", help="test figs directry")
@click.option("--output_dir", "-o", help="output directry")
@click.option("--normalize_mode", "-nm", default=2, type=int,
              help="normalize_fig: 1/ normalize_simple_fig: 2(default)")
@click.option("--transform_mode", "-tm", default=None,
              help="幾何平均: default / 算術平均: 'mean'")
def main(conf: str,
         log: str,
         checkpoint: str,
         test_figs_dir: str,
         output_dir: str,
         normalize_mode: int,
         transform_mode):
    assert normalize_mode in {1, 2}
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    # 保存するファイル名を指定
    log_file = log
    log_dir_name, log_file_name = os.path.split(log_file)
    os.makedirs(log_dir_name, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    check_date = os.path.split(checkpoint)[0].split("/")[-1]
    output_path = os.path.join(output_dir, check_date)
    os.makedirs(output_path, exist_ok=True)
    # output_config(conf, log_dir_name)
    # ログの初期設定を行う
    # logger = setup_logger(log_file)
    # 設定を読み込み
    conf = load_config(conf)
    set_seed(conf.seed)
    # logger.debug(conf)

    # モデルの読み込み
    model_conf = conf.model
    model = get_module([models], model_conf.name)(**model_conf.params)
    criterion_conf = conf.criterion
    criterion = get_module([tornn, catnn, losses, models],
                           criterion_conf.name)(**criterion_conf.params)
    optimizer_conf = conf.optimizer
    optimizer = get_module([optim], optimizer_conf.name)([
        dict(params=model._model.encoder.parameters(), lr=optimizer_conf.params.encoder_lr),
        dict(params=model._model.decoder.parameters(), lr=optimizer_conf.params.decoder_lr),
    ])
    checkpoint = load_checkpoint(path=checkpoint)
    unpack_checkpoint(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )
    model.eval()

    # データの読み込み
    def _load_test_image(path):
        # test画像4枚をまとめて1枚にする
        # images.shape = (width, hight, ch)
        image_path = np.sort(glob(os.path.join(path, "*")))
        images = []
        for ip in image_path:
            im = np.asarray(Image.open(ip))
            images.append(im)
        return np.stack(images, axis=2)

    def generator_test_images(input_path):
        # testを4chの画像にまとめてしまう
        # 大概input_path = ".../data/raw/test_images/"
        test_path = np.sort(glob(os.path.join(input_path, "*")))
        for t in test_path:
            test_image = _load_test_image(t)
            _, name = os.path.split(t)
            yield test_image, name

    gti = generator_test_images(input_path=test_figs_dir)

    # TTA
    for large_image, name in gti:
        if normalize_mode == 2:
            large_image = normalize_simple_fig(large_image)
        elif normalize_mode == 1:
            large_image = normalize_fig(large_image)
        res = []
        for mode in ("normal", "flipud", "fliplr", "flip"):
            if mode == "normal":
                res.append(
                    Large_Fig_TTA(image=large_image, model=model,
                                  tile_size=(256, 256), tile_step=(64, 64),
                                  bool_io=True, output_channels=1,
                                  batch_size=16, device=DEVICE, temp_dir="temp",
                                  fragment_transform=fragment_transform,
                                  transform_mode=transform_mode)
                )
            elif mode == "flipud":
                temp_large_image = np.flipud(large_image)
                res.append(
                    np.flipud(
                        Large_Fig_TTA(image=temp_large_image, model=model,
                                      tile_size=(256, 256), tile_step=(64, 64),
                                      bool_io=True, output_channels=1,
                                      batch_size=16, device=DEVICE, temp_dir="temp",
                                      fragment_transform=fragment_transform,
                                      transform_mode=transform_mode)
                    )
                )
            elif mode == "fliplr":
                temp_large_image = np.fliplr(large_image)
                res.append(
                    np.fliplr(
                        Large_Fig_TTA(image=temp_large_image, model=model,
                                      tile_size=(256, 256), tile_step=(64, 64),
                                      bool_io=True, output_channels=1,
                                      batch_size=16, device=DEVICE, temp_dir="temp",
                                      fragment_transform=fragment_transform,
                                      transform_mode=transform_mode)
                    )
                )
            elif mode == "flip":
                temp_large_image = np.flip(large_image, (0, 1))
                res.append(
                    np.flip(
                        Large_Fig_TTA(image=temp_large_image, model=model,
                                      tile_size=(256, 256), tile_step=(64, 64),
                                      bool_io=True, output_channels=1,
                                      batch_size=16, device=DEVICE, temp_dir="temp",
                                      fragment_transform=fragment_transform,
                                      transform_mode=transform_mode),
                        (0, 1)
                    )
                )
        pred_image = np.mean(res, axis=0)
        w, h, _ = pred_image.shape
        pred_image = pred_image.reshape(w, h)
        np.save(
            os.path.join(output_path, name + ".npy"),
            pred_image
        )


if __name__ == '__main__':
    main()
