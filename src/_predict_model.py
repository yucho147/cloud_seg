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
    load_config,
)
from src.models import Large_Fig_TTA, fragment_transform, normalize_fig
import click
import numpy as np
import torch


@click.command()
@click.option('--conf', '-c', help="config file path")
@click.option("--checkpoint", "-cp", help="model check point path")
@click.option("--figs_path", "-f", help="test figs path")
@click.option("--output_path", "-o", help="output file path(.npy)")
def main(conf: str,
         checkpoint: str,
         figs_path: str,
         output_path: str):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    # 設定を読み込み
    conf = load_config(conf)

    # モデルの読み込み
    model_conf = conf.model
    model = get_module([models], model_conf.name)(**model_conf.params)
    criterion_conf = conf.criterion
    criterion = get_module([tornn, catnn, losses],
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

    large_image = _load_test_image(figs_path)[:500, :500, :]

    # TTA
    large_image = normalize_fig(large_image)
    res = []
    for mode in ("normal", "flipud", "fliplr", "flip"):
        if mode == "normal":
            res.append(
                Large_Fig_TTA(image=large_image, model=model,
                              tile_size=(256, 256), tile_step=(128, 128),
                              bool_io=True, output_channels=1,
                              batch_size=8, device=DEVICE, temp_dir="temp",
                              fragment_transform=fragment_transform)
            )
        elif mode == "flipud":
            temp_large_image = np.flipud(large_image)
            res.append(
                np.flipud(
                    Large_Fig_TTA(image=temp_large_image, model=model,
                                  tile_size=(256, 256), tile_step=(128, 128),
                                  bool_io=True, output_channels=1,
                                  batch_size=8, device=DEVICE, temp_dir="temp",
                                  fragment_transform=fragment_transform)
                )
            )
        elif mode == "fliplr":
            temp_large_image = np.fliplr(large_image)
            res.append(
                np.fliplr(
                    Large_Fig_TTA(image=temp_large_image, model=model,
                                  tile_size=(256, 256), tile_step=(128, 128),
                                  bool_io=True, output_channels=1,
                                  batch_size=8, device=DEVICE, temp_dir="temp",
                                  fragment_transform=fragment_transform)
                )
            )
        elif mode == "flip":
            temp_large_image = np.flip(large_image, (0, 1))
            res.append(
                np.flip(
                    Large_Fig_TTA(image=temp_large_image, model=model,
                                  tile_size=(256, 256), tile_step=(128, 128),
                                  bool_io=True, output_channels=1,
                                  batch_size=8, device=DEVICE, temp_dir="temp",
                                  fragment_transform=fragment_transform),
                    (0, 1)
                )
            )
    pred_image = np.mean(res, axis=0)
    w, h, _ = pred_image.shape
    pred_image = pred_image.reshape(w, h)
    np.save(
        output_path,
        pred_image
    )


if __name__ == '__main__':
    main()
