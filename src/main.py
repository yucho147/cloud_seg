#!/usr/bin/env python3

from glob import glob
import os

from catalyst import dl
from catalyst.contrib import nn as catnn
from torch import nn as tornn
from pytorch_toolbelt import losses
from catalyst.utils import (
    load_checkpoint,
    unpack_checkpoint,
)
from src import models
from src.util import (
    get_module,
    get_timestamp,
    load_config,
    output_config,
    set_seed,
    setup_logger,
)
from src import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import lightning as L
import click
from torch import optim

data_partition = {
    1: [f"train_{i:02}" for i in [0, 4, 12, 16, 20, 21, 26]],  # fold1
    2: [f"train_{i:02}" for i in [1, 6, 8, 13, 18, 22, 23]],   # fold2
    3: [f"train_{i:02}" for i in [3, 5, 7, 9, 14, 19, 25]],    # fold3
    4: [f"train_{i:02}" for i in [2, 10, 11, 15, 17, 24, 27]],  # fold4
}


# TODO: 全データ学習にする必要がある?
def main(conf: str,
         log: str,
         hold: int,
         suffix: str,
         checkpoint,
         all_train):
    # 保存するファイル名を指定
    log_file = log
    log_dir_name, log_file_name = os.path.split(log_file)
    if suffix:
        log_dir_name = log_dir_name + "-" + suffix
    if all_train:
        log_dir_name = log_dir_name + "-all_train"
    else:
        log_dir_name = log_dir_name + "-" + str(hold)
    log_file = os.path.join(log_dir_name, log_file_name)
    os.makedirs(log_dir_name, exist_ok=True)
    output_config(conf, log_dir_name)
    # ログの初期設定を行う
    logger = setup_logger(log_file)
    # 設定を読み込み
    conf = load_config(conf)
    set_seed(conf.seed)
    if checkpoint:
        logger.info(f"re-train: {checkpoint}")
    logger.info(conf)

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
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, min_lr=1e-6)
    if checkpoint:
        checkpoint = load_checkpoint(path=checkpoint)
        unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
        )

    # data
    data_conf = conf.data
    if all_train:
        images_directory = data_conf.params.images_directory
        train_images_filenames = [
            i.split("/")[-1] for i in glob(os.path.join(images_directory, "*"))
        ]
        transform = get_module([data], data_conf.params.train_transform)()
        train_dataset = get_module([models], data_conf.name)(
            images_filenames=train_images_filenames,
            transform=transform,
            phase="train",
            **data_conf.params
            )
        loaders = {
            "train": DataLoader(
                train_dataset, batch_size=conf.batch_size, shuffle=True
            ),
        }
    else:
        images_directory = data_conf.params.images_directory
        train_images_filenames = [
            i.split("/")[-1] for (k, v) in data_partition.items() if k != hold
            for j in v
            for i in glob(os.path.join(images_directory, f"{j}*"))
        ]
        valid_images_filenames = [
            i.split("/")[-1] for j in data_partition[hold]
            for i in glob(os.path.join(images_directory, f"{j}*"))
        ]
        transform = get_module([data], data_conf.params.train_transform)()
        train_dataset = get_module([models], data_conf.name)(
            images_filenames=train_images_filenames,
            transform=transform,
            phase="train",
            **data_conf.params
        )
        transform = get_module([data], data_conf.params.valid_transform)()
        valid_dataset = get_module([models], data_conf.name)(
            images_filenames=valid_images_filenames,
            transform=transform,
            phase="valid",
            **data_conf.params
        )
        loaders = {
            "train": DataLoader(
                train_dataset, batch_size=conf.batch_size, shuffle=True,
                drop_last=True
            ),
            "valid": DataLoader(
                valid_dataset, batch_size=conf.batch_size
            ),
        }

    class CustomRunner(dl.SupervisedRunner):
        def handle_batch(self, batch):
            x = batch[self._input_key]
            target = batch[self._target_key]
            x_ = self.model(x)
            self.batch = {self._input_key: x, self._output_key: x_,
                          self._target_key: target}

    runner = CustomRunner(
        input_key="images", output_key="outputs", target_key="targets", loss_key="loss"
    )
    # model training
    # TODO: all_train
    if all_train:
        train_conf = conf.train
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=train_conf.params.num_epochs,
            scheduler=scheduler,
            callbacks=[
                dl.BatchTransformCallback(input_key="outputs", output_key="transformed",
                                          scope="on_batch_end", transform="F.sigmoid"),
                dl.IOUCallback(input_key="transformed", target_key="targets"),
                dl.DiceCallback(input_key="transformed", target_key="targets"),
                dl.CheckpointCallback(
                    logdir=log_dir_name,
                    loader_key="train", metric_key="loss", minimize=True
                ),
            ],
            logdir=os.path.join(log_dir_name, "catalyst_files"),
            valid_loader="train",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
    else:
        train_conf = conf.train
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=train_conf.params.num_epochs,
            scheduler=scheduler,
            callbacks=[
                dl.BatchTransformCallback(input_key="outputs", output_key="transformed",
                                          scope="on_batch_end", transform="F.sigmoid"),
                dl.IOUCallback(input_key="transformed", target_key="targets"),
                dl.DiceCallback(input_key="transformed", target_key="targets"),
                dl.CheckpointCallback(
                    logdir=log_dir_name,
                    loader_key="valid", metric_key="loss", minimize=True
                ),
                dl.EarlyStoppingCallback(
                    patience=train_conf.params.patience,
                    loader_key="valid", metric_key="loss", minimize=True
                )
            ],
            logdir=os.path.join(log_dir_name, "catalyst_files"),
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
    logger.info("正常終了")


if __name__ == '__main__':
    main()
