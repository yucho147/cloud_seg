from glob import glob
from typing import Tuple
import os
import shutil

from torch.utils.data import DataLoader
from torch.nn import Module
import numpy as np
import torch

from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import image_to_tensor, to_numpy


def Large_Fig_TTA(image: np.array,
                  model: Module,
                  tile_size: Tuple[int] = (256, 256),
                  tile_step: Tuple[int] = (128, 128),
                  bool_io: bool = True,
                  output_channels: int = 1,
                  batch_size: int = 8,
                  device="cpu",
                  temp_dir="/dev/temp_fig",
                  fragment_transform=None,
                  transform_mode=None):
    """
    image: でかい画像
    model: torch.Moduleのモデル(input -> outputする関数であれば良い)
    tile_size: 分割された後の画像の大きさ
    tile_step: step幅(基本的にtile_size >= tile_stepとなる)
    bool_io: 分割した画像を一時出力し、メモリの節約をするかのbool
    batch_size: batch_size(メモリの大きさと相談)
    output_channels: 最終的なoutputのchannel数
    device: device
    temp_dir: 一時出力するディレクトリ
    fragment_transform: 入力画像を処理させてmodelに食わせる場合に利用する(今回必ず使う)
        少なくとも下記にある _fragment_transform をラップする形式をお勧めする
    """
    # Cut large image into overlapping tiles
    # ImageSlicer(image_shape, tile_size, tile_step=0, image_margin=0, weight="mean")
    tiler = ImageSlicer(image_shape=image.shape,
                        tile_size=tile_size, tile_step=tile_step)
    target_shape = tiler.target_shape
    weight = tiler.weight
    crops = tiler.crops

    if bool_io:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        # HWC -> CHW. Optionally, do normalization here
        # split(image, border_type=cv2.BORDER_CONSTANT, value=0)
        for n, tile_image in enumerate(tiler.split(image)):
            np.save(os.path.join(temp_dir, f"{n}.npy"), tile_image)
        tiles = _Load_Figures_File(temp_dir)
    else:
        # HWC -> CHW. Optionally, do normalization here
        # split(image, border_type=cv2.BORDER_CONSTANT, value=0)
        tiles = [image_to_tensor(tile) for tile in tiler.split(image)]

    # Allocate a CUDA buffer for holding entire mask
    # merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)
    # Allocate a CPU buffer for holding entire mask
    # TileMerger(image_shape, channels, weight, device="cpu", dtype=torch.float32)
    merger = TileMerger(target_shape, output_channels, weight, device="cpu")

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, crops)),
                                                batch_size=batch_size,
                                                pin_memory=True):
        if fragment_transform:
            # XXX: This shape is (B, C, W, H)
            tiles_batch = fragment_transform(tiles_batch, transform_mode)
        if device == "cpu":
            tiles_batch = tiles_batch.float()
            pred_batch = model(tiles_batch)
        else:
            tiles_batch = tiles_batch.float().cuda()
            model.to(device)
            pred_batch = model(tiles_batch).to('cpu').detach().clone()

        merger.integrate_batch(pred_batch, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    image = np.moveaxis(to_numpy(merger.merge()), 0, -1)
    image = tiler.crop_to_orignal_size(image)
    return image


class _Load_Figures_File(object):
    """Documentation for Load_Figures_File
    """
    def __init__(self, path):
        super().__init__()
        self._path = path
        _files = [
            int(
                os.path.split(i)[-1].split(".")[0]
            ) for i in glob(os.path.join(path, "*"))
        ]
        self._files = [str(i) + ".npy" for i in np.sort(_files)]

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        return image_to_tensor(
            np.asarray(
                np.load(
                    os.path.join(self._path, self._files[idx])
                )
            )
        )


def fragment_transform(batch_image, transform_mode=None):
    if transform_mode is None:
        aggre_0 = np.clip(
            (batch_image[:, 1, :, :] * batch_image[:, 0, :, :]) ** 0.5,
            0., 255.
        )
        aggre_1 = np.clip(
            (batch_image[:, 3, :, :] * batch_image[:, 2, :, :]) ** 0.5,
            0., 255.
        )
    elif transform_mode == "mean":
        aggre_0 = np.clip(
            (batch_image[:, 1, :, :] + batch_image[:, 0, :, :]) * 0.5,
            0., 255.
        )
        aggre_1 = np.clip(
            (batch_image[:, 3, :, :] + batch_image[:, 2, :, :]) * 0.5,
            0., 255.
        )
    image = torch.stack([batch_image[:, 0, :, :],
                         batch_image[:, 1, :, :],
                         aggre_0,

                         batch_image[:, 2, :, :],
                         batch_image[:, 3, :, :],
                         aggre_1], axis=1)
    return image


def _normalize_lumi(arr, max_lumi=1.0, gamma=0.25):
    # XXX: make_datasetとは別の関数であるので注意
    # max_lumi : [0, 1]に規格化した上でclipの上限を定める
    # gamma : gamma補正する
    assert len(arr.shape) == 2
    arr = np.clip((arr / (3200)) ** gamma, None, max_lumi)  # test_max_val = 3180.7551
    arr = np.clip(arr / (max_lumi), 0., 1.)                 # 標準化
    return arr


def normalize_fig(large_image):
    large_image[:, :, 0] = _normalize_lumi(large_image[:, :, 0], max_lumi=0.25, gamma=0.2)  # before_VH
    large_image[:, :, 1] = _normalize_lumi(large_image[:, :, 1], max_lumi=0.3, gamma=0.25)  # before_VV
    large_image[:, :, 2] = _normalize_lumi(large_image[:, :, 2], max_lumi=0.25, gamma=0.2)  # after_VH
    large_image[:, :, 3] = _normalize_lumi(large_image[:, :, 3], max_lumi=0.3, gamma=0.25)  # after_VV
    return large_image


def normalize_simple_fig(large_image):
    large_image[:, :, 0] = _normalize_lumi(large_image[:, :, 0], max_lumi=1.0, gamma=0.07)  # before_VH
    large_image[:, :, 1] = _normalize_lumi(large_image[:, :, 1], max_lumi=1.0, gamma=0.07)  # before_VV
    large_image[:, :, 2] = _normalize_lumi(large_image[:, :, 2], max_lumi=1.0, gamma=0.07)  # after_VH
    large_image[:, :, 3] = _normalize_lumi(large_image[:, :, 3], max_lumi=1.0, gamma=0.07)  # after_VV
    return large_image
