import os
from typing import Callable, Optional, List

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)
from src.data._Transforms import get_transforms


class SegmentationDataset(Dataset):
    def __init__(
        self,
        features_dir: str,
        labels_dir: str,
        bands: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        セグメンテーション用のデータセット。

        Parameters
        ----------
        features_dir : str
            特徴量（入力画像）のディレクトリパス。
        labels_dir : str
            ラベル（マスク）のディレクトリパス。
        bands : List[str]
            読み込むバンドのファイル名リスト（例：["B02.tif", "B03.tif", "B08.tif"]）。
        transform : Callable, optional
            Albumentationsの変換を適用する関数。
        """
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.bands = bands
        self.transform = transform

        # サンプルIDのリスト（features_dir内のサブディレクトリ名）
        self.sample_ids = [
            name for name in os.listdir(features_dir)
            if os.path.isdir(os.path.join(features_dir, name))
        ]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # バンドごとの画像を読み込み、チャンネル方向にスタック
        band_images = []
        for band in self.bands:
            band_path = os.path.join(self.features_dir, sample_id, band)
            # バンド画像を読み込み
            band_image = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
            if band_image is None:
                raise FileNotFoundError(f"ファイルが見つかりません: {band_path}")
            band_images.append(band_image)

        # チャンネル方向にスタックしてマルチバンド画像を作成
        image = np.stack(band_images, axis=-1)  # 形状：(高さ, 幅, チャンネル数)

        # 対応するラベルを読み込み
        label_path = os.path.join(self.labels_dir, f"{sample_id}.tif")
        mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {label_path}")

        # 変換を適用
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # テンソルに変換
        image = image.transpose(2, 0, 1)  # (チャンネル, 高さ, 幅)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        bands: List[str],
        gamma: float = 1.0,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: Optional[str] = None):
        features_dir = os.path.join(self.data_dir, "raw", "data", "train_features")
        labels_dir = os.path.join(self.data_dir, "raw", "data", "train_labels")

        # フルデータセットを作成
        full_dataset = SegmentationDataset(
            features_dir=features_dir,
            labels_dir=labels_dir,
            bands=self.bands,
            transform=None,
        )

        # データセットの分割
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        test_size = int(total_size * self.test_split)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # 変換の設定（ガンマ補正を含む）
        self.train_dataset.dataset.transform = get_transforms(train=True, gamma=self.gamma)
        self.val_dataset.dataset.transform = get_transforms(train=False, gamma=self.gamma)
        self.test_dataset.dataset.transform = get_transforms(train=False, gamma=self.gamma)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
