import os
from collections.abc import Callable

import cv2
import lightning as L
import numpy as np
import polars as pl
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)
from .Transforms import get_transforms


class SegmentationDataset(Dataset):
    def __init__(
            self,
            features_dir: str,
            labels_dir: str,
            bands: list[str],
            transform: Callable | None = None,
    ) -> None:
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

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
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
        mask = torch.from_numpy(mask).float()

        # ラベルにチャンネル次元を追加 (1, 高さ, 幅)
        mask = mask.unsqueeze(0)
        return image, mask


class SegmentationDataModule(L.LightningModule):
    def __init__(
            self,
            data_dir: str,
            bands: list[str],
            gamma: float = 1.0,
            batch_size: int = 8,
            num_workers: int = 4,
            transform: Callable | None = None,
            train_holds: list[int] = [0, 1, 2, 3, 4, 5, 6],
            val_holds: list[int] = [7, 8],
            test_holds: list[int] = [9],
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.train_holds = train_holds
        self.val_holds = val_holds
        self.test_holds = test_holds

    def setup(self, stage: str | None = None) -> None:
        # interim/train_metadata.csvが存在しない場合には下記を実行
        if not os.path.exists(os.path.join(self.data_dir, "interim", "train_metadata.csv")):
            # メタデータの読み込み
            # リークが発生しないようにデータの分割を行う
            # ここでは、locationとdatetimeの組み合わせでグループ化し、
            # それぞれのグループに対してレコード数をカウントし、
            # 昇順に0~9のindexをholdとして新たなカラムを追加
            df = pl.read_csv(
                os.path.join(self.data_dir, "raw", "train_metadata.csv")
            )
            df.join(
                df
                .group_by(
                    ["location", "datetime"]
                )
                .len()
                .sort("len")
                .with_row_index(name="hold")
                .with_columns(
                    pl.col("hold") % 10
                ),
                on=["location", "datetime"]
            ).write_csv(
                os.path.join(self.data_dir, "interim", "train_metadata.csv")
            )
        id2hold_dict = {
            id: hold for id, hold in
            pl.read_csv(
                os.path.join(self.data_dir, "interim", "train_metadata.csv")
            )
            .select(["chip_id", "hold"])
            .to_numpy()
        }

        features_dir = os.path.join(self.data_dir, "raw", "data", "train_features")
        labels_dir = os.path.join(self.data_dir, "raw", "data", "train_labels")

        # フルデータセットを作成
        full_dataset = SegmentationDataset(
            features_dir=features_dir,
            labels_dir=labels_dir,
            bands=self.bands,
            transform=self.transform,
        )

        # データセットの分割
        # holdがtrain_holdsに含まれるものをtrain、val_holdsに含まれるものをval、test_holdsに含まれるものをtestとする
        # それぞれのサンプルのファイル名を取得
        train_file_name = [k for k, v in id2hold_dict.items() if v in self.train_holds]
        val_file_name = [k for k, v in id2hold_dict.items() if v in self.val_holds]
        test_file_name = [k for k, v in id2hold_dict.items() if v in self.test_holds]
        # ファイル名からインデックスを取得
        train_index = [full_dataset.sample_ids.index(k) for k in train_file_name]
        val_index = [full_dataset.sample_ids.index(k) for k in val_file_name]
        test_index = [full_dataset.sample_ids.index(k) for k in test_file_name]

        # データセットの分割
        self.train_dataset = Subset(full_dataset, train_index)
        self.val_dataset = Subset(full_dataset, val_index)
        self.test_dataset = Subset(full_dataset, test_index)

        # 変換の設定（ガンマ補正を含む）
        self.train_dataset.dataset.transform = get_transforms(train=True, gamma=self.gamma)
        self.val_dataset.dataset.transform = get_transforms(train=False, gamma=self.gamma)
        self.test_dataset.dataset.transform = get_transforms(train=False, gamma=self.gamma)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
