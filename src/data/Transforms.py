import albumentations as A
import numpy as np


# 正規化
class Normalize(A.ImageOnlyTransform):
    def __init__(self, dtype: str, always_apply: bool = True, p: float = 1.0) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.dtype = dtype

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = np.clip(img, a_min=0, a_max=None)
        if self.dtype == "uint16":
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0
        return img


# ガンマ補正
class GammaCorrection(A.ImageOnlyTransform):
    def __init__(self, gamma: float = 1.0, always_apply: bool = True, p: float = 1.0) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.gamma = gamma

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img ** self.gamma  # ガンマ補正
        return img


def get_transforms(train: bool = True, gamma: float = 1.0) -> A.Compose:
    transforms_list = [
        Normalize(dtype="uint16"),
        GammaCorrection(gamma=gamma),
    ]

    if train:
        transforms_list.extend([
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1,
                rotate_limit=45, border_mode=2, p=0.75
            ),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=35, max_height=15, max_width=15, p=0.8),
            A.RandomGridShuffle(grid=(3, 3), p=0.75),
            A.Downscale(scale_min=0.35, scale_max=0.90, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.GaussNoise(p=0.9, var_limit=(10., 50.)),
            A.GaussianBlur(blur_limit=(3, 7), p=0.85),
            A.OneOf([
                A.CoarseDropout(max_holes=35, max_height=15, max_width=15, p=0.8),
                A.Downscale(scale_min=0.75, scale_max=0.90, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.GaussNoise(p=1.0, var_limit=(10., 50.)),
            ], p=1.0),
            A.OneOf([
                A.CoarseDropout(max_holes=35, max_height=15, max_width=15, p=0.8),
                A.Downscale(scale_min=0.75, scale_max=0.90, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.GaussNoise(p=1.0, var_limit=(10., 50.)),
            ], p=1.0),
        ])

    return A.Compose(transforms_list)
