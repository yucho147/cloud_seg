import albumentations as A
import numpy as np


class GammaCorrection(A.ImageOnlyTransform):
    def __init__(self, gamma=1.0, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.gamma = gamma

    def apply(self, img, **params):
        # 0を含むため、適切な範囲にスケーリング
        img = img.astype(np.float32)
        img = np.clip(img, a_min=0, a_max=None)  # 負の値を0に
        max_val = img.max()
        if max_val == 0:
            # 全ての値が0の場合、補正は不要
            return img
        img = img / max_val  # [0, 1]に正規化
        img = img ** self.gamma  # ガンマ補正
        img = img * max_val  # 元のスケールに戻す
        return img


def get_transforms(train: bool = True, gamma: float = 1.0):
    transforms_list = [
        A.Resize(512, 512),
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
