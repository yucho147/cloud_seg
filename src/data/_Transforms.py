import albumentations as albu


def train_transforms_basic():
    """
    ・RandomRotate90
    ・ShiftScaleRotate
    ・VerticalFlip and HorizontalFlip
    ・RandomGridShuffle
    ・Downscale
    ・GridDistortion
    ・RandomBrightnessContrast
    ・GaussNoise
    ・GaussianBlur
    """
    return albu.Compose(
        [
            albu.RandomRotate90(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                                  rotate_limit=45, border_mode=2, p=0.75),
            albu.VerticalFlip(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.CoarseDropout(max_holes=35, max_height=15, max_width=15, p=0.8),
            albu.RandomGridShuffle(grid=(3, 3), p=0.75),
            albu.Downscale(scale_min=0.35, scale_max=0.90, p=0.8),
            albu.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            albu.GaussNoise(p=0.9, var_limit=(10., 50.)),
            albu.GaussianBlur(blur_limit=(3, 7), p=0.85),
            albu.OneOf([
                albu.CoarseDropout(max_holes=35, max_height=15, max_width=15, p=0.8),
                albu.Downscale(scale_min=0.75, scale_max=0.90, p=1.0),
                albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                albu.GaussNoise(p=1.0, var_limit=(10., 50.)),
            ], p=1.0),
            albu.OneOf([
                albu.CoarseDropout(max_holes=35, max_height=15, max_width=15, p=0.8),
                albu.Downscale(scale_min=0.75, scale_max=0.90, p=1.0),
                albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                albu.GaussNoise(p=1.0, var_limit=(10., 50.)),
            ], p=1.0),
        ]
    )


def valid_transforms_basic():
    return albu.Compose([
    ])
