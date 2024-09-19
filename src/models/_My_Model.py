from segmentation_models_pytorch import (
    DeepLabV3Plus,
    Unet,
    UnetPlusPlus,
)
from torch import nn
import torch


class Six_Ch_One_Fig_EffNet_Unet(nn.Module):
    """Documentation for Six_Ch_One_Fig_EffNet_Unet
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = Unet(encoder_name=encoder_name,
                           encoder_depth=encoder_depth,
                           encoder_weights=encoder_weights,
                           classes=classes,
                           in_channels=6)
        self._activate = activate

    def forward(self, x):
        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x


class Six_Ch_One_Fig_EffNet_Unetpp(nn.Module):
    """Documentation for Six_Ch_One_Fig_EffNet_Unetpp
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = UnetPlusPlus(encoder_name=encoder_name,
                                   encoder_depth=encoder_depth,
                                   encoder_weights=encoder_weights,
                                   classes=classes,
                                   in_channels=6)
        self._activate = activate

    def forward(self, x):
        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x


class Three_Ch_Fig_EffNet_Unetpp(nn.Module):
    """Documentation for Three_Ch_Fig_EffNet_Unetpp
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = UnetPlusPlus(encoder_name=encoder_name,
                                   encoder_depth=encoder_depth,
                                   encoder_weights=encoder_weights,
                                   classes=classes,
                                   in_channels=3)
        self._activate = activate

    def forward(self, x):
        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x


class Three_Ch_Fig_EffNet_Deeplabv3p(nn.Module):
    """Documentation for Three_Ch_Fig_EffNet_Deeplabv3p
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = DeepLabV3Plus(encoder_name=encoder_name,
                                    encoder_depth=encoder_depth,
                                    encoder_weights=encoder_weights,
                                    classes=classes,
                                    in_channels=3)
        self._activate = activate

    def forward(self, x):
        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x


class Six_Ch_One_Fig_EffNet_Deeplabv3p(nn.Module):
    """Documentation for Six_Ch_One_Fig_EffNet_Deeplabv3p
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = DeepLabV3Plus(encoder_name=encoder_name,
                                    encoder_depth=encoder_depth,
                                    encoder_weights=encoder_weights,
                                    classes=classes,
                                    in_channels=6)
        self._activate = activate

    def forward(self, x):
        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x


class Three_Ch_Two_Fig_EffNet_Unet(nn.Module):
    """Documentation for Three_Ch_Two_Fig_EffNet_Unet
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = Unet(encoder_name=encoder_name,
                           encoder_depth=encoder_depth,
                           encoder_weights=encoder_weights,
                           classes=classes,
                           in_channels=6)
        self._head_attachment_1 = nn.Sequential(
            nn.Conv2d(in_channels=int(self._model.encoder.conv_stem.in_channels/2),
                      out_channels=int(self._model.encoder.conv_stem.out_channels/2),
                      kernel_size=self._model.encoder.conv_stem.kernel_size,
                      stride=self._model.encoder.conv_stem.stride,
                      padding=self._model.encoder.conv_stem.padding,
                      bias=self._model.encoder.conv_stem.bias),
            nn.BatchNorm2d(num_features=int(self._model.encoder.conv_stem.out_channels/2))
        )
        self._head_attachment_2 = nn.Sequential(
            nn.Conv2d(in_channels=int(self._model.encoder.conv_stem.in_channels/2),
                      out_channels=int(self._model.encoder.conv_stem.out_channels/2),
                      kernel_size=self._model.encoder.conv_stem.kernel_size,
                      stride=self._model.encoder.conv_stem.stride,
                      padding=self._model.encoder.conv_stem.padding,
                      bias=self._model.encoder.conv_stem.bias),
            nn.BatchNorm2d(num_features=int(self._model.encoder.conv_stem.out_channels/2))
        )
        self._model.encoder.conv_stem = nn.Identity()
        self._model.encoder.bn1 = nn.Identity()
        self._activate = activate

    def forward(self, x):
        # 分割
        x_1, x_2 = x.chunk(2, dim=1)
        x_1 = self._head_attachment_1(x_1)
        x_2 = self._head_attachment_2(x_2)

        # 結合
        x = torch.cat((x_1, x_2), dim=1)

        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x


class Three_Ch_Two_Fig_EffNet_Unetpp(nn.Module):
    """Documentation for Three_Ch_Two_Fig_EffNet_Unetpp
    """
    def __init__(self,
                 encoder_name: str = 'timm-efficientnet-b3',
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 classes: int = 1,
                 activate=None):
        super().__init__()
        self._model = UnetPlusPlus(encoder_name=encoder_name,
                                   encoder_depth=encoder_depth,
                                   encoder_weights=encoder_weights,
                                   classes=classes,
                                   in_channels=6)
        self._head_attachment_1 = nn.Sequential(
            nn.Conv2d(in_channels=int(self._model.encoder.conv_stem.in_channels/2),
                      out_channels=int(self._model.encoder.conv_stem.out_channels/2),
                      kernel_size=self._model.encoder.conv_stem.kernel_size,
                      stride=self._model.encoder.conv_stem.stride,
                      padding=self._model.encoder.conv_stem.padding,
                      bias=self._model.encoder.conv_stem.bias),
            nn.BatchNorm2d(num_features=int(self._model.encoder.conv_stem.out_channels/2))
        )
        self._head_attachment_2 = nn.Sequential(
            nn.Conv2d(in_channels=int(self._model.encoder.conv_stem.in_channels/2),
                      out_channels=int(self._model.encoder.conv_stem.out_channels/2),
                      kernel_size=self._model.encoder.conv_stem.kernel_size,
                      stride=self._model.encoder.conv_stem.stride,
                      padding=self._model.encoder.conv_stem.padding,
                      bias=self._model.encoder.conv_stem.bias),
            nn.BatchNorm2d(num_features=int(self._model.encoder.conv_stem.out_channels/2))
        )
        self._model.encoder.conv_stem = nn.Identity()
        self._model.encoder.bn1 = nn.Identity()
        self._activate = activate

    def forward(self, x):
        # 分割
        x_1, x_2 = x.chunk(2, dim=1)
        x_1 = self._head_attachment_1(x_1)
        x_2 = self._head_attachment_2(x_2)

        # 結合
        x = torch.cat((x_1, x_2), dim=1)

        x = self._model(x)
        if self._activate:
            x = self._activate(x)
        return x
