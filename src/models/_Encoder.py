import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from timm import create_model
from timm.models.efficientnet import default_cfgs
from torch import nn
import torch


def prepare_settings(settings):
    return {
        "mean": settings["mean"],
        "std": settings["std"],
        "url": settings["url"],
        "input_range": (0, 1),
        "input_space": "RGB",
    }


class Six_Ch_One_Fig_Encoder(nn.Module, EncoderMixin):
    """Documentation for Six_Ch_One_Fig_Encoder
    """
    def __init__(
            self,
            stage_idxs=(2, 3, 5),
            out_channels=(6, 48, 32, 56, 160, 448),
            model_name: str = "efficientnet_b3",
            depth: int = 5,
            **kwargs
    ):
        super().__init__()
        self.stage_idxs = stage_idxs  # (2, 3, 5)
        self._depth = depth
        self._in_channels = 6
        self._out_channels = out_channels
        self._m = create_model(model_name,
                               pretrained=False,
                               in_chans=self._in_channels)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._m.conv_stem, self._m.bn1, self._m.act1),
            self._m.blocks[:self.stage_idxs[0]],
            self._m.blocks[self.stage_idxs[0]:self.stage_idxs[1]],
            self._m.blocks[self.stage_idxs[1]:self.stage_idxs[2]],
            self._m.blocks[self.stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        state_dict.pop("conv_stem.weight")
        state_dict.pop("classifier.bias")
        state_dict.pop("classifier.weight")
        self._m.load_state_dict(state_dict, strict=False)


smp.encoders.encoders["six_one_encoder"] = {
    "encoder": Six_Ch_One_Fig_Encoder,
    "pretrained_settings": {
        "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b3"]),
        "advprop": prepare_settings(default_cfgs["tf_efficientnet_b3_ap"]),
        "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b3_ns"]),
    },
    "params": {}
}


class Three_Ch_Two_Fig_Encoder(nn.Module, EncoderMixin):
    """Documentation for Three_Ch_Two_Fig_Encoder
    """
    def __init__(
            self,
            stage_idxs=(2, 3, 5),
            out_channels=(6, 48, 32, 56, 160, 448),
            model_name: str = "efficientnet_b3",
            depth: int = 5,
            **kwargs
    ):
        super().__init__()
        self.stage_idxs = stage_idxs
        self._depth = depth
        self._in_channels = 6
        self._out_channels = out_channels
        self._m = create_model(model_name,
                               pretrained=False,
                               in_chans=3)
        self._m.conv_stem = nn.ModuleList([
            nn.Conv2d(in_channels=self._m.conv_stem.in_channels,
                      out_channels=int(self._m.conv_stem.out_channels/2),
                      kernel_size=self._m.conv_stem.kernel_size,
                      stride=self._m.conv_stem.stride,
                      padding=self._m.conv_stem.padding,
                      bias=self._m.conv_stem.bias
                      ),
            nn.Conv2d(in_channels=self._m.conv_stem.in_channels,
                      out_channels=int(self._m.conv_stem.out_channels/2),
                      kernel_size=self._m.conv_stem.kernel_size,
                      stride=self._m.conv_stem.stride,
                      padding=self._m.conv_stem.padding,
                      bias=self._m.conv_stem.bias
                      )
        ])
        self._m.bn1 = nn.Identity()
        # ModuleList([
        #     nn.BatchNorm2d(num_features=int(self._m.bn1.num_features/2)),
        #     nn.BatchNorm2d(num_features=int(self._m.bn1.num_features/2))
        #     ])
        self._m.classifier = nn.Identity(in_features=self._m.classifier.in_features,
                                         out_features=1)

    def get_stages(self):
        return [
            nn.Identity(),
            (nn.Sequential(self._m.conv_stem[0], self._m.bn1, self._m.act1),
             nn.Sequential(self._m.conv_stem[1], self._m.bn1, self._m.act1)),
            self._m.blocks[:self.stage_idxs[0]],
            self._m.blocks[self.stage_idxs[0]:self.stage_idxs[1]],
            self._m.blocks[self.stage_idxs[1]:self.stage_idxs[2]],
            self._m.blocks[self.stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        x = stages[0](x)
        features.append(x)

        # 分割
        x_1, x_2 = x.chunk(chuncks=2, dim=1)
        x_1 = stages[1][0](x_1)
        x_2 = stages[1][1](x_2)

        # 結合
        x = torch.cat((x_1, x_2), dim=1)
        features.append(x)

        for i in range(2, self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        state_dict.pop("conv_stem.weight")
        # state_dict.pop("conv_stem.1.weight")
        state_dict.pop("classifier.bias")
        state_dict.pop("classifier.weight")
        self._m.load_state_dict(state_dict, strict=False)


smp.encoders.encoders["three_two_encoder"] = {
    "encoder": Three_Ch_Two_Fig_Encoder,
    "pretrained_settings": {
        "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b3"]),
        "advprop": prepare_settings(default_cfgs["tf_efficientnet_b3_ap"]),
        "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b3_ns"]),
    },
    "params": {}
}


class ResNet200dEncoder(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels = [3, 64, 256, 512, 1024, 2048]  # output channels
        self._depth = 5         # UNet depth
        self._in_channels = 3
        self._m = create_model(  # 何らかのモデル
            model_name='resnet200d',
            in_chans=self._in_channels,
            pretrained=False
        )
        self._m.global_pool = nn.Identity()
        self._m.fc = nn.Identity()

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._m.conv1, self._m.bn1, self._m.act1),
            nn.Sequential(self._m.maxpool, self._m.layer1),
            self._m.layer2,
            self._m.layer3,
            self._m.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        self._m.load_state_dict(state_dict, strict=False)
