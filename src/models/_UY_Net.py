""" A simple UY-Net w/ timm backbone encoder

Based off an old version of Unet in https://github.com/qubvel/segmentation_models.pytorch
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model


class UYnet(nn.Module):
    """UYnet is a fully convolution neural network for change detection

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        center: if ``True`` add ``Conv2dReLU`` block on encoder head

    NOTE: This is based off an old version of Unet in https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
            self,
            encoder_name='efficientnet_b0',
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_chans=6,
            classes=1,
            center=False,
            norm_layer=nn.BatchNorm2d,
            activate=None,
            aggregate=False,
            **kwargs
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different encoder_name indices specified based on the alignment of features
        # and some models won't have a full enough range of feature strides to work properly.
        encoder_before = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=int(in_chans/2),
            pretrained=True, **backbone_kwargs)
        encoder_after = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=int(in_chans/2),
            pretrained=True, **backbone_kwargs)
        self._aggregate = aggregate
        if aggregate:
            encoder_channels = [int(i) for i in encoder_before.feature_info.channels()[::-1]]
        else:
            encoder_channels = [int(i*2) for i in encoder_before.feature_info.channels()[::-1]]

        encoder = nn.ModuleList([encoder_before, encoder_after])

        if not decoder_use_batchnorm:
            norm_layer = None
        decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=classes,
            norm_layer=norm_layer,
            center=center,
        )
        self._model = nn.ModuleDict(
            dict(
                encoder=encoder,
                decoder=decoder
                )
        )
        self._activate = activate

    def forward(self, x: torch.Tensor):
        x_before, x_after = torch.chunk(x, 2, dim=1)
        x_before = self._model.encoder[0](x_before)
        x_after = self._model.encoder[1](x_after)
        x_before.reverse()      # torchscript doesn't work with [::-1]
        x_after.reverse()       # torchscript doesn't work with [::-1]
        if self._aggregate:
            x = [a - b for b, a in zip(x_before, x_after)]
        else:
            x = [torch.cat([b, a], dim=1) for b, a in zip(x_before, x_after)]
        x = self._model.decoder(x)
        if self._activate:
            x = self._activate(x)
        return x


class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=False,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(channels, channels, scale_factor=1.0, norm_layer=norm_layer)
        else:
            self.center = nn.Identity()

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x


class PyramidUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=False,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(channels, channels, scale_factor=1.0, norm_layer=norm_layer)
            self.format_center = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1), padding_mode="replicate")
        else:
            self.center = nn.Identity()
            channels = encoder_channels[0]
            self.format_center = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1), padding_mode="replicate")

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0]
        )]
        out_channels = decoder_channels
        self.up = nn.ModuleList(
            [nn.Upsample(scale_factor=int(2 ** i), mode="bilinear", align_corners=True)
             for i in reversed(range(1, len(in_channels)+1))]
        )
        self.up.append(nn.Identity())

        self.blocks = nn.ModuleList()
        self.format_blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
            self.format_blocks.append(nn.Conv2d(out_chs, 3, kernel_size=(3, 3), stride=(1, 1),
                                                padding=(1, 1), padding_mode="replicate"))

        self.output_conv = nn.Sequential(
            nn.Conv2d((len(out_channels) + 1) * 3, 3, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), padding_mode='replicate'),
            nn.SiLU(),
            nn.Conv2d(3, final_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), padding_mode="replicate")
            )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        out_list = []
        out = self.center(x[0])
        out_list.append(self.up[0](
            self.format_center(out)
        ))
        skips = x[1:]
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(x[1:]) else None
            out = b(out, skip)
            out_list.append(self.up[1 + i](
                self.format_blocks[i](out)
            ))
        x = torch.cat(out_list, dim=1)
        x = self.output_conv(x)
        return x


class PyramidUYnet(nn.Module):
    """PyramidUYnet is a fully convolution neural network for change detection

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        center: if ``True`` add ``Conv2dReLU`` block on encoder head

    NOTE: This is based off an old version of Unet in https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
            self,
            encoder_name='efficientnet_b0',
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_chans=6,
            classes=1,
            center=False,
            norm_layer=nn.BatchNorm2d,
            activate=None,
            **kwargs
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different encoder_name indices specified based on the alignment of features
        # and some models won't have a full enough range of feature strides to work properly.
        encoder_before = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=int(in_chans/2),
            pretrained=True, **backbone_kwargs)
        encoder_after = create_model(
            encoder_name, features_only=True, out_indices=backbone_indices, in_chans=int(in_chans/2),
            pretrained=True, **backbone_kwargs)
        encoder_channels = [int(i*2) for i in encoder_before.feature_info.channels()[::-1]]

        encoder = nn.ModuleList([encoder_before, encoder_after])

        if not decoder_use_batchnorm:
            norm_layer = None
        decoder = PyramidUnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=classes,
            norm_layer=norm_layer,
            center=center,
        )
        self._model = nn.ModuleDict(
            dict(
                encoder=encoder,
                decoder=decoder
                )
        )
        self._activate = activate

    def forward(self, x: torch.Tensor):
        x_before, x_after = torch.chunk(x, 2, dim=1)
        x_before = self._model.encoder[0](x_before)
        x_after = self._model.encoder[1](x_after)
        x_before.reverse()      # torchscript doesn't work with [::-1]
        x_after.reverse()       # torchscript doesn't work with [::-1]
        x = [torch.cat([b, a], dim=1) for b, a in zip(x_before, x_after)]
        x = self._model.decoder(x)
        if self._activate:
            x = self._activate(x)
        return x
