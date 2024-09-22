from collections.abc import Callable

import torch.nn as nn
import segmentation_models_pytorch as smp


class VanillaUnet(nn.Module):
    """
    任意のパラメータを渡して `smp.Unet` モデルを作成するためのラッパークラス。

    Parameters
    ----------
    encoder_name : str, default='resnet34'
        エンコーダとして使用するモデルの名前。`smp.encoders` に定義されたエンコーダ名から選択します。

    encoder_depth : int, default=5
        エンコーダの深さを指定します。1から5の範囲で設定可能で、エンコーダのステージ数を決定します。

    encoder_weights : str or None, default='imagenet'
        エンコーダの事前学習済み重みを指定します。`imagenet`、`None` などが選択可能です。`None` の場合はランダムな初期化が行われます。

    decoder_channels : tuple of int, default=(256, 128, 64, 32, 16)
        デコーダの各レイヤーのチャンネル数を指定します。タプルで各ステージのチャンネル数を定義します。

    decoder_use_batchnorm : bool or str, default=True
        デコーダでバッチ正規化を使用するかどうかを指定します。`True`、`False`、`'inplace'` のいずれかを選択できます。

    decoder_attention_type : str or None, default=None
        デコーダのブロックに適用する注意機構の種類を指定します。`'scse'`（Squeeze-and-Excitation）や `None`（注意機構を使用しない）を選択できます。

    in_channels : int, default=3
        入力データのチャンネル数を指定します。例えば、RGB画像であれば `3` になります。

    classes : int, default=1
        出力マスクのクラス数を指定します。セグメンテーションするクラスの数です。

    activation : str, callable or None, default=None
        出力に適用するアクティベーション関数を指定します。`'sigmoid'`、`'softmax'`、`callable`、または `None` が使用できます。

    aux_params : dict or None, default=None
        補助的な分類ヘッドを追加するためのパラメータを含む辞書を指定します。`'classes'`、`'dropout'`、`'activation'` などのキーを含めることができます。

    Attributes
    ----------
    model : smp.Unet
        `segmentation_models_pytorch` の U-Net モデルのインスタンス。

    Examples
    --------
    >>> import torch
    >>> from src.models import VanillaUnet
    >>> model = VanillaUnet(
    ...     encoder_name='resnet34',
    ...     encoder_weights='imagenet',
    ...     in_channels=3,
    ...     classes=1,
    ... )
    >>> input_tensor = torch.randn(1, 3, 256, 256)
    >>> output = model(input_tensor)
    >>> print(output.shape)
    torch.Size([1, 1, 256, 256])
    """

    def __init__(
            self,
            encoder_name: str = 'resnet34',
            encoder_depth: int = 5,
            encoder_weights: str | None = 'imagenet',
            decoder_channels: tuple[int] = (256, 128, 64, 32, 16),
            decoder_use_batchnorm: bool | str = True,
            decoder_attention_type: str | None = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: str | Callable | None = None,
            aux_params: dict | None = None,
    ):
        super(VanillaUnet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)
