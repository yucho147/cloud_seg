from collections.abc import Callable

import segmentation_models_pytorch as smp
from ._BaseModule import (
    BaseModule,
)


class VanillaUnet(BaseModule):
    """
    任意のパラメータを渡して `smp.Unet` モデルを作成するためのラッパークラス。

    Parameters
    ----------
    encoder_name : str, default="resnet34"
        エンコーダとして使用するモデルの名前。`smp.encoders` に定義されたエンコーダ名から選択します。

    encoder_depth : int, default=5
        エンコーダの深さを指定します。1から5の範囲で設定可能で、エンコーダのステージ数を決定します。

    encoder_weights : str or None, default="imagenet"
        エンコーダの事前学習済み重みを指定します。`imagenet`、`None` などが選択可能です。`None` の場合はランダムな初期化が行われます。

    decoder_channels : tuple of int, default=(256, 128, 64, 32, 16)
        デコーダの各レイヤーのチャンネル数を指定します。タプルで各ステージのチャンネル数を定義します。

    decoder_use_batchnorm : bool or str, default=True
        デコーダでバッチ正規化を使用するかどうかを指定します。`True`、`False`、`"inplace"` のいずれかを選択できます。

    decoder_attention_type : str or None, default=None
        デコーダのブロックに適用する注意機構の種類を指定します。`"scse"`(Squeeze-and-Excitation)や `None`(注意機構を使用しない)を選択できます。

    in_channels : int, default=3
        入力データのチャンネル数を指定します。例えば、RGB画像であれば `3` になります。

    classes : int, default=1
        出力マスクのクラス数を指定します。セグメンテーションするクラスの数です。

    activation : str, callable or None, default=None
        出力に適用するアクティベーション関数を指定します。`"sigmoid"`、`"softmax"`、`callable`、または `None` が使用できます。

    aux_params : dict or None, default=None
        補助的な分類ヘッドを追加するためのパラメータを含む辞書を指定します。`"classes"`、`"dropout"`、`"activation"` などのキーを含めることができます。
    loss_func_name : str, default="BCEWithLogitsLoss"
        損失関数の名前を指定します。`torch.nn.functional` や `src.models._My_Loss` に定義された損失関数名から選択します。
    loss_func_params : dict, default={}
        損失関数のパラメータを指定します。
    optimizer_name : str, default="Adam"
        最適化手法の名前を指定します。`torch.optim` に定義された最適化手法名から選択します。
    optimizer_params : dict, default={"lr": 1e-3}
        最適化手法のパラメータを指定します。
    scheduler_name : str, default="ReduceLROnPlateau"
        学習率スケジューラの名前を指定します。`torch.optim.lr_scheduler` に定義されたスケジューラ名から選択します。
    scheduler_params : dict, default={"mode": "min", "factor": 0.1, "patience": 5, "threshold": 0.0001}
        学習率スケジューラのパラメータを指定します。
    callback_configs : dict or None, default=None
        コールバックの設定を格納した辞書を指定します。

    Attributes
    ----------
    model : smp.Unet
        `segmentation_models_pytorch` の U-Net モデルのインスタンス。
    loss_func : torch.nn.Module
        損失関数のインスタンス。
    optimizer : torch.optim.Optimizer
        最適化手法のインスタンス。
    lr_scheduler_config : dict
        学習率スケジューラの設定を格納した辞書。
    callback_configs : dict or None
        コールバックの設定を格納した辞書。

    Examples
    --------
    >>> import torch
    >>> from src.models import VanillaUnet
    >>> model = VanillaUnet(
    ...     encoder_name="resnet34",
    ...     encoder_depth=5,
    ...     encoder_weights="imagenet",
    ...     decoder_channels=(256, 128, 64, 32, 16),
    ...     decoder_use_batchnorm=True,
    ...     decoder_attention_type=None,
    ...     in_channels=3,
    ...     classes=1,
    ...     activation="sigmoid",
    ...     aux_params=None,
    ...     loss_func_name="BCELoss"",
    ...     loss_func_params={},
    ...     optimizer_name="Adam",
    ...     optimizer_params={"lr": 1e-3},
    ...     scheduler_name="ReduceLROnPlateau",
    ...     scheduler_params={"mode": "min", "factor": 0.1, "patience": 5, "threshold": 0.0001},
    ... )
    >>> x = torch.randn(1, 3, 256, 256)
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 1, 256, 256])
    """

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: str | None = "imagenet",
            decoder_channels: tuple[int] | list[int] = (256, 128, 64, 32, 16),
            decoder_use_batchnorm: bool | str = True,
            decoder_attention_type: str | None = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: str | Callable = "sigmoid",
            aux_params: dict | None = None,
            loss_func_name: str = "BCELoss",
            loss_func_params: dict = {},
            optimizer_name: str = "Adam",
            optimizer_params: dict = {"lr": 1e-3},
            scheduler_name: str = "ReduceLROnPlateau",
            scheduler_params: dict = {
                "mode": "min",
                "factor": 0.1,
                "patience": 5,
                "threshold": 0.0001
            },
            callback_configs: dict | None = None,
    ):
        super().__init__(callback_configs=callback_configs)
        if len(decoder_channels) != encoder_depth:
            raise ValueError(f"The length of decoder_channels ({len(decoder_channels)}) must match encoder_depth ({encoder_depth}).")
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
        self.set_training_step(
            loss_func_name=loss_func_name,
            loss_func_params=loss_func_params,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            scheduler_name=scheduler_name,
            scheduler_params=scheduler_params,
        )
