import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_toolbelt import losses as ptl
from src.util import (
    get_module,
    set_module,
)


def create_criterion(
        loss_func_name: str,
        loss_func_params: dict
) -> nn.Module:
    """Create a loss function instance.

    Parameters
    ----------
    loss_func_name : str
        The name of the loss function.
    loss_func_params : dict
        The parameters for the loss function.

    Returns
    -------
    torch.nn.Module
        The loss function instance.

    Examples
    --------
    >>> create_criterion(
    ...     "BCEWithLogitsLoss",
    ...     {},
    ... )
    """
    return get_module(
        [nn.functional, nn.modules.loss, ptl],
        loss_func_name,
    )(**loss_func_params)


def create_optimizer(
        optimizer_name: str,
        model_params: dict,
        optimizer_params: dict
) -> optim.Optimizer:
    """Create an optimizer instance.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer.
    model_params : dict
        The model parameters to optimize.
    optimizer_params : dict
        The parameters for the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer instance.

    Examples
    --------
    >>> create_optimizer(
    ...     "Adam",
    ...     model.parameters(),
    ...     {"lr": 1e-3},
    ... )
    """
    return get_module(
        [optim],
        optimizer_name,
    )(
        model_params,
        **optimizer_params
    )


def create_scheduler(
        scheduler_name: str,
        optimizer: optim.Optimizer,
        scheduler_params: dict
) -> optim.lr_scheduler._LRScheduler:
    """Create a learning rate scheduler instance.

    Parameters
    ----------
    scheduler_name : str
        The name of the scheduler.
    optimizer : torch.optim.Optimizer
        The optimizer instance.
    scheduler_params : dict
        The parameters for the scheduler.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        The scheduler instance.

    Examples
    --------
    >>> create_scheduler(
    ...     "ReduceLROnPlateau",
    ...     optimizer,
    ...     {"mode": "min", "factor": 0.1, "patience": 5, "threshold": 0.0001},
    ... )
    """
    return get_module(
        [optim.lr_scheduler],
        scheduler_name,
    )(optimizer, **scheduler_params)


class BaseModule(L.LightningModule):
    """Base class for PyTorch Lightning modules.
    基本的には自作のモデルクラスにおいて、このクラスを継承して作成する。

    Parameters
    ----------
    callback_configs : list[dict] | None
        The configuration for the callbacks.

    Attributes
    ----------
    model : torch.nn.Module
        The model instance.
    loss_func : torch.nn.Module
        The loss function instance.
    optimizer : torch.optim.Optimizer
        The optimizer instance.
    lr_scheduler_config : dict
        The configuration for the learning rate scheduler.
    callback_configs : list[dict]
        The configuration for the callbacks.
    """

    def __init__(
            self,
            callback_configs: list[dict] | None = None
    ) -> None:
        super().__init__()
        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.lr_scheduler_config = None
        self.callback_configs = (
            callback_configs if callback_configs is not None else {
                'early_stopping': {
                    'name': 'EarlyStopping',
                    'params': {
                        'monitor': 'val_loss',
                        'mode': 'min',
                        'patience': 5,
                    }
                },
                'model_checkpoint': {
                    'name': 'ModelCheckpoint',
                    'params': {
                        'monitor': 'val_loss',
                        'save_top_k': 1,
                    }
                }
            }
        )

    def set_training_step(
            self,
            loss_func_name: str,
            loss_func_params: dict,
            optimizer_name: str,
            optimizer_params: dict,
            scheduler_name: str,
            scheduler_params: dict,
    ) -> None:
        self.loss_func = create_criterion(
            loss_func_name,
            loss_func_params,
        )
        self.optimizer = create_optimizer(
            optimizer_name,
            self.model.parameters(),
            optimizer_params,
        )
        lr_scheduler = create_scheduler(
            scheduler_name,
            self.optimizer,
            scheduler_params,
        )

        self.lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be "step".
            # "epoch" updates the scheduler on epoch end whereas "step"
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified "monitor"
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler_config,
        }

    def configure_callbacks(self) -> list:
        if self.callback_configs is None:
            # デフォルトのコールバック設定
            self.callback_configs = {
                'early_stopping': {
                    'name': 'EarlyStopping',
                    'params': {
                        'monitor': 'val_loss',
                        'mode': 'min',
                        'patience': 5,
                    }
                },
                'model_checkpoint': {
                    'name': 'ModelCheckpoint',
                    'params': {
                        'monitor': 'val_loss',
                        'save_top_k': 1,
                    }
                }
            }

        callbacks = []
        for key in self.callback_configs:
            callback = set_module(
                groups=[L.pytorch.callbacks],
                config=self.callback_configs,
                key=key,
            )
            callbacks.append(callback)
        return callbacks

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log(
            "train_loss",       # metiricsの名前
            loss,               # metiricsの値
            on_step=True,       # stepごとに表示
            on_epoch=True,      # epochごとに表示
            prog_bar=True,      # プログレスバーに表示
            logger=True,        # ログを保存
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
