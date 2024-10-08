from pathlib import Path

from lightning import Callback
from lightning import LightningModule
from lightning import Trainer
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class SaveConfigOnStartCallback(Callback):
    """
    PyTorch Lightningのトレーニング開始時にconfigファイルを保存するコールバック.

    Parameters
    ----------
    config : DictConfig
        Omegaconfで読み込んだYAML形式の設定ファイル.
    save_dir : str
        設定ファイルを保存するディレクトリのパス.
    """

    def __init__(self, config: DictConfig, save_dir: str) -> None:
        """
        コンストラクタでconfigと保存ディレクトリを指定.

        Parameters
        ----------
        config : DictConfig
            Omegaconfで読み込んだYAML形式の設定ファイル.
        save_dir : str
            設定ファイルを保存するディレクトリのパス.
        """
        self.config = config
        self.save_dir = save_dir

    def on_train_start(
            self,
            trainer: Trainer,   # noqa: ARG002
            pl_module: LightningModule,   # noqa: ARG002
    ) -> None:
        """
        トレーニング開始時にconfigファイルを保存する.

        Parameters
        ----------
        trainer : Trainer
            PyTorch LightningのTrainerオブジェクト.
        pl_module : LightningModule
            トレーニング中のモデルを表すLightningModuleオブジェクト.
        """
        config_save_path = Path(self.save_dir) / "config.yaml"
        OmegaConf.save(config=self.config, f=config_save_path)
