import lightning as L
from loguru import logger
from src import models
from src.models.MyCallback import SaveConfigOnStartCallback
from src.data import dataset as my_dataset
from src.util import (
    get_module,
    load_config,
    set_module,
    set_seed,
)
from typer import Typer

app = Typer()


@app.command()
def cli_run_training(config_path: str) -> None:
    logger.info("start training.")
    run_training(config_path)
    logger.success("Finish training.")


def run_training(config_path: str) -> None:
    # Load config
    config = load_config(config_path)
    logger.info(f"Config: {config}")

    # Set seed
    logger.info("Set seed.")
    logger.info(f"Seed: {config.seed}")
    set_seed(config.seed)

    # set datamodule
    logger.info("Set datamodule.")
    key = "datamodule"
    logger.info(f"name: {config[key]['name']}")
    logger.info(f"params: {config[key]['params']}")
    datamodule: L.LightningDataModule = set_module(
        groups=[my_dataset],
        config=config,
        key=key,
        transform=None,    # 追加のtransformを設定する場合はここに記述
    )

    # set model
    logger.info("Set model.")
    key = "model"
    logger.info(f"name: {config[key]['name']}")
    logger.info(f"params: {config[key]['params']}")
    model: L.LightningModule = set_module(
        groups=[models],
        config=config,
        key=key,
    )

    # set trainer
    logger.info("Set trainer.")
    key = "trainer"
    logger.info(f"name: {config[key]['name']}")
    logger.info(f"params: {config[key]['params']}")
    trainer: L.Trainer = get_module(
        groups=[L],
        name=config[key]["name"],
    )(**config[key]["params"])
    # trainerインスタンスに後からconfigを保存するcallbackを追加
    trainer.callbacks.append(
        SaveConfigOnStartCallback(
            config=config,
            save_dir=trainer.logger.log_dir,
        ),
    )

    # train
    logger.info("Start training.")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    app()
