from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image
from loguru import logger

app = typer.Typer()


def load_image(image_path: Path) -> np.ndarray:
    logger.info("Loading image...")
    return np.array(Image.open(image_path))


@app.command()
def main(
        features_path: Path,
        labels_path: Path,
        image_index: str,
):
    file_names = ["B02.tif", "B03.tif", "B08.tif"]
    logger.info("Generating plot from data...")
    label_image = load_image(labels_path / (image_index + ".tif"))
    label_colored = np.zeros(
        (label_image.shape[0], label_image.shape[1], 3),
        dtype=np.uint8,
    )
    label_colored[label_image == 1] = [255, 0, 0]
    # uint16の最大値で割り、255倍してuint8に変換
    feature_image = (
        (
            (
                np.stack(
                    [
                        load_image((features_path / image_index) / fine_name)
                        for fine_name in file_names
                    ],
                    axis=2,
                ) / 65535
            ) ** 0.25         # 正規化した上で、0.25乗する(ガンマ補正)
        ) * 255
    ).astype(np.uint8)
    overlay = (feature_image * 255).astype(np.uint8) * 0.7 + label_colored * 0.3

    # 入力画像、ラベル画像、オーバーレイ画像を横に並べて表示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(feature_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(label_colored)
    axes[1].set_title("Label Image")
    axes[1].axis("off")

    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Overlay Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
