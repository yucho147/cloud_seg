# 本ファイルはもう使っていない
from glob import glob
import os

import click
import numpy as np
from PIL import Image

from src.util import get_module
from src import trans


@click.command()
@click.option("--input_dir", "-i", type=str, help="input images directry")
@click.option("--output_dir", "-o", type=str, help="output images directry")
@click.option("--threshold", "-th", type=float, help="threshold via sigmoid")
@click.option("--transform", "-tr", default=None, help="if you want transform func")
def main(input_dir, output_dir, threshold, transform):
    os.makedirs(output_dir, exist_ok=True)
    files = glob(os.path.join(input_dir, "*"))
    for f in files:
        arr = np.load(f)
        arr = (arr > threshold).astype(np.uint8)
        if transform:
            arr = get_module([trans], transform)(arr)
        im = Image.fromarray(arr)
        im.save(
            os.path.join(output_dir, f.split("/")[-1].replace(".npy", ".png"))
        )


if __name__ == '__main__':
    main()
