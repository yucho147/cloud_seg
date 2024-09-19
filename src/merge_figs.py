from glob import glob
import os

import click
import numpy as np
from PIL import Image

from src.util import get_module
from src import trans


@click.command()
@click.option("--dirs", "-d", multiple=True, help="directries")
@click.option("--weights", "-w", multiple=True, default=None, help="weights")
@click.option("--threshold", "-t", default=None, help="threshold")
@click.option("--output_dir", "-o", help="output directry")
@click.option("--transform", "-tr", default=None, help="if you want transform func")
def main(dirs, weights, threshold, output_dir, transform):
    if threshold is not None:
        threshold = float(threshold)
    if not weights:
        weights = [1/len(dirs) for i in range(len(dirs))]
    else:
        weights = [float(i) for i in weights]
        weights = np.array(weights) / np.sum(weights)
    assert len(weights) == len(dirs)

    os.makedirs(output_dir, exist_ok=True)
    files = [f.split("/")[-1] for f in glob(os.path.join(dirs[0], "*"))]
    for f in files:
        arr = np.sum(
            [trans.sigmoid(np.load(os.path.join(d, f)))*w for d, w in zip(dirs, weights)],
            axis=0
        )
        if threshold:
            arr = (arr > threshold).astype(np.uint8)
            if transform:
                arr = get_module([trans], transform)(arr)
            im = Image.fromarray(arr)
            im.save(
                os.path.join(output_dir, f.replace(".npy", ".png"))
            )
        elif threshold is None:
            np.save(os.path.join(output_dir, f), arr)


if __name__ == '__main__':
    main()
