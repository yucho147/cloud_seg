# -*- coding: utf-8 -*-
from glob import glob
import click
import logging
import os
import random

from PIL import Image, ImageOps
import albumentations as albu
import numpy as np
from tqdm import tqdm

seed = 147
random.seed(seed)
np.random.seed(seed)


def generator_crop_image(img, img_anno, size=256, num=100):
    # imageをcropしてnum枚になるまで保存し続ける関数
    # padding if need #########################################################
    pad = albu.PadIfNeeded(p=1.0, min_height=size, min_width=size, border_mode=2)
    output = pad(image=img, mask=img_anno)
    img, img_anno = output["image"], output["mask"]
    # #########################################################################
    img_transform = [
        albu.RandomCrop(size, size),
        # albu.OneOf([
        #     albu.SafeRotate(p=1.0, border_mode=2),
        #     albu.GridDistortion(p=1.0, distort_limit=0.3, border_mode=2),
        #     albu.ElasticTransform(p=1.0, border_mode=2),
        #     albu.OpticalDistortion(p=1.0, distort_limit=0.05, shift_limit=0.05, border_mode=2)
        # ], p=0.5)
    ]
    transforms = albu.Compose(img_transform)
    count = 0
    while count < num:
        output = transforms(image=img, mask=img_anno)
        if output["mask"].sum() > 100:  # 100ピクセルくらいはcrop画像に欲しい
            yield output["image"], output["mask"]
            count += 1


def normalize_lumi(arr, max_lumi=1.0, gamma=1/3,
                   hist_equalize=True, is_hard_normalize=True):
    # max_lumi : [0, 1]に標準化した上でclipの上限を定める
    # gamma : gamma補正する
    assert len(arr.shape) == 2
    arr = np.clip((arr / (3200)) ** gamma, None, max_lumi)  # test_max_val = 3180.7551
    if is_hard_normalize:
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255  # 標準化
    else:
        arr = np.clip(arr / (max_lumi) * 255, 0, 255)  # 標準化
    if hist_equalize:
        img = Image.fromarray(np.uint8(arr))
        arr = np.asarray(ImageOps.equalize(img))
    return arr


def generator_train_anno_images(input_path):
    # trainを4chの画像にまとめてしまう
    # train画像とannotationされた画像をtupleで出力する
    train_path = np.sort(glob(os.path.join(input_path, "train_images", "*")))
    annot_path = np.sort(glob(os.path.join(input_path, "train_annotations", "*")))
    for t, a in zip(train_path, annot_path):
        train_image = load_train_image(t)
        annot_image = load_annot_image(a)
        images = (train_image, annot_image)
        _, name = os.path.split(a)
        yield images, name.split(".")[0]


def load_train_image(path):
    # train画像4枚をまとめて1枚にする
    # load_train_anno_imagesの中で利用する
    # images.shape = (width, hight, ch)
    image_path = np.sort(glob(path + "/*"))
    images = []
    for ip in image_path:
        im = np.asarray(Image.open(ip))
        images.append(im)
    return np.stack(images, axis=2)


def load_annot_image(path):
    # annotationされた画像をreturnする
    # load_train_annot_imagesの中で利用する
    # image.shape = (width, hight)
    image = np.asarray(Image.open(path))
    return image


def check_annotations_image(input_path):
    # 画像を保存しチェックする
    # annotation画像が0 or 1でプレビューで見てもわからんため、本編とは関係なくチェック用の関数を作成
    path = glob(os.path.join(input_path, "*"))
    for p in path:
        im = np.asarray(Image.open(p)) * 255
        im = Image.fromarray(im)
        directry, name = os.path.split(p)
        directry, _ = os.path.split(directry)
        new_path_name = os.path.join(directry, "check_annotations", name)
        im.save(os.path.join(new_path_name))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('num_data', type=int, default=100)
@click.argument('is_three_ch', type=bool, default=False)
def main(input_filepath, output_filepath, num_data: int, is_three_ch):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    Examples:
        $ python make_dataset.py "data/raw" "data/processed"
    """
    logger = logging.getLogger(__name__)
    logger.info('チェック関数を実行')
    os.makedirs(os.path.join(input_filepath, "check_annotations"), exist_ok=True)
    check_annotations_image(os.path.join(input_filepath, "train_annotations"))

    logger.info("必要なディレクトリを作成")
    os.makedirs(os.path.join(output_filepath, "train_data"), exist_ok=True)
    os.makedirs(os.path.join(output_filepath, "mask_data"), exist_ok=True)
    os.makedirs(os.path.join(output_filepath, "check_annotations"), exist_ok=True)

    logger.info('main関数の本題を実施')
    logger.info(f"num_data = {num_data}")
    gtai = generator_train_anno_images(input_filepath)
    for (t, a), name in tqdm(gtai, total=28):
        if not is_three_ch:
            t[:, :, 0] = normalize_lumi(
                t[:, :, 0], max_lumi=0.25, gamma=0.2, hist_equalize=False)  # before_VH
            t[:, :, 1] = normalize_lumi(
                t[:, :, 1], max_lumi=0.3, gamma=0.25, hist_equalize=False)  # before_VV
            t[:, :, 2] = normalize_lumi(
                t[:, :, 2], max_lumi=0.25, gamma=0.2, hist_equalize=False)  # after_VH
            t[:, :, 3] = normalize_lumi(
                t[:, :, 3], max_lumi=0.3, gamma=0.25, hist_equalize=False)  # after_VV
        else:
            t[:, :, 0] = normalize_lumi(
                t[:, :, 0], max_lumi=1.0, gamma=0.07,
                hist_equalize=False, is_hard_normalize=False)  # before_VH
            t[:, :, 1] = normalize_lumi(
                t[:, :, 1], max_lumi=1.0, gamma=0.07,
                hist_equalize=False, is_hard_normalize=False)  # before_VV
            t[:, :, 2] = normalize_lumi(
                t[:, :, 2], max_lumi=1.0, gamma=0.07,
                hist_equalize=False, is_hard_normalize=False)  # after_VH
            t[:, :, 3] = normalize_lumi(
                t[:, :, 3], max_lumi=1.0, gamma=0.07,
                hist_equalize=False, is_hard_normalize=False)  # after_VV

        t = t.astype(np.uint8)

        output_iter = generator_crop_image(t, a, size=256, num=num_data)
        for i, (train_arr, annot_arr) in enumerate(output_iter):
            train_im = Image.fromarray(train_arr.astype(np.uint8))
            annot_im = Image.fromarray(annot_arr.astype(np.uint8))
            check_im = Image.fromarray((annot_arr * 255).astype(np.uint8))
            train_im.save(
                os.path.join(output_filepath, "train_data", name + f'_{i+1}.png')
            )
            annot_im.save(
                os.path.join(output_filepath, "mask_data", name + f'_{i+1}.png')
            )
            check_im.save(
                os.path.join(output_filepath, "check_annotations", name + f'_{i+1}.png')
            )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
