from typing import List
import albumentations as albu
import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    return f, ax


class Just_Transform_Dataset(Dataset):
    """Documentation for Just_Transform_Dataset
    """
    def __init__(self,
                 images_filenames: List[str],
                 images_directory: str,
                 masks_directory: str,
                 transform=None,
                 **kwargs):
        self._images_filenames = images_filenames
        self._images_directory = images_directory
        self._masks_directory = masks_directory
        self._transform = transform

    def __len__(self):
        return len(self._images_filenames)

    def __getitem__(self, idx):
        image_filename = self._images_filenames[idx]
        image = np.asarray(Image.open(os.path.join(self._images_directory,
                                                   image_filename)),
                           dtype=np.float32)
        mask = np.asarray(Image.open(
            os.path.join(self._masks_directory, image_filename)),
                          dtype=np.float32)
        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        before_diff = (image[:, :, 0] + image[:, :, 1])/2
        after_diff = (image[:, :, 2] + image[:, :, 3])/2
        image = np.stack([image[:, :, 0],
                          image[:, :, 1],
                          before_diff,
                          image[:, :, 2],
                          image[:, :, 3],
                          after_diff], axis=2).transpose(2, 0, 1) / 255
        mask = mask[np.newaxis, :, :]
        sample = {'images': image, 'targets': mask}
        return sample


class Tile_and_Just_Transform_Dataset(Dataset):
    """Documentation for Tile_and_Just_Transform_Dataset
    This function is used when train time only.
    """
    def __init__(self,
                 images_filenames: List[str],
                 images_directory: str,
                 masks_directory: str,
                 transform=None,
                 num_of_tiles: int = 8,
                 phase: str = "train",
                 **kwargs):
        self._images_filenames = images_filenames
        self._images_directory = images_directory
        self._masks_directory = masks_directory
        self._transform = transform
        self._num_of_tiles = num_of_tiles
        self._phase = phase

    def __len__(self):
        return len(self._images_filenames)

    def __getitem__(self, idx):
        image_filename = self._images_filenames[idx]
        image = np.asarray(Image.open(os.path.join(self._images_directory,
                                                   image_filename)),
                           dtype=np.uint8)
        mask = np.asarray(Image.open(
            os.path.join(self._masks_directory, image_filename)),
                          dtype=np.uint8)
        image_t = image.copy()
        mask_t = mask.copy()
        if self._phase == "train":
            if np.random.rand() < 0.85:  # 85%の確率でタイル化させる
                image_t, mask_t = self._shuffle_tile(image_t, mask_t, self._num_of_tiles)
        if self._transform is not None:
            transformed = self._transform(image=image_t, mask=mask_t)
            image_t = transformed["image"].astype(np.float32)
            mask_t = transformed["mask"].astype(np.float32)

        vh_diff = np.clip(
            (image_t[:, :, 2] - image_t[:, :, 0])/2. + 128.,
            0., 255.
        )
        vv_diff = np.clip(
            (image_t[:, :, 3] - image_t[:, :, 1])/2. + 128.,
            0., 255.
        )
        vh_avg = (image_t[:, :, 2] + image_t[:, :, 0]) / 2.
        vv_avg = (image_t[:, :, 3] + image_t[:, :, 1]) / 2.

        diff_avg = (vv_diff - vh_diff) / 2. + 128.
        pol_avg = (vh_avg * vv_avg)**0.5

        image_t = np.stack([
            vv_diff,
            vh_diff,
            diff_avg,

            vv_avg,
            vh_avg,
            pol_avg,
        ], axis=2).transpose(2, 0, 1) / 255.
        mask_t = mask_t[np.newaxis, :, :]
        sample = {'images': image_t.astype(np.float32),
                  'targets': mask_t.astype(np.float32)}
        return sample

    def _tiles(self, image, mask, num_of_tiles):
        shape = mask.shape
        assert shape[0] % num_of_tiles == 0
        flag = np.zeros((num_of_tiles, num_of_tiles), dtype=int)
        image_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                       for g in np.vsplit(image, num_of_tiles)]
        mask_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                      for g in np.vsplit(mask, num_of_tiles)]
        for g, mg in enumerate(mask_tiles):
            for r, rg in enumerate(mg):
                flag[g][r] = 1 if rg.any() else 0
        return flag, image_tiles, mask_tiles

    def _shuffle_tile(self, image, mask, num_of_tiles):
        flag, image_tiles, mask_tiles = self._tiles(image, mask, num_of_tiles)
        gyou, retu = np.where(flag)
        image_candidate = np.array([image_tiles[g][r] for g, r in zip(gyou, retu)])
        mask_candidate = np.array([mask_tiles[g][r] for g, r in zip(gyou, retu)])
        length = len(mask_candidate)
        output_image = []
        output_mask = []
        transform = albu.Flip()
        for n in range(num_of_tiles):
            index = np.random.randint(length, size=num_of_tiles)
            temp_image = []
            temp_mask = []
            for m in index:
                transformed = transform(image=image_candidate[m], mask=mask_candidate[m])
                temp_image.append(transformed["image"])
                temp_mask.append(transformed["mask"])
            output_image.append(np.concatenate(temp_image, axis=1))
            output_mask.append(np.concatenate(temp_mask, axis=1))
        return np.concatenate(output_image), np.concatenate(output_mask)


class Tile_and_Just_Transform_Simple_Dataset(Dataset):
    """Documentation for Tile_and_Just_Transform_Dataset
    This function is used when train time only.
    """
    def __init__(self,
                 images_filenames: List[str],
                 images_directory: str,
                 masks_directory: str,
                 transform=None,
                 num_of_tiles: int = 8,
                 phase: str = "train",
                 **kwargs):
        self._images_filenames = images_filenames
        self._images_directory = images_directory
        self._masks_directory = masks_directory
        self._transform = transform
        self._num_of_tiles = num_of_tiles
        self._phase = phase

    def __len__(self):
        return len(self._images_filenames)

    def __getitem__(self, idx):
        image_filename = self._images_filenames[idx]
        image = np.asarray(Image.open(os.path.join(self._images_directory,
                                                   image_filename)),
                           dtype=np.uint8)
        mask = np.asarray(Image.open(
            os.path.join(self._masks_directory, image_filename)),
                          dtype=np.uint8)
        image_t = image.copy()
        mask_t = mask.copy()
        if self._phase == "train":
            if np.random.rand() < 0.85:  # 85%の確率でタイル化させる
                image_t, mask_t = self._shuffle_tile(image_t, mask_t, self._num_of_tiles)
        if self._transform is not None:
            transformed = self._transform(image=image_t, mask=mask_t)
            image_t = transformed["image"].astype(np.float32)
            mask_t = transformed["mask"].astype(np.float32)

        aggre_0 = np.clip(
            (image_t[:, :, 1] * image_t[:, :, 0]) ** 0.5,
            0., 255.
        )
        aggre_1 = np.clip(
            (image_t[:, :, 3] * image_t[:, :, 2]) ** 0.5,
            0., 255.
        )

        image_t = np.stack([
            image_t[:, :, 0],
            image_t[:, :, 1],
            aggre_0,

            image_t[:, :, 2],
            image_t[:, :, 3],
            aggre_1,
        ], axis=2).transpose(2, 0, 1) / 255.
        mask_t = mask_t[np.newaxis, :, :]
        sample = {'images': image_t.astype(np.float32),
                  'targets': mask_t.astype(np.float32)}
        return sample

    def _tiles(self, image, mask, num_of_tiles):
        shape = mask.shape
        assert shape[0] % num_of_tiles == 0
        flag = np.zeros((num_of_tiles, num_of_tiles), dtype=int)
        image_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                       for g in np.vsplit(image, num_of_tiles)]
        mask_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                      for g in np.vsplit(mask, num_of_tiles)]
        for g, mg in enumerate(mask_tiles):
            for r, rg in enumerate(mg):
                flag[g][r] = 1 if rg.any() else 0
        return flag, image_tiles, mask_tiles

    def _shuffle_tile(self, image, mask, num_of_tiles):
        flag, image_tiles, mask_tiles = self._tiles(image, mask, num_of_tiles)
        gyou, retu = np.where(flag)
        image_candidate = np.array([image_tiles[g][r] for g, r in zip(gyou, retu)])
        mask_candidate = np.array([mask_tiles[g][r] for g, r in zip(gyou, retu)])
        length = len(mask_candidate)
        output_image = []
        output_mask = []
        transform = albu.Flip()
        for n in range(num_of_tiles):
            index = np.random.randint(length, size=num_of_tiles)
            temp_image = []
            temp_mask = []
            for m in index:
                transformed = transform(image=image_candidate[m], mask=mask_candidate[m])
                temp_image.append(transformed["image"])
                temp_mask.append(transformed["mask"])
            output_image.append(np.concatenate(temp_image, axis=1))
            output_mask.append(np.concatenate(temp_mask, axis=1))
        return np.concatenate(output_image), np.concatenate(output_mask)


class Tile_and_Just_Transform_Simple_Mean_Dataset(Dataset):
    """Documentation for Tile_and_Just_Transform_Dataset
    This function is used when train time only.
    """
    def __init__(self,
                 images_filenames: List[str],
                 images_directory: str,
                 masks_directory: str,
                 transform=None,
                 num_of_tiles: int = 8,
                 phase: str = "train",
                 **kwargs):
        self._images_filenames = images_filenames
        self._images_directory = images_directory
        self._masks_directory = masks_directory
        self._transform = transform
        self._num_of_tiles = num_of_tiles
        self._phase = phase

    def __len__(self):
        return len(self._images_filenames)

    def __getitem__(self, idx):
        image_filename = self._images_filenames[idx]
        image = np.asarray(Image.open(os.path.join(self._images_directory,
                                                   image_filename)),
                           dtype=np.uint8)
        mask = np.asarray(Image.open(
            os.path.join(self._masks_directory, image_filename)),
                          dtype=np.uint8)
        image_t = image.copy()
        mask_t = mask.copy()
        if self._phase == "train":
            if np.random.rand() < 0.85:  # 85%の確率でタイル化させる
                image_t, mask_t = self._shuffle_tile(image_t, mask_t, self._num_of_tiles)
        if self._transform is not None:
            transformed = self._transform(image=image_t, mask=mask_t)
            image_t = transformed["image"].astype(np.float32)
            mask_t = transformed["mask"].astype(np.float32)

        aggre_0 = np.clip(
            (image_t[:, :, 1] + image_t[:, :, 0]) * 0.5,
            0., 255.
        )
        aggre_1 = np.clip(
            (image_t[:, :, 3] + image_t[:, :, 2]) * 0.5,
            0., 255.
        )

        image_t = np.stack([
            image_t[:, :, 0],
            image_t[:, :, 1],
            aggre_0,

            image_t[:, :, 2],
            image_t[:, :, 3],
            aggre_1,
        ], axis=2).transpose(2, 0, 1) / 255.
        mask_t = mask_t[np.newaxis, :, :]
        sample = {'images': image_t.astype(np.float32),
                  'targets': mask_t.astype(np.float32)}
        return sample

    def _tiles(self, image, mask, num_of_tiles):
        shape = mask.shape
        assert shape[0] % num_of_tiles == 0
        flag = np.zeros((num_of_tiles, num_of_tiles), dtype=int)
        image_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                       for g in np.vsplit(image, num_of_tiles)]
        mask_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                      for g in np.vsplit(mask, num_of_tiles)]
        for g, mg in enumerate(mask_tiles):
            for r, rg in enumerate(mg):
                flag[g][r] = 1 if rg.any() else 0
        return flag, image_tiles, mask_tiles

    def _shuffle_tile(self, image, mask, num_of_tiles):
        flag, image_tiles, mask_tiles = self._tiles(image, mask, num_of_tiles)
        gyou, retu = np.where(flag)
        image_candidate = np.array([image_tiles[g][r] for g, r in zip(gyou, retu)])
        mask_candidate = np.array([mask_tiles[g][r] for g, r in zip(gyou, retu)])
        length = len(mask_candidate)
        output_image = []
        output_mask = []
        transform = albu.Flip()
        for n in range(num_of_tiles):
            index = np.random.randint(length, size=num_of_tiles)
            temp_image = []
            temp_mask = []
            for m in index:
                transformed = transform(image=image_candidate[m], mask=mask_candidate[m])
                temp_image.append(transformed["image"])
                temp_mask.append(transformed["mask"])
            output_image.append(np.concatenate(temp_image, axis=1))
            output_mask.append(np.concatenate(temp_mask, axis=1))
        return np.concatenate(output_image), np.concatenate(output_mask)


class Tile_and_Just_Transform_Dataset_3(Dataset):
    """Documentation for Tile_and_Just_Transform_Dataset_3
    This function is used when train time only.
    """
    def __init__(self,
                 images_filenames: List[str],
                 images_directory: str,
                 masks_directory: str,
                 transform=None,
                 num_of_tiles: int = 8,
                 phase: str = "train",
                 magni: float = 2.0,
                 **kwargs):
        self._images_filenames = images_filenames
        self._images_directory = images_directory
        self._masks_directory = masks_directory
        self._transform = transform
        self._num_of_tiles = num_of_tiles
        self._phase = phase
        self._magni = magni

    def __len__(self):
        return len(self._images_filenames)

    def __getitem__(self, idx):
        image_filename = self._images_filenames[idx]
        image = np.asarray(Image.open(os.path.join(self._images_directory,
                                                   image_filename)),
                           dtype=np.uint8)
        mask = np.asarray(Image.open(
            os.path.join(self._masks_directory, image_filename)),
                          dtype=np.uint8)
        image_t = image.copy()
        mask_t = mask.copy()
        if self._phase == "train":
            if np.random.rand() > (1. - 0.75):  # 75%の確率でタイル化させる
                image_t, mask_t = self._shuffle_tile(image_t, mask_t, self._num_of_tiles)
        if self._transform is not None:
            transformed = self._transform(image=image_t, mask=mask_t)
            image_t = transformed["image"]
            mask_t = transformed["mask"]
        vh_diff = np.clip(
            (image_t[:, :, 2] - image_t[:, :, 0])*self._magni + 128,
            0, 255
        )
        vv_diff = np.clip(
            (image_t[:, :, 3] - image_t[:, :, 1])*self._magni + 128,
            0, 255
        )
        all_diff = np.clip(
            (vv_diff - vh_diff)*self._magni + 128,
            0, 255
        )

        image_t = np.stack([vv_diff, vh_diff, all_diff],
                           axis=2).transpose(2, 0, 1) / 255
        mask_t = mask_t[np.newaxis, :, :]
        sample = {'images': image_t.astype(np.float32),
                  'targets': mask_t.astype(np.float32)}
        return sample

    def _tiles(self, image, mask, num_of_tiles):
        shape = mask.shape
        assert shape[0] % num_of_tiles == 0
        flag = np.zeros((num_of_tiles, num_of_tiles), dtype=int)
        image_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                       for g in np.vsplit(image, num_of_tiles)]
        mask_tiles = [[r.copy() for r in np.hsplit(g.copy(), num_of_tiles)]
                      for g in np.vsplit(mask, num_of_tiles)]
        for g, mg in enumerate(mask_tiles):
            for r, rg in enumerate(mg):
                flag[g][r] = 1 if rg.any() else 0
        return flag, image_tiles, mask_tiles

    def _shuffle_tile(self, image, mask, num_of_tiles):
        flag, image_tiles, mask_tiles = self._tiles(image, mask, num_of_tiles)
        gyou, retu = np.where(flag)
        image_candidate = np.array([image_tiles[g][r] for g, r in zip(gyou, retu)])
        mask_candidate = np.array([mask_tiles[g][r] for g, r in zip(gyou, retu)])
        length = len(mask_candidate)
        output_image = []
        output_mask = []
        transform = albu.Flip()
        for n in range(num_of_tiles):
            index = np.random.randint(length, size=num_of_tiles)
            temp_image = []
            temp_mask = []
            for m in index:
                transformed = transform(image=image_candidate[m], mask=mask_candidate[m])
                temp_image.append(transformed["image"])
                temp_mask.append(transformed["mask"])
            output_image.append(np.concatenate(temp_image, axis=1))
            output_mask.append(np.concatenate(temp_mask, axis=1))
        return np.concatenate(output_image), np.concatenate(output_mask)
