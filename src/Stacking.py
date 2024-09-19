from glob import glob
import os

from PIL import Image
from catalyst import dl
from pytorch_toolbelt.losses import JointLoss, BinaryFocalLoss, DiceLoss
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import albumentations as albu
import numpy as np
import torch
import torch.nn as nn

from src.util import set_seed


class Stack_NN(nn.Module):
    """Documentation for Stack_NN
    """
    def __init__(self, channels=(3, 5, 5, 3)):
        super().__init__()
        self._channels = [1] + [i for j in channels for i in [j, j]]
        self._model = nn.ModuleList([
            layer for i, o in zip(self._channels[:-1], self._channels[1:])
            for layer in [nn.Conv2d(in_channels=i,
                                    out_channels=o,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2),
                                    padding_mode="replicate"),
                          nn.SiLU()]
        ])
        self._model.insert(index=0, module=nn.Sigmoid())
        self._model.append(
            nn.Conv2d(in_channels=self._channels[-1] + 1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      padding_mode="replicate")
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        input_image = x
        for layer in self._model[:-1]:
            x = layer(x)
        x = self._model[-1](torch.cat(
            [input_image, x], dim=1
        ))
        return x


class Stack_Dataset(Dataset):
    """Documentation for Stack_Dataset

    """
    def __init__(self, input_path, anno_path):
        super().__init__()
        self._files = glob(os.path.join(input_path, "*", "*"))
        self._anno_path = anno_path
        self._transform = albu.Compose([
            albu.PadIfNeeded(min_height=512, min_width=512),
            albu.RandomCrop(height=512, width=512),
                                       ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        image = np.load(self._files[idx])
        mask = np.asarray(Image.open(
            os.path.join(self._anno_path, self._files[idx].split("/")[-1].replace(".npy", ".png"))
        ),
                          dtype=np.float32)
        transformed = self._transform(image=image, mask=mask)
        image = transformed["image"][np.newaxis, :, :]
        mask = transformed["mask"][np.newaxis, :, :]
        sample = {"images": image, "targets": mask}
        return sample


class CustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch):
        x = batch[self._input_key]
        target = batch[self._target_key]
        x_ = self.model(x)
        self.batch = {self._input_key: x, self._output_key: x_,
                      self._target_key: target}


def main():
    set_seed(147)
    model = Stack_NN()
    data = Stack_Dataset(
        input_path="./outputs",
        anno_path="../data/raw/train_annotations"
    )
    loaders = {
        "train": DataLoader(
            data, batch_size=32, shuffle=True,
            drop_last=False
        )
    }
    # criterion = nn.BCEWithLogitsLoss()
    criterion = JointLoss(
        DiceLoss(mode="binary"),
        BinaryFocalLoss(alpha=0.1, gamma=2.5),
        0.9, 0.1
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=13, gamma=0.2)
    runner = CustomRunner(
        input_key="images", output_key="outputs", target_key="targets", loss_key="loss"
    )
    runner.train(
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=30,
        callbacks=[
            dl.BatchTransformCallback(input_key="outputs", output_key="transformed",
                                      scope="on_batch_end", transform="F.sigmoid"),
            dl.IOUCallback(input_key="transformed", target_key="targets"),
            dl.DiceCallback(input_key="transformed", target_key="targets"),
            dl.CheckpointCallback(
                logdir="../logs/stacks",
                loader_key="train", metric_key="loss", minimize=True
            ),
        ],
        logdir=os.path.join("../logs/stacks", "catalyst_files"),
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()
