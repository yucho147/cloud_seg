from pytorch_toolbelt.losses import JointLoss
from pytorch_toolbelt.losses import DiceLoss
from pytorch_toolbelt.losses import BinaryFocalLoss
from torch.nn.modules.loss import _Loss
from torch.nn import BCEWithLogitsLoss


class Dice_BCE(_Loss):
    def __init__(self, weight=[0.5, 0.5]):
        super().__init__()
        criterion_1 = DiceLoss(mode="binary")
        criterion_2 = BCEWithLogitsLoss()
        self._loss = JointLoss(criterion_1, criterion_2, weight[0], weight[1])

    def forward(self, *input):
        return self._loss(*input)


class Dice_Focal(_Loss):
    def __init__(self, weight=[0.5, 0.5]):
        super().__init__()
        criterion_1 = DiceLoss(mode="binary")
        criterion_2 = BinaryFocalLoss()
        self._loss = JointLoss(criterion_1, criterion_2, weight[0], weight[1])

    def forward(self, *input):
        return self._loss(*input)
