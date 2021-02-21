import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn

from torchvision.models import resnet34 as resnet

# This is a Linear layer having same weights for each input and different biases for each output unit.
class LinearShared(nn.Module):

    def __init__(self, in_units, out_units):
        super().__init__()

        self.w = nn.Parameter(torch.rand(in_units, 1))
        self.b = nn.Parameter(2 * torch.randn(1, out_units))

    def forward(self, x):
        x = x @ self.w
        return self.b + x

# Resnet-34 + 1 FC + 1 FC output with sigmoid activation.
class Coral(LightningModule):

    def __init__(self, weights, resnet_units, fc_units, out_units):
        super().__init__()

        assert weights.size() == (1, out_units)

        self.weights = weights
        self.resnet = resnet(pretrained=True, progress=True)
        self.fc = nn.Linear(resnet_units, fc_units)
        self.out = LinearShared(fc_units, out_units)

        self.cost = nn.BCELoss(reduction='none')

    def forward(self, x):
        x = self.resnet(x)

        x = self.fc(x)
        x = self.out(x)
        return torch.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return torch.mean(self.weights * self.cost(y_hat, y))