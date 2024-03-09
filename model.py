from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn


class ResBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, downsample: bool):
        super(ResBlock, self).__init__()

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)

        out = out + shortcut

        return nn.ReLU()(out)


class ResNet18(pl.LightningModule):

    def __init__(self, in_channels, outputs=1000):
        super(ResNet18, self).__init__()

        self.loss = nn.CrossEntropyLoss()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False),
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False),
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, outputs)

    def forward(self, input):
        out = self.layer0(input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.gap(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        preds = self.forward(imgs)
        loss = self.loss(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}
