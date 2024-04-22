from typing import Any

import lightning as pl
from lightning.pytorch.callbacks import Callback
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import torchvision.models as models


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


class ResBottleneckBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=2 if downsample else 1,
                               padding=1)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1)
        self.shortcut = nn.Sequential()

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(pl.LightningModule):

    def __init__(self, in_channels, repeat, use_bottleneck=False, outputs=1000,
                 optimizer_lr=0.01, optimizer_momentum=0.9, scheduler_step=7, scheduler_gamma=0.1):
        super().__init__()
        self.optimizer_lr = optimizer_lr
        self.optimizer_momentum = optimizer_momentum
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        self.save_hyperparameters('optimizer_lr', 'optimizer_momentum', 'scheduler_step', 'scheduler_gamma')

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.loss = nn.CrossEntropyLoss()

        if use_bottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', ResBottleneckBlock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module('conv2_%d' % (i + 1,), ResBottleneckBlock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', ResBottleneckBlock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module('conv3_%d' % (i + 1,), ResBottleneckBlock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', ResBottleneckBlock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i + 1,), ResBottleneckBlock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', ResBottleneckBlock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d' % (i + 1,), ResBottleneckBlock(filters[4], filters[4], downsample=False))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filters[4], outputs)

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

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        preds = self.forward(imgs)
        loss = self.loss(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.1,
        #     patience=5,
        #     threshold=0.1,
        #     verbose=True,
        #     cooldown=5,
        #     min_lr=1e-8,
        # )
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_acc'}
        # return {'optimizer': optimizer}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class AccuracyCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        print("train_epoch_mean " + epoch_mean)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        pl_module.log("val_epoch_mean", epoch_mean)
