import lightning as pl
import torch
from torchvision import models
from torch import nn


class ResNetTv(pl.LightningModule):

    def __init__(self, optimizer_lr=0.01, optimizer_momentum=0.9, scheduler_step=7, scheduler_gamma=0.1):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

        self.optimizer_lr = optimizer_lr
        self.optimizer_momentum = optimizer_momentum
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters('optimizer_lr', 'optimizer_momentum', 'scheduler_step', 'scheduler_gamma')

        backbone = models.resnet50(weights=None)

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        num_target_classes = 5
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)

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
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
