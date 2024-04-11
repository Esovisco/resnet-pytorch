from pathlib import Path

import torch
from torchsummary import summary
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

from dataset import SnacksDataModule
from model import ResNet18, ResNet34

resnet18 = ResNet18(3, outputs=5)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

resnet34 = ResNet34(3, outputs=5)
resnet34 = ResNet34.load_from_checkpoint("tb_logs/resnet34/version_12/checkpoints/epoch=49-step=4400.ckpt", in_channels=3, outputs=5)
resnet34.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

data_module = SnacksDataModule(
    data_dir=Path('assets/split_dataset'),
    batch_size=16
)
data_module.setup()

print("Snacks dataset sizes:")
print("Train:", len(data_module.train_dataset))
print("Validation:", len(data_module.val_dataset))
print("Test:", len(data_module.test_dataset))

tensorboard_logger = TensorBoardLogger(
    save_dir='tb_logs',
    name='resnet34'
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=50,
    logger=tensorboard_logger,
)

trainer.fit(resnet34, data_module)
