from pathlib import Path

import torch
from torchsummary import summary
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import SnacksDataModule
from model import ResNet18

resnet18 = ResNet18(3, outputs=1000)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

data_module = SnacksDataModule(
    data_dir=Path('assets/images/data'),
    batch_size=32
)
data_module.setup()

print("Snacks dataset sizes:")
print("Train:", len(data_module.train_dataset))
print("Validation:", len(data_module.val_dataset))
print("Test:", len(data_module.test_dataset))

tensorboard_logger = TensorBoardLogger(
    save_dir='tb_logs',
    name='resnet18'
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=25,
    logger=tensorboard_logger,
)

trainer.fit(resnet18, data_module)
