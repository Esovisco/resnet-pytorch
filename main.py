from pathlib import Path

import torch
from torchsummary import summary
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt

from dataLoaders.split1_data_module import Split1DataModule
from dataset import SnacksDataModule
from model import ResNet, AccuracyCallback
from model_resnet_tv import ResNetTv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary

resnet18 = ResNet(in_channels=3, repeat=[2, 2, 2, 2], use_bottleneck=False, outputs=5, optimizer_lr=0.008, scheduler_step=20, scheduler_gamma=0.1)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

# resnet34 = ResNet34(3, outputs=5)
resnet34 = ResNet(in_channels=3, repeat=[3, 4, 6, 3], use_bottleneck=False, outputs=5, optimizer_lr=0.0001, scheduler_step=25, scheduler_gamma=0.1)
# resnet34 = ResNet34.load_from_checkpoint("tb_logs/resnet34/version_14/checkpoints/epoch=49-step=4400.ckpt", in_channels=3, outputs=5)
resnet34.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


resnet34_alt = ResNetTv(optimizer_lr=0.001, scheduler_step=25)
resnet34_alt.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

resnet50_alt = ResNetTv(optimizer_lr=0.005, scheduler_step=200)
resnet50_alt.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

resnet50 = ResNet(in_channels=3, repeat=[3, 4, 6, 3], use_bottleneck=True, outputs=5, optimizer_lr=0.000001, scheduler_step=25, scheduler_gamma=0.1)
resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# resnet101 = ResNet(in_channels=3, repeat=[3, 4, 23, 3], use_bottleneck=True, outputs=5)
# resnet101.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

data_module = Split1DataModule(
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
    name='resnet34_alt'
)

model_checkpoint_callback = ModelCheckpoint(
    dirpath='pl_logs_best_model', monitor='val_acc', save_top_k=3)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=100,
    logger=tensorboard_logger,
    callbacks=[RichModelSummary()]
)

torch.set_float32_matmul_precision('high')

trainer.fit(resnet34_alt, data_module)

# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(resnet34_alt, data_module, attr_name='optimizer_lr')
#
# fig = lr_finder.plot(suggest=True)
# fig.show()
