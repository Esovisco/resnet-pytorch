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

<<<<<<< HEAD
    axes.flat[i].imshow(image)
    axes.flat[i].set_xticks([])
    axes.flat[i].set_yticks([])
    i += 1

for image, label in iter(data_module.histogram):
    if i >= 10:
        break

    image = image.squeeze().permute(1, 2, 0)

    axes.flat[i].imshow(image)
    axes.flat[i].set_xticks([])
    axes.flat[i].set_yticks([])
    i += 1

plt.tight_layout()
plt.show()

# print("Snacks dataset sizes:")
# print("Train:", len(data_module.train_dataset))
# print("Validation:", len(data_module.val_dataset))
# print("Test:", len(data_module.test_dataset))
#
# tensorboard_logger = TensorBoardLogger(
#     save_dir='tb_logs',
#     name='resnet18'
# )
#
# trainer = pl.Trainer(
#     accelerator='gpu',
#     devices=1,
#     max_epochs=50,
#     logger=tensorboard_logger,
# )

# trainer.fit(resnet18, data_module)
=======
trainer.fit(resnet34, data_module)
>>>>>>> 8abcfa0ad015 (add val step and start tweaking some hyperparameters)
