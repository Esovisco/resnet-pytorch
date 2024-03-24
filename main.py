from pathlib import Path

import torch
from torchsummary import summary
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

from dataset import SnacksDataModule
from model import ResNet18

# resnet18 = ResNet18(3, outputs=20)
# resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

data_module = SnacksDataModule(
    data_dir=Path('assets/data'),
    batch_size=1
)
data_module.setup()

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

i = 0
for image, label in iter(data_module.before):
    if i >= 5:
        break

    image = image.squeeze().permute(1, 2, 0)

    axes.flat[i].imshow(image)
    axes.flat[i].set_xticks([])
    axes.flat[i].set_yticks([])
    i += 1

for image, label in iter(data_module.flip_after):
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
