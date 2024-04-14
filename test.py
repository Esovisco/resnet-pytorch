from pathlib import Path

import numpy as np
import torch
import lightning as pl
import matplotlib.pyplot as plt
from dataset import SnacksDataModule
from model import ResNet
import torchvision.transforms as transforms


def index_to_prediction(idx):
    if idx == 0:
        return 'apple'
    if idx == 1:
        return 'banana'
    if idx == 2:
        return 'cookie'
    if idx == 3:
        return 'hot dog'
    if idx == 4:
        return 'muffin'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet18 = ResNet.load_from_checkpoint('tb_logs/resnet18/version_4/checkpoints/epoch=49-step=4400.ckpt', in_channels=3, repeat=[2, 2, 2, 2], use_bottleneck=False, outputs=5, optimizer_lr=0.01)
resnet18.to(device)
resnet18.eval()

data_module = SnacksDataModule(
    data_dir=Path('assets/split_dataset'),
    batch_size=1
)
data_module.setup()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
)

# run inference
random_indices = np.random.choice(len(data_module.test_dataset), size=25, replace=False)
random_samples = [data_module.test_dataset[i] for i in random_indices]

fig, axes = plt.subplots(5, 5, figsize=(15, 15))

for i, (image, label) in enumerate(random_samples):
    dev_image = image.to(device)

    with torch.no_grad():
        output = resnet18(dev_image.unsqueeze(0))

    _, predicted = torch.max(output, 1)

    image_np = transforms.ToPILImage()(image)
    ax = axes[i // 5, i % 5]
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_title(f'{index_to_prediction(predicted)}')

plt.tight_layout()
plt.show()
