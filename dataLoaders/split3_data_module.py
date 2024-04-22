from pathlib import Path

import lightning as pl
import numpy.random
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Split3DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size=64):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomEqualize(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        normalized_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        self.train_dataset = ImageFolder(root=str(self.data_dir / 'train'), transform=train_transform)
        self.test_dataset = ImageFolder(root=str(self.data_dir / 'test'), transform=normalized_transform)
        image_folder_val = ImageFolder(root=str(self.data_dir / 'train'), transform=normalized_transform)
        val_len = (len(self.train_dataset) + len(self.test_dataset) * 2) // 10
        self.val_dataset = torch.utils.data.Subset(image_folder_val, numpy.random.choice(len(image_folder_val), val_len, replace=False))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
