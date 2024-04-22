from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder


class Split2DataModule(pl.LightningDataModule):
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
            transforms.ToTensor(),

            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            transforms.RandomEqualize(p=0.5),

            transforms.ToTensor()
        ])

        normalized_transform = transforms.Compose([
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.ToTensor()
        ])

        self.train_dataset = ImageFolder(root=str(self.data_dir / 'train'), transform=train_transform)
        self.val_dataset = ImageFolder(root=str(self.data_dir / 'val'), transform=normalized_transform)
        self.test_dataset = ImageFolder(root=str(self.data_dir / 'test'), transform=normalized_transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)

