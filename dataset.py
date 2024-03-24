from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder


def load_image(file_path):
    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


class SnacksDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = load_image(self.file_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image


class SnacksDataModule(pl.LightningDataModule):
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

            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            transforms.RandomEqualize(),

            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        self.train_dataset = ImageFolder(root=str(self.data_dir / 'train'), transform=train_transform)
        self.val_dataset = ImageFolder(root=str(self.data_dir / 'val'), transform=val_transform)
        self.test_dataset = ImageFolder(root=str(self.data_dir / 'test'), transform=val_transform)

        # TODO wyjebać poniższe
        color_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])

        rotation_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        crop_transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        self.before = ImageFolder(root=str(self.data_dir / 'test'), transform=transforms.Compose([transforms.ToTensor()]))
        self.colors_after = ImageFolder(root=str(self.data_dir / 'test'), transform=color_transform)
        self.rotation_after = ImageFolder(root=str(self.data_dir / 'test'), transform=rotation_transform)
        self.flip_after = ImageFolder(root=str(self.data_dir / 'test'), transform=flip_transform)
        self.crop_after = ImageFolder(root=str(self.data_dir / 'test'), transform=crop_transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
