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
        # self.transform = transforms.Compose(
        #     [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def setup(self, stage=None):
        train_transform = transforms.Compose([

            # transforms.RandomCrop(400, pad_if_needed=True, padding_mode='reflect'),



            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # ], 0.2),
            # transforms.RandomApply([
            #     transforms.RandomRotation(10),
            # ], 0.2),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

        self.train_dataset = ImageFolder(root=str(self.data_dir / 'train'), transform=train_transform)
        self.val_dataset = ImageFolder(root=str(self.data_dir / 'val'), transform=val_transform)
        self.test_dataset = ImageFolder(root=str(self.data_dir / 'test'), transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
