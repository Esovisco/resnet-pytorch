import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(dataset_path: Path):
    classes = os.listdir(dataset_path)

    for cls in classes:
        class_path = dataset_path / cls

        class_images = os.listdir(class_path)

        print(len(class_images))

        train_images, test_val_images = train_test_split(class_images, test_size=0.2, random_state=2137)

        test_images, val_images = train_test_split(test_val_images, test_size=0.5, train_size=0.5, random_state=2137)

        print(f'train {len(train_images)} val {len(val_images)} test {len(test_images)}')

        for test_image in test_images:
            shutil.copy(class_path / test_image, Path('assets/split_dataset/test') / test_image)

        for val_image in val_images:
            shutil.copy(class_path / val_image, Path('assets/split_dataset/val') / val_image)

        for train_image in train_images:
            shutil.copy(class_path / train_image, Path('assets/split_dataset/train') / train_image)


split_data(Path('assets/dataset'))
