import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def plot_images(image_class):
    image_class_path = Path('assets/images/data/val') / image_class

    images = load_images(image_class_path)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    fig.suptitle('label: ' + image_class, fontsize=24)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(image_class)

    plt.tight_layout()
    plt.show()


def load_images(dir_path: Path):
    images = []
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            images.append(img)
    return images


plot_images('muffin')
