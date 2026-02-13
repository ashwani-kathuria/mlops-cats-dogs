from PIL import Image
import os
import albumentations as A
import numpy as np

def get_train_augmentation():
    """
    Returns augmentation pipeline for training images.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
    ])
    return transform


def load_and_resize_image(image_path, size=(224, 224)):
    """
    Loads an image and resizes it to required size.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    return img


if __name__ == "__main__":
    # simple test
    sample_path = "data/raw/PetImages/Cat/0.jpg"
    
    if os.path.exists(sample_path):
        img = load_and_resize_image(sample_path)
        print("Image loaded successfully:", img.size)
    else:
        print("Sample image not found")
