from PIL import Image
import numpy as np
from config import IMG_SIZE


def load_image(image_file):
    image = Image.open(image_file).convert("L")
    return image

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=(0, -1))


def postprocess_mask(mask, threshold=0.5):
    mask = (mask > threshold).astype(np.uint8)
    mask = mask.squeeze()
    return mask