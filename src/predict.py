import numpy as np

from config import IMG_SIZE

from src.preprocess import (
    load_image,
    preprocess_image
)

from src.classifier import predict_pancreas
from src.segmentation import predict_mask


def calculate_area(mask):
    return int(np.sum(mask))


def create_overlay(image_array, mask):

    overlay = image_array.copy()

    overlay = np.stack(
        [overlay, overlay, overlay],
        axis=-1
    )

    overlay[mask == 1] = [255, 0, 0]

    return overlay


def run_pipeline(image_file):

    original_image = load_image(image_file)

    # Preprocess once and use for both models
    processed_image = preprocess_image(
        original_image
    )

    classification = predict_pancreas(
        processed_image
    )

    mask = predict_mask(
        processed_image
    )

    # Create image for overlay visualization
    image_array = np.array(
        original_image.resize(
            (IMG_SIZE, IMG_SIZE)
        )
    )

    overlay = create_overlay(
        image_array,
        mask
    )

    area = calculate_area(mask)

    result = {
        "detected": area > 100,
        "score": classification["score"],
        "mask": mask,
        "area": area,
        "overlay": overlay,
        "original": original_image
    }

    return result