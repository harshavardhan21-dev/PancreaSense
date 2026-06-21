import tensorflow as tf

from config import (
    UNET_MODEL_PATH,
    MASK_THRESHOLD
)

from src.preprocess import postprocess_mask

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true_f)
        + tf.reduce_sum(y_pred_f)
        + smooth
    )


unet_model = tf.keras.models.load_model(
    UNET_MODEL_PATH,
    compile=False
)


def predict_mask(image):

    prediction = unet_model.predict(
        image,
        verbose=0
    )

    mask = postprocess_mask(
        prediction,
        MASK_THRESHOLD
    )

    return mask