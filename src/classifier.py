import tensorflow as tf

from config import (
    CLASSIFIER_MODEL_PATH,
    CLASS_THRESHOLD
)

classifier_model = tf.keras.models.load_model(
    CLASSIFIER_MODEL_PATH,
    compile=False
)


def predict_pancreas(image):

    prediction = classifier_model.predict(
        image,
        verbose=0
    )[0][0]

    pancreas_detected = prediction >= CLASS_THRESHOLD

    return {
        "detected": bool(pancreas_detected),
        "score": float(prediction)
    }