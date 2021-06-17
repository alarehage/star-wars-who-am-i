from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_inputs,
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_inputs
import yaml


SCRIPT_PATH = Path(__file__).parent.absolute()

config = yaml.safe_load(open(SCRIPT_PATH / "models/model_cfg.yaml"))

MODEL_NAME = config["model_name"]

if MODEL_NAME == "MobileNetV2":
    preprocess_input = mobilenet_inputs
elif MODEL_NAME == "ResNet50":
    preprocess_input = resnet_inputs

# def load_img(file_path):
#     """
#     load img file
#     Args:
#         file_path: path to file
#     Returns:
#         img_array: array of image
#     """
#     img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))

#     return img


def preprocess(img):
    """
    preprocess image file
    Args:
        file_path: path to file
    Returns:
        img_array: array of image after preprocessing
    """
    img = img.resize(size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array[None, :, :])

    return img_array


def load_model(model_path: Path):
    """
    load trained model from path

    Args:
        model_path: path to saved model
    Returns:
        model: saved model
    """
    model = tf.keras.models.load_model(model_path)

    return model


def predict_image(model, img_array):
    """
    predict image

    Args:
        model: trained model
        img_array: array of image to predict
    Returns:
        pred_class: predicted class type
        pred_prob: predicted probability
    """
    preds = model.predict(x=img_array).squeeze()
    pred_class = np.round(np.argmax(preds)).astype(int)
    pred_prob = preds[pred_class]

    classes = yaml.safe_load(open(SCRIPT_PATH / "crawler/download_images_cfg.yaml"))[
        "keywords"
    ]

    classes.sort()

    class_dict = {}
    for i, c in enumerate(classes):
        class_dict[i] = c

    pred_class = class_dict[pred_class].title()

    return pred_class, pred_prob
