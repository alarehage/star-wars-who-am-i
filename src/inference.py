from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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


# def preprocess_og(file_path):
#     """
#     preprocess image file
#     Args:
#         file_path: path to file
#     Returns:
#         img_array: array of image after preprocessing
#     """
#     img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224,224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = preprocess_input(img_array[None,:,:])

#     return img_array


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

    class_dict = {
        0: "ackbar",
        1: "anakin skywalker",
        2: "bail organa",
        3: "bb8",
        4: "boba fett",
        5: "c3po",
        6: "captain phasma",
        7: "chewbacca",
        8: "count dooku",
        9: "darth maul",
        10: "darth vader",
        11: "finn",
        12: "greedo",
        13: "grievous",
        14: "han solo",
        15: "jabba the hutt",
        16: "jango fett",
        17: "jar jar binks",
        18: "ki adi mundi",
        19: "kit fisto",
        20: "lama su",
        21: "lando calrissian",
        22: "leia organa",
        23: "luke skywalker",
        24: "mace windu",
        25: "mon mothma",
        26: "nute gunray",
        27: "obi wan kenobi",
        28: "padme amidala",
        29: "palpatine",
        30: "plo koon",
        31: "poe dameron",
        32: "qui gon jinn",
        33: "r2d2",
        34: "rey",
        35: "shaak ti",
        36: "shmi skywalker",
        37: "tarfful",
        38: "taun we",
        39: "wicket systri warrick",
        40: "wilhuff tarkin",
        41: "yoda",
    }
    pred_class = class_dict[pred_class]

    return pred_class, pred_prob
