"""
CNN for identifying Star Wars characters
"""

from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Union
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
import yaml

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_inputs,
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_inputs
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("CNN")

SCRIPT_PATH = Path(__file__).parent.absolute()


@dataclass
class StarWarsChars:

    model: tf.keras.models = None

    def __post_init__(self):
        pass

    @staticmethod
    def get_data_gens(
        data_path: Union[Path, str],
        seed: int,
        height: int,
        width: int,
        batch_size: int,
        model_name: str,
    ) -> ImageDataGenerator:
        """
        Get data generators for train/val/test

        Args:
            data_path (Union[Path, str]): path to images
            seed (int): random seed
            height (int): input height
            width (int): input width
            batch_size (int): training batch size
            model_name (str): name of model

        Returns:
            train/val/test data generators (ImageDataGenerator)
        """
        logger.info("Getting datagens")
        train_data_path = Path(data_path) / "train"
        val_data_path = Path(data_path) / "val"
        test_data_path = Path(data_path) / "test"

        if model_name == "MobileNetV2":
            preprocess_input = mobilenet_inputs
        elif model_name == "ResNet50":
            preprocess_input = resnet_inputs

        # train
        train_datagen = ImageDataGenerator(
            # rescale=1./255,
            # shear_range=30,
            # rotation_range=20,
            # zoom_range=0.3,
            # brightness_range=[1, 1.3]
            preprocessing_function=preprocess_input
        )

        train_generator = train_datagen.flow_from_directory(
            train_data_path,
            seed=seed,
            target_size=(height, width),
            batch_size=batch_size,
        )

        # val
        val_datagen = ImageDataGenerator(
            # rescale=1./255
            preprocessing_function=preprocess_input
        )

        val_generator = val_datagen.flow_from_directory(
            val_data_path,
            seed=seed,
            target_size=(height, width),
            batch_size=batch_size,
        )

        # test
        test_datagen = ImageDataGenerator(
            # rescale=1./255
            preprocessing_function=preprocess_input
        )

        test_generator = test_datagen.flow_from_directory(
            test_data_path,
            seed=seed,
            target_size=(height, width),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_generator, val_generator, test_generator

    def create_model(
        self,
        height: int,
        width: int,
        learning_rate: float,
        model_name: str,
        n_classes: int,
    ) -> None:
        """
        Create CNN model for task

        Args:
            height (int): input height
            width (int): input width
            learning_rate (int): learning_rate
            model_name (str): name of model
            n_classes (int): number of classes
        """
        logger.info("Creating model")

        models = {"MobileNetV2": MobileNetV2, "ResNet50": ResNet50}

        base_model = models[model_name](
            input_shape=(height, width, 3),
            weights="imagenet",
            include_top=False,
        )

        base_model.trainable = False

        # add fc layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)

        predictions = Dense(n_classes, activation="softmax")(x)

        self.model = Model(base_model.input, predictions)

        self.model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(
        self,
        train_generator: ImageDataGenerator,
        val_generator: ImageDataGenerator,
        epochs: int,
    ) -> tf.keras.callbacks.History:
        """
        Fit CNN model

        Args:
            train_generator (ImageDataGenerator): train generator
            val_generator (ImageDataGenerator): validation generator
            learning_rate (int): learning_rate
        """
        logger.info("Fitting model")
        early_stop = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=3, verbose=1, mode="auto"
        )
        # model_checkpoint = ModelCheckpoint(
        #     "checkpoint.h5", monitor="val_loss", save_best_only=True
        # )

        # fit model
        history = self.model.fit_generator(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[
                early_stop,
                # model_checkpoint,
            ],
        )
        return history

    def predict(self, test_generator: ImageDataGenerator) -> np.ndarray:
        """
        Predict targets

        Args:
            test_generator (ImageDataGenerator): test generator

        Returns:
            preds (np.ndarray): class predictions
        """
        logger.info("Getting predictions")
        preds = np.argmax(self.model.predict(x=test_generator), axis=-1)

        return preds

    def evaluate(
        self,
        test_generator: ImageDataGenerator,
        preds: np.ndarray,
        history: tf.keras.callbacks.History = None,
    ):
        """
        1. print test loss and acc
        2. plot classification report, confusion matrix, train/val loss+acc

        Args:
            test_generator (ImageDataGenerator): test generator
            preds (np.ndarray): predictions
            history (tf.keras.callbacks.History): model's history captured during fitting

        Returns:
            loss (float): test loss
            accuracy (float): test accuracy
        """
        model_loss, model_accuracy = self.model.evaluate(test_generator)
        print(f"Test: loss={model_loss:.4f}, accuracy={model_accuracy:.4f}")

        # actual classes
        actuals = test_generator.labels

        # class mappings
        class_indices = test_generator.class_indices
        class_labels = list(class_indices.keys())
        classes = np.array(class_labels)

        # reports
        print(
            classification_report(classes[actuals], classes[preds], labels=class_labels)
        )
        conf_mat = confusion_matrix(
            classes[actuals], classes[preds], labels=class_labels
        )
        cm_disp = ConfusionMatrixDisplay(conf_mat, display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(20, 20))
        cm_disp.plot(xticks_rotation="vertical", ax=ax)

        if history:
            # plot acc and loss
            acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]

            plt.figure(figsize=(10, 10))

            # plot acc
            plt.subplot(2, 2, 1)
            plt.plot(acc, label="Training Accuracy")
            plt.plot(val_acc, label="Validation Accuracy")
            plt.legend(loc="lower right")
            plt.ylabel("Accuracy")
            plt.ylim([min(plt.ylim()), 1])
            plt.title("Training and Validation Accuracy")

            # plot loss
            plt.subplot(2, 2, 2)
            plt.plot(loss, label="Training Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.legend(loc="upper right")
            plt.ylabel("Cross Entropy")
            plt.title("Training and Validation Loss")
            plt.xlabel("epoch")
            plt.show()

        return model_loss, model_accuracy

    def save_model(self, save_path: Union[Path, str], model_name: str):
        """
        Save trained

        Args:
            save_path (Union[Path, str]): folder to save model to
            model_name (str): name of model
        """

        self.model.save(
            save_path
            / f"star_wars_{model_name}_{datetime.now().strftime('%Y-%m-%d_%H%M')}.h5"
        )


if __name__ == "__main__":
    config = yaml.safe_load(open(SCRIPT_PATH / "model_cfg.yaml"))
    N_CLASSES = len(
        yaml.safe_load(open(SCRIPT_PATH / "../crawler/download_images_cfg.yaml"))[
            "keywords"
        ]
    )

    MODEL_NAME = config["model_name"]
    DATA_PATH = config["data_path"]
    SEED = config["seed"]
    HEIGHT = config["height"]
    WIDTH = config["width"]
    BATCH_SIZE = config["batch_size"]
    LEARNING_RATE = config["learning_rate"]
    EPOCHS = config["epochs"]
    SAVE = config["save"]
    SAVE_PATH = Path(config["save_path"])

    model = StarWarsChars()
    model.create_model(HEIGHT, WIDTH, LEARNING_RATE, MODEL_NAME, N_CLASSES)
    train_generator, val_generator, test_generator = model.get_data_gens(
        DATA_PATH, SEED, HEIGHT, WIDTH, BATCH_SIZE, MODEL_NAME
    )

    history = model.fit(train_generator, val_generator, EPOCHS)

    if SAVE:
        model.save_model(SAVE_PATH, MODEL_NAME)

    preds = model.predict(test_generator)

    loss, acc = model.evaluate(test_generator, preds, history)
