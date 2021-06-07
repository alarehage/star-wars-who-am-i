import os
import marko
from flask import Flask, jsonify, request, render_template
from waitress import serve
import logging
from pathlib import Path

from .inference import preprocess, load_model, predict_image
from .utils.utils import base64_to_pil

SCRIPT_PATH = Path(__file__).absolute()
MODEL_PATH = SCRIPT_PATH.parent.parent / "saved_models"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s", level=logging.INFO
)
logger = logging.getLogger("app")

app = Flask(__name__)

model = load_model(MODEL_PATH / "star_wars_ResNet50_2021-06-02_0415.h5")
logger.info("Model loaded")


@app.route("/", methods=["GET"])
def index():
    if request.method == "GET":
        return render_template("index.html")


@app.route("/info", methods=["GET"])
def short_description():
    if request.method == "GET":
        info = jsonify(
            {
                "model": "MobileNetv2",
                "input-size": "224x224x3",
                "num-classes": 65,
                "pretrained-on": "ImageNet",
            }
        )
        return info


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # convert to pillow format
        img = base64_to_pil(request.json)
        logger.info("Image file read and converted")

        # read and preprocess image
        img_array = preprocess(img)
        logger.info("Image preprocessed")

        # get prediction
        pred, proba = predict_image(model, img_array)
        logger.info("Image predicted")

        # consolidate results
        result = f"Class: {pred}<br/>Proba: {str(round(proba, 2))}"
        logger.info(f"Predicted class: {pred}")
        logger.info(f"Proba: {proba:0.2f}")

        return jsonify(result=result)


# @app.route("/readme", methods=["GET"])
# def readme():
#     if request.method == "GET":
#         with open("static/README_WEB.md", "r") as f:
#             readme = marko.convert(f.read())

#         return render_template("readme.html", data=readme)


if __name__ == "__main__":
    # app.run(debug=True, port=8000)

    serve(app, host="0.0.0.0", port=8000)
