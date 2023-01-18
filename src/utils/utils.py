import base64
import re
from io import BytesIO

import requests
from PIL import Image


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub("^data:image/.+;base64,", "", img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    return pil_image

def get_image_from_url(url):
    """
    Read image from given URL

    Args:
        url (str): input url
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    return img
