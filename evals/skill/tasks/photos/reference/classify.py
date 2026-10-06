"""Name the shape and the colour in each picture of images/.

Run it:   python classify.py
Results:  predictions.csv (file, shape, color)
"""
import base64
import io
from pathlib import Path
from typing import Literal

import functai
import lmcc
from functai import _ai
from dpyr import read
from PIL import Image

functai.configure(lm="gpt-4.1-mini", temperature=0, cache_replies="disk")


# A picture, sent to the model as an image (functai has no image type of its own yet).
class Photo(dict):
    """An image part."""


lmcc.format(Photo, shape={"media": "image"}, write=lambda photo: [dict(photo)])


def photo(path):
    image = Image.open(path).convert("RGB")
    image.info.clear()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Photo(type="image", media_type="image/png", data=base64.b64encode(buffer.getvalue()).decode())


@functai.ai
def shape_and_color(picture: Photo) -> tuple[str, str]:
    """The main shape in the picture and its colour."""
    shape: Literal["circle", "square", "triangle"] = _ai
    color: Literal["red", "blue", "green", "yellow"] = _ai
    return shape, color


files = sorted(Path("images").glob("*.png"))
run = shape_and_color.map([{"file": f.name, "picture": photo(f)} for f in files], threads=8).to_pandas()
read({"file": run["file"], "shape": run["pred_shape"], "color": run["pred_color"]}).write_csv("predictions.csv")
print(run[["file", "pred_shape", "pred_color", "error"]])
