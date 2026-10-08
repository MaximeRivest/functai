"""Name the shape and the colour in each picture of images/.

Run it:   python classify.py
Results:  predictions.csv (file, shape, color)
"""
from pathlib import Path
from typing import Literal

import functai
from dpyr import read
from functai import _ai, ai
from lm15.types import ImagePart, image

functai.configure(lm="gpt-4.1-mini", temperature=0, cache_replies="disk")


@ai
def shape_and_color(picture: ImagePart) -> tuple[str, str]:
    """The main shape in the picture and its colour."""
    shape: Literal["circle", "square", "triangle"] = _ai
    color: Literal["red", "blue", "green", "yellow"] = _ai
    return shape, color


files = sorted(Path("images").glob("*.png"))
run = shape_and_color.map([{"file": f.name, "picture": image(path=f)} for f in files], threads=8).to_pandas()
read({"file": run["file"], "shape": run["pred_shape"], "color": run["pred_color"]}).write_csv("predictions.csv")
print(run[["file", "pred_shape", "pred_color", "error"]])
