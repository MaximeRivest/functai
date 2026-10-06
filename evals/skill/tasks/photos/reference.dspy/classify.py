"""Name the shape and the colour in each picture of images/ (DSPy).

Run it:   python classify.py
Results:  predictions.csv (file, shape, color)
"""
from pathlib import Path
from typing import Literal

import dspy
import pandas as pd

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", temperature=0))


class Describe(dspy.Signature):
    """The main shape in the picture and its colour."""
    picture: dspy.Image = dspy.InputField()
    shape: Literal["circle", "square", "triangle"] = dspy.OutputField()
    color: Literal["red", "blue", "green", "yellow"] = dspy.OutputField()


describe = dspy.Predict(Describe)
files = sorted(Path("images").glob("*.png"))
examples = [dspy.Example(picture=dspy.Image.from_path(str(f))).with_inputs("picture") for f in files]
results = describe.batch(examples, num_threads=8)
pd.DataFrame({"file": [f.name for f in files], "shape": [r.shape for r in results],
              "color": [r.color for r in results]}).to_csv("predictions.csv", index=False)
