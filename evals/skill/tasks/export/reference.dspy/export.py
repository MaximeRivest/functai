"""Write the training conversations for the fine-tuned model to train.jsonl (DSPy), in
DSPy's own fine-tuning format, for the call we will make: dspy.Predict(Simplify.delete("style_guide")).
No model call."""
import json

import dspy
import pandas as pd

from simplify import Simplify

student = Simplify.delete("style_guide")      # the style guide is learned, not sent
adapter = dspy.ChatAdapter()                  # DSPy's default adapter
rows = pd.read_csv("rows.csv")
with open("train.jsonl", "w") as f:
    for text, result in zip(rows["text"], rows["result"]):
        f.write(json.dumps(adapter.format_finetune_data(student, [], {"text": text}, {"result": result})) + "\n")
