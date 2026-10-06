"""The DSPy reference for the export task: DSPy's own fine-tuning format
(ChatAdapter.format_finetune_data, what BootstrapFinetune writes) for the call the
task names, dspy.Predict(Simplify.delete("style_guide")). Prints JSON: one message
list per row of rows.csv."""
import csv
import importlib.util
import json
from pathlib import Path

import dspy

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("simplify_dspy", HERE / "files.dspy" / "simplify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
student = module.Simplify.delete("style_guide")
adapter = dspy.ChatAdapter()
rows = list(csv.DictReader(open(HERE / "files" / "rows.csv")))
print(json.dumps([adapter.format_finetune_data(student, [], {"text": r["text"]}, {"result": r["result"]})["messages"]
                  for r in rows]))
