"""Write the training conversations for the simplify function to train.jsonl (no model call)."""
import functai.bake
from dpyr import read
from simplify import simplify
GUIDE = open("style_guide.md").read()
ex = functai.bake.examples(simplify, read("rows.csv"), fixed={"style_guide": GUIDE})
print(ex.stats())
ex.save("train.jsonl")
