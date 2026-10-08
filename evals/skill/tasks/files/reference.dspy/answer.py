"""Answer each question in questions.csv from its document (DSPy).

Run it:   python answer.py
Results:  answers.csv (id, answer)
"""
import dspy
import pandas as pd
import pymupdf
from PIL import Image

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", temperature=0))


def pdf(path):
    """The PDF's text, and each page as a picture (Figure 1's values are only in the chart)."""
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        pages.append(dspy.Image.from_PIL(Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
    return "\n\n".join(page.get_text() for page in doc), pages


def sheet(path):
    """Every sheet as CSV text; no pictures."""
    sheets = pd.read_excel(path, sheet_name=None)
    return "\n\n".join(f"Sheet {name}:\n{df.to_csv(index=False)}" for name, df in sheets.items()), []


DOCS = {"report.pdf": pdf("docs/report.pdf"), "budget.xlsx": sheet("docs/budget.xlsx")}


class Answer(dspy.Signature):
    """Answer the question from the document, as a single word or number."""
    question: str = dspy.InputField()
    text: str = dspy.InputField(desc="the document's text")
    pages: list[dspy.Image] = dspy.InputField(desc="the document's pages as pictures")
    answer: str = dspy.OutputField()


answer = dspy.Predict(Answer)
questions = pd.read_csv("questions.csv")
examples = [dspy.Example(question=q, text=DOCS[f][0], pages=DOCS[f][1]).with_inputs("question", "text", "pages")
            for q, f in zip(questions["question"], questions["file"])]
results = answer.batch(examples, num_threads=8)
pd.DataFrame({"id": questions["id"], "answer": [r.answer for r in results]}).to_csv("answers.csv", index=False)
print(pd.read_csv("answers.csv"))
