"""Answer each question in questions.csv from its document (FunctAI + Attachments).

Run it:   python answer.py
Results:  answers.csv (id, answer)
"""
import functai
from attachments import Artifacts, att
from dpyr import read
from functai import ai

functai.configure(lm="gpt-4.1-mini", temperature=0, cache_replies="disk")

# The report's pages as pictures too: Figure 1's values are only in the chart.
DOCS = {"report.pdf": att("docs/report.pdf[images: true]"), "budget.xlsx": att("docs/budget.xlsx")}


@ai
def answer(question: str, document: Artifacts) -> str:
    """Answer the question from the document, as a single word or number."""
    ...


questions = read("questions.csv").to_pandas()
run = answer.map([{"question": q, "document": DOCS[f]} for q, f in zip(questions["question"], questions["file"])],
                 threads=8).to_pandas()
read({"id": questions["id"], "answer": run["pred_result"]}).write_csv("answers.csv")
print(read("answers.csv"))
