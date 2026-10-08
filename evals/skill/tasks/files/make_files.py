"""Make the files task's documents (run once; the files are committed).

report.pdf: three pages. Page 2 holds Figure 1, a bar chart drawn as a picture:
its values are in no text layer, so only a model shown the page picture can
read them. budget.xlsx: one sheet, the answers need its numbers.
"""
import csv
import io
import json
from pathlib import Path

import openpyxl
import pymupdf
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
DOCS = HERE / "files" / "docs"
REGIONS = {"North": 42, "South": 17, "East": 29, "West": 35}

# ---- the chart, as a picture
img = Image.new("RGB", (900, 560), "white")
d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=26)
d.line((90, 480, 860, 480), fill="black", width=3)
d.line((90, 60, 90, 480), fill="black", width=3)
for i, (region, value) in enumerate(REGIONS.items()):
    x = 140 + i * 180
    top = 480 - value * 9
    d.rectangle((x, top, x + 110, 480), fill=(60, 90, 160))
    d.text((x + 55, top - 36), str(value), fill="black", font=font, anchor="ma")
    d.text((x + 55, 495), region, fill="black", font=font, anchor="ma")
d.text((450, 20), "Revenue by region, Q3 (thousands of dollars)", fill="black", font=font, anchor="ma")
png = io.BytesIO()
img.save(png, format="PNG")

# ---- the report
doc = pymupdf.open()
p1 = doc.new_page()
p1.insert_text((72, 90), "Larkspur Supplies - Quarterly report, Q3 2026", fontsize=16)
p1.insert_textbox(pymupdf.Rect(72, 120, 520, 400),
                  "Sales held up across the quarter. Revenue for each region is shown in Figure 1 "
                  "on the next page. Costs are discussed on page 3.", fontsize=11)
p2 = doc.new_page()
p2.insert_text((72, 80), "Figure 1", fontsize=13)
p2.insert_image(pymupdf.Rect(72, 100, 522, 380), stream=png.getvalue())
p3 = doc.new_page()
p3.insert_textbox(pymupdf.Rect(72, 80, 520, 400),
                  "Costs rose with fuel prices. At its September meeting the board approved a dividend "
                  "of 0.42 dollars per share, payable in November.", fontsize=11)
doc.save(DOCS / "report.pdf")

# ---- the budget
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Q3"
ws.append(["Department", "Budget", "Spent"])
budget = [("Engineering", 1_300_000, 1_240_000), ("Marketing", 400_000, 515_000),
          ("Operations", 650_000, 690_000), ("Support", 220_000, 205_000)]
for row in budget:
    ws.append(row)
wb.save(DOCS / "budget.xlsx")

# ---- the questions and their answers
questions = [
    ("q1", "report.pdf", "Which region had the highest revenue in Figure 1?", ["north"]),
    ("q2", "report.pdf", "What revenue does Figure 1 show for the South region, in thousands of dollars?", ["17"]),
    ("q3", "report.pdf", "What dividend per share did the board approve, in dollars?", ["0.42"]),
    ("q4", "budget.xlsx", "Which department went over its budget by the most?", ["marketing"]),
    ("q5", "budget.xlsx", "How much did Engineering spend, in dollars?", ["1240000"]),
]
with open(HERE / "files" / "questions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "file", "question"])
    for q, file, text, _ in questions:
        w.writerow([q, file, text])
(HERE / "truth.json").write_text(json.dumps({q: {"accept": a, "image_only": q in ("q1", "q2")} for q, _, _, a in questions},
                                            indent=1))
print("made", sorted(p.name for p in DOCS.iterdir()))
