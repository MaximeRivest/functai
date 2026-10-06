"""Draw the task's 12 pictures (run once; the PNGs are committed). Neutral file names, no metadata."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
COLORS = {"red": (215, 40, 40), "blue": (40, 80, 215), "green": (40, 160, 70), "yellow": (235, 200, 30)}
SHAPES = ["circle", "square", "triangle"]
random.seed(7)
items = [(s, c) for s in SHAPES for c in COLORS]
random.shuffle(items)
truth = {}
for i, (shape, color) in enumerate(items, 1):
    img = Image.new("RGB", (256, 256), (245, 245, 240))
    d = ImageDraw.Draw(img)
    size = random.randint(90, 150)
    x, y = random.randint(20, 236 - size), random.randint(20, 236 - size)
    box = (x, y, x + size, y + size)
    if shape == "circle":
        d.ellipse(box, fill=COLORS[color])
    elif shape == "square":
        d.rectangle(box, fill=COLORS[color])
    else:
        d.polygon([(x + size / 2, y), (x, y + size), (x + size, y + size)], fill=COLORS[color])
    name = f"img{i:02d}.png"
    img.save(HERE / "files" / "images" / name)
    truth[name] = {"shape": shape, "color": color}
(HERE / "truth.json").write_text(json.dumps(truth, indent=1))
print(len(truth))
