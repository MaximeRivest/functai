---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data,files]", "pillow"]
---

# Pictures and files

*A picture, a PDF, a spreadsheet, a web page or a whole folder as an AI function's input: the model sees the pages, not a file name.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai
```

The pictures on this page are drawn here, in a scratch folder, so the page
runs anywhere:

```python
import os, tempfile
from PIL import Image, ImageDraw

os.chdir(tempfile.mkdtemp())   # a scratch folder for this page's files

def draw(path, shape, colour):
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse((48, 48, 208, 208), fill=colour)
    else:
        d.rectangle((48, 48, 208, 208), fill=colour)
    img.save(path)

draw("red_circle.png", "circle", "red")
draw("blue_square.png", "square", "blue")
```

## A picture

A picture is an input of type `ImagePart`, lm15's picture. Give it with
`image(path=...)`, `image(data=..., media_type=...)` or `image(url=...)`:

```python
from typing import Literal
from lm15.types import ImagePart, image

@ai
def shape(picture: ImagePart) -> Literal["circle", "square", "triangle"]:
    """The main shape in the picture."""
    ...

shape(image(path="red_circle.png")), shape(image(path="blue_square.png"))
```

```output
('circle', 'square')
```

The model is sent the picture as a picture, between the input's tags, and
never its file name:

```python
[part.type for part in shape.render(image(path="red_circle.png")).messages[0].parts]
```

```output
['text', 'image', 'text']
```

A picture given by its path is read when the call is made: the request,
the [reply cache](reliability.md)'s key and the [call log](call-log.md)
hold the picture itself. So a file changed under the same name is a new
request (an old answer is never given for a new picture), and a logged
call replays on a machine that does not have the file. A file that cannot
be read refuses the call before anything is sent:

```python
try:
    shape(image(path="missing.png"))
except functai.InterfaceError as error:
    print(error)
```

```output
shape: input 'picture': the file missing.png cannot be read (No such file or directory)
```

Give each picture its own input. A list of pictures (`list[ImagePart]`) is
written into the prompt as text the model cannot see, and FunctAI warns
when a function takes one; for several files, use one `Artifacts` input,
below.

## Documents, spreadsheets, web pages, folders

[Attachments](https://github.com/maximerivest/attachments) turns almost
anything into text and pictures: `att("report.pdf")`, `att("budget.xlsx")`,
`att("https://example.com")`, `att("docs/")`. Its result, `Artifacts`, is an
input type. Install it with `pip install "functai[files]"`.

```python
from attachments import Artifacts, att

with open("plan.md", "w") as f:
    f.write("# Launch plan\n\nShip the beta on Friday. Marketing starts Monday.\n")

@ai
def first_step(doc: Artifacts) -> str:
    """The first thing this document says will happen, in a few words."""
    ...

first_step(att("plan.md"))
```

```output
'Ship the beta on Friday.'
```

Several files are one value, read in order:

```python
@ai
def matches(doc: Artifacts) -> bool:
    """Does the picture show what the note describes?"""
    ...

with open("note.md", "w") as f:
    f.write("The logo is a red circle.")

matches(att(["note.md", "red_circle.png"])), matches(att(["note.md", "blue_square.png"]))
```

```output
(True, False)
```

What the model gets is Attachments' own content: each file's text, and its
pictures as pictures. A PDF sends its text; with `[images: true]` its pages
come one after the other, each page's text followed by that page's image
(`att("report.pdf[pages: 1-4, images: true]")`), so the model reads a chart
next to the words about it. File names are shown,
as `att()` prints them: a folder's files are told apart by them. When a
name could give the answer away (a classifier over `cat_01.png`), give the
picture as an `ImagePart` instead.

```python
[part.type for part in matches.render(att(["note.md", "red_circle.png"])).messages[0].parts]
```

```output
['text', 'image', 'text']
```

`att()` never raises: a file it could not read becomes an entry with an
error and no text. FunctAI refuses such an input before anything is sent,
saying which file and why, instead of letting the model answer an empty
document:

```python
try:
    first_step(att("budget-2031.pdf"))
except functai.InterfaceError as error:
    print(error)
```

```output
first_step: input 'doc': 1 of its files could not be read, so the model would get an empty or partial document: budget-2031.pdf: unpack-error — unpack failed: Unsupported or non-existent input: budget-2031.pdf. Do what each says (att(...).errors lists them), or leave that file out
```

The call log, the reply cache and replays keep Attachments' wire form (its
JSON, pictures in base64), so a document read in one process is the same
`Artifacts` when a conversation or a saved program is opened in another.
`att(...).estimate_tokens()` counts the pictures too, and
`att("report.pdf[max_dim: 1568, image_format: jpeg]")` sends smaller pages.
