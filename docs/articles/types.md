---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Types are the contract

*Ask for a type, get that type back: lists, dataclasses, enums, literals, pydantic models, as outputs and as inputs.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

The return type of an AI function is a contract. The model is shown what
shape the answer must have, and the reply is read back into that type
before your code sees it. If it can't be read, functai asks the model
again once, then raises; you never get a half-parsed string by accident.

This article goes through the types you will use most, from simple to
structured. All of them also work as **inputs**.

| type | example | the model sees |
|---|---|---|
| text, numbers, booleans | `str`, `int`, `float`, `bool` | the value's kind |
| a fixed set of answers | `Literal["a", "b"]`, an `Enum` | the allowed answers |
| collections | `list[str]`, `dict[str, int]`, `tuple[str, int]`, `set[str]` | the shape, as JSON |
| maybe nothing | `str | None`, `Optional[str]` | that it may be empty |
| records | a dataclass, a `TypedDict`, a pydantic model, a plain annotated class | the fields, their types and comments |

## Plain values

```python
@ai
def word_count_guess(text: str) -> int:
    """Roughly how many words the text has."""
    ...

word_count_guess("The quick brown fox jumps over the lazy dog.")
```

```output
9
```

The result is a real `int`, ready for arithmetic:

```python
word_count_guess("One two three.") + 1
```

```output
4
```

## A fixed set of answers

For classification, give the answers as a `Literal` or an `Enum`. The
model is told the allowed values, and anything else is refused.

```python
from enum import Enum
from typing import Literal

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@ai
def priority(issue: str) -> Priority:
    """How urgently the issue must be fixed."""
    ...

priority("The main database is unresponsive and every customer sees an error.")
```

```output
<Priority.HIGH: 'high'>
```

```python
@ai
def topic(headline: str) -> Literal["sport", "politics", "science", "other"]:
    """The headline's topic."""
    ...

topic("Rover finds traces of ancient riverbed on Mars")
```

```output
'science'
```

## Collections

```python
@ai
def keywords(article: str) -> list[str]:
    """Five key terms from the article, lowercase."""
    ...

keywords("Python type hints serve as the contract between your code and a "
         "language model: the model is shown a schema, and replies are parsed back.")
```

```output
['python', 'type hints', 'contract', 'language model', 'schema']
```

A dictionary, or a tuple for several values in one answer:

```python
@ai
def ingredient_counts(recipe: str) -> dict[str, int]:
    """How many of each countable ingredient the recipe uses."""
    ...

ingredient_counts("Beat 3 eggs with 2 bananas, add 1 cup of flour and 2 cups of milk.")
```

```output
{'eggs': 3, 'bananas': 2, 'cups of flour': 1, 'cups of milk': 2}
```

## Records

A dataclass describes an answer with several named parts. Comments on
fields tell the model what goes in each one.

```python
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float            # in dollars, without the currency sign
    features: list[str]
    in_stock: bool

@ai
def extract_product(description: str) -> Product:
    """Extract the product's details from its description."""
    ...

extract_product("iPhone 15 Pro - $999, 5G, titanium design, available now")
```

```output
Product(name='iPhone 15 Pro', price=999.0, features=['5G', 'titanium design'], in_stock=True)
```

A plain class with annotations works too: functai turns it into a
dataclass for you.

```python
class Person:
    name: str      # full name, as written
    age: int       # in years
    city: str | None  # None when the text does not say

@ai
def person(text: str) -> Person:
    """The person the text is about."""
    ...

person("Marie Curie, 66, spent her last years in Passy.")
```

```output
Person(name='Marie Curie', age=66, city='Passy')
```

Pydantic models work the same way, and their validators run on the reply:

```python
from pydantic import BaseModel, Field

class Review(BaseModel):
    stars: int = Field(ge=1, le=5)
    summary: str

@ai
def read_review(text: str) -> Review:
    """The review's rating and a one-sentence summary."""
    ...

read_review("Great tacos, loud music, slow service. I'll be back though.")
```

```output
Review(stars=3, summary='The tacos are great and the atmosphere lively, but the service is slow.')
```

Records nest: a field can be a list of another record, an enum, or an
optional value.

## Types as inputs

Inputs can be structured too. The model is shown their fields, and the
values are written as JSON.

```python
@ai
def pitch(product: Product) -> str:
    """A one-sentence sales pitch for the product."""
    ...

pitch(Product(name="Trail kettle", price=39.0, features=["titanium", "folding handle"], in_stock=True))
```

```output
'The Trail Kettle is a lightweight, durable titanium kettle with a convenient folding handle, perfect for outdoor adventures at just $39.'
```

## When the reply doesn't fit

A reply that is almost right (a misspelled tag, extra words around a
number) is repaired and read. A reply that can't be read is sent back
once with a hint of what went wrong. If that fails too, the call raises
`lmcc.Refusal`, with a `.hint` saying what was expected. See
[When the model gets it wrong](reliability.md).
