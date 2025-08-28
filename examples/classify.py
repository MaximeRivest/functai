from functai import magic


@magic(adapter="json")  # or adapter=dspy.JSONAdapter()
def classify(text: str) -> str:
    """Return 'positive' or 'negative'."""
    ...
