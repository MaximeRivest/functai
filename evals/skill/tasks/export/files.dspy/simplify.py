import dspy


class Simplify(dspy.Signature):
    """Rewrite this sentence from a science paper so a curious 12-year-old understands it,
    following the style guide. Keep every fact; add nothing."""

    text: str = dspy.InputField()
    style_guide: str = dspy.InputField()
    result: str = dspy.OutputField()


simplify = dspy.Predict(Simplify)
