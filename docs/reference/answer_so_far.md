# answer_so_far { #functai.answer_so_far }

```{.python .no-run}
answer_so_far(text, shape)
```

The answer so far, read from its text by its shape (contract/streaming.md,
*The answer so far*): the text for a text answer; the JSON read so far for a
record or a list (complete values; a string still being written as its text
so far, unless it is a choice, which waits for its closing quote; a number,
true, false or null once complete); None for any other answer, and when
the text is not the start of what the shape gives. Provisional: the call's
result is the typed, checked value.