# file { #functai.file }

```{.python .no-run}
file(path)
```

A data file the program reads: ``open(functai.file("data/stopwords.txt"))``.

A relative path is relative to the file of the code that calls this (the
current directory in a notebook). ``functai.save`` copies every file named
this way with a literal path into the saved program, and the loaded program
reads its own copy.