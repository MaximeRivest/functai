# load_plugin { #functai.load_plugin }

```{.python .no-run}
load_plugin(path)
```

A plugin from a Python file that defines ``plugin`` (an
``Plugin``). Loading runs the file: load only code you trust. A file is
read again when it changed (``reload`` after editing it).