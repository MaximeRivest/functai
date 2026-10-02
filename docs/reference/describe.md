# describe { #functai.describe }

```{.python .no-run}
describe(source, node=None)
```

What a saved program takes and gives, without loading it or running
anything: the node's interface (contract/saved.md, *Describing without
loading*), whatever language wrote the folder.

``source``: a saved folder, its ``functai.json``, or the manifest as a
dict; ``node``: a program's key (``"module:name"``), the entry by default.
Raises ``LoadRefused`` (``.code``: ``saved-format``, ``saved-malformed``,
``saved-not-ai``, ``interface-malformed``, ``saved-differs``,
``saved-no-interface``). A node written before interfaces were saved: an
AI function is described by its signature, with its instruction (the
prompt) as the description; a module is not known.