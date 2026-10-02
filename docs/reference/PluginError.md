# PluginError { #functai.PluginError }

```{.python .no-run}
PluginError(code, message, *, plugin=None, hook=None)
```

A plugin refused or failed (contract/plugins.md). ``code`` is one
of ``plugin-api`` (it targets an API version this FunctAI does not
implement), ``plugin-hook`` (no such hook), ``plugin-name``,
``plugin-change`` (a change this hook cannot make, or a value that does
not fit), ``plugin-failed`` (its handler raised: the call stops, since
a change it was meant to make, a redaction say, did not happen),
``plugin-load`` (a file that does not define one). ``plugin`` and
``hook`` name where.