# plugins.Spent { #functai.plugins.Spent }

```{.python .no-run}
plugins.Spent(exchanges, seconds, reported=0.0)
```

What a call tree spent so far in this process (contract/plugins.md,
*What a tree spent*): each request a provider answered, as ``(model,
usage)`` (lm15's token counts, by name); replies from the reply cache and
replies a resumed turn reuses cost nothing now and are not in it.
``requests``, ``usage`` (summed: a counter only when every request
reported it), ``tokens`` (``total_tokens`` summed, or None when a request
did not report it), ``seconds`` (since the tree's outermost call started),
``reported`` (dollars code in the tree said it spent: ``functai.spend``)
and ``cost(prices)``, which counts both.

## Methods

| Name | Description |
| --- | --- |
| [cost](#functai.plugins.Spent.cost) | The list-price cost of the requests, from ``prices`` (a model's |

### cost { #functai.plugins.Spent.cost }

```{.python .no-run}
plugins.Spent.cost(prices)
```

The list-price cost of the requests, from ``prices`` (a model's
name, as asked, to an lm15 ``InferencePricing`` or a dict of its
fields), plus what code reported it spent: exact or None, never a
guess (a model without a price, or a request whose counts do not
settle its price, makes it None).