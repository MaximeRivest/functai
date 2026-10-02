"""Keep earlier turns' bulky fields under a budget (the rule of Chattering's
`image-budget`, on any field: photos, documents, logs).

When the kept fields of the earlier turns pass ``high`` characters, the
oldest are left out until ``low`` remains; the newest always stays. The two
marks make the set left out change rarely, so the provider's prompt cache
keeps working, and the rule depends only on the turns, so the same history
always gives the same request. Left out, a field is named in the turn's
``without`` (its record says so): nothing is rewritten."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import functai


def plugin(fields: Sequence[str], *, high: int = 16_000_000, low: int = 8_000_000) -> functai.Plugin:
    p = functai.Plugin("bulky-budget", version="1.0.0")

    @p.context
    def budget(context: Any) -> Optional[functai.Change]:
        items = []
        for t in context.turns:
            for name in fields:
                if name in t.without:
                    continue
                value = t.inputs.get(name, t.outputs.get(name))
                if value is not None:
                    items.append((t.id, name, len(json.dumps(value, ensure_ascii=False))))
        kept, start = 0, 0
        for i, (_t, _n, size) in enumerate(items):
            kept += size
            if kept > high:
                while kept > low and start < i:
                    kept -= items[start][2]
                    start += 1
        if not start:
            return None
        drop: dict = {}
        for tid, name, _s in items[:start]:
            drop.setdefault(tid, []).append(name)
        return functai.Change(without=drop)

    return p
