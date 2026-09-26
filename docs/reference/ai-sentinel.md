# _ai { #functai._ai }

`_ai`

The model's answer, inside an AI function's body.

A bare ``_ai`` is always the answer, and behaves like the value it stands
for: ``return _ai``, ``return round(_ai, 2)``, ``return critique, _ai``.
``x: T = _ai`` declares one more output, named ``x`` and written before the
answer; a comment on the line (or ``_ai["..."]``) describes it.