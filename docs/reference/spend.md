# spend { #functai.spend }

```{.python .no-run}
spend(usd)
```

Say what code inside a call spent beyond the models' tokens, in dollars
(a browsing service's charge, a paid API): ``functai.spend(0.05)``, from a
module's body or a tool (contract/calls.md, *Spending*). The call's record
keeps the sum (``spent``), and the ``limits`` plugin counts it in the
tree's cost. Outside any call it does nothing.