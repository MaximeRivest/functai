# logout { #functai.logout }

```{.python .no-run}
logout(provider, *, auth=None)
```

Forget a saved login or key, on this machine.

Forget the saved login or key for a provider, on this machine (the account
itself is untouched). API keys in the environment still work afterwards, except
where lm15 blocks them on purpose: a signed-out xAI subscription does not fall
back to ``XAI_API_KEY``.