# Signing in, so FunctAI is enough to use a subscription or save a key: lm15
# does the work, and its credentials file is shared by every lm15 language
# (a sign-in made from Python works here, and the other way round).

"""
    FunctAI.login(provider; key)

Sign in to a provider once; every later session (in any FunctAI language)
uses it. Subscriptions (`"claude"`, `"chatgpt"`, `"copilot"`, `"grok"`,
`"kimi"`) open a browser, or print a link or code over SSH; for an API
provider (`"openai"`, `"anthropic"`, `"groq"`, …) the key is asked for, or
given as `key`, and saved.

```julia
FunctAI.login("claude")
@ai lm = "claude:claude-sonnet-4-5" function …
```
"""
function login(provider::AbstractString; key=nothing)
    auth = LM15.local_auth()
    id = get(PREFIX_ALIASES, lowercase(provider), lowercase(provider))
    connection = key === nothing ? LM15.login(auth, id; ui=LM15.TerminalUI()) : LM15.set_api_key(auth, id, key)
    forget_routes!()
    connection
end

"""
    FunctAI.logins()

The providers this machine is signed in to (lm15's saved connections; no
secrets are shown).
"""
logins() = [(provider=c.provider, method=c.method_id, label=c.label, created=c.created_at)
            for c in LM15.connections(LM15.local_auth())]

"""
    FunctAI.logout(provider)

Forget the saved sign-in of a provider.
"""
function logout(provider::AbstractString)
    out = LM15.logout(LM15.local_auth(), get(PREFIX_ALIASES, lowercase(provider), lowercase(provider)))
    forget_routes!()
    out
end
