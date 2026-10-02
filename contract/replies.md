# The reply cache (format 1)

A model's reply to a request, kept and reused when the same request is
made again. On by the setting `cache_replies` (off by default): `true`
keeps replies in the process's memory; `"disk"` or a path keeps them in
one file every process on the machine shares, so a long run interrupted
and started again sends only what has no kept reply (Python
`python/functai/replies.py`). `cases/replies/` pin the key.

## The key

`"sha256:"` and the hex SHA-256 of the canonical JSON
([calls.md](calls.md), *Canonical JSON*) of

```json
{"functai_reply": 1, "request": <the request>, "replicate": 0}
```

- `request` is lm15's canonical JSON of the request sent (model,
  messages, settings, tools): the same in every language for the same
  request.
- `replicate` is the setting of that name (default `0`): the n-th
  independent answer to the same request. Asking the same request five
  times on purpose (to measure how answers vary) gives five replicates
  five keys; the cache never turns five samples into one.

## What is kept

- **Only a reply that was read.** A reply is kept once lmcc read it and
  its values fit their types. An unreadable reply, a failed request, a
  cancelled one is never kept. So nothing half-written and nothing
  "started" can be found by a later run.
- **One flight per key.** While one caller asks the model, a second
  caller of the same key waits and gets the first one's reply: in a
  process (a lock per key) and across processes (a claim with a lease,
  below). If the first fails, the second asks the model itself.
- **Content.** A call any `log_content` layer drops a field of
  ([calls.md](calls.md), *Content*) is never written to a store that
  outlives the process: a kept reply holds the whole request and reply,
  which is what `log_content` forbids keeping. The memory cache may
  still serve it.
- A reply from the cache is an exchange of the call like any other,
  with `cached: true` and `seconds: 0` ([calls.md](calls.md)), and shown
  as one piece per field ([streaming.md](streaming.md), law 6).

## The file

One SQLite database (default: `$XDG_CACHE_HOME/functai/replies.sqlite`,
`~/Library/Caches/functai/replies.sqlite` on macOS,
`%LOCALAPPDATA%\functai\replies.sqlite` on Windows), created readable
by its owner only, in WAL mode:

```sql
CREATE TABLE replies (key TEXT PRIMARY KEY, format INTEGER NOT NULL, created TEXT NOT NULL,
                      model TEXT, response TEXT NOT NULL);
CREATE TABLE claims  (key TEXT PRIMARY KEY, owner TEXT NOT NULL, until REAL NOT NULL);
```

- `response` is lm15's canonical JSON of the response; `format` is `1`
  (a reader skips a row of a format it does not know); `created` is the
  call log's time format.
- **A claim** is taken in one transaction (`BEGIN IMMEDIATE`) when the key
  has no reply and no claim whose `until` (Unix seconds) is still ahead,
  or one of the same `owner` (`<host>:<pid>:<random>`). Its lease is 120
  seconds; a claim whose lease ran out is taken over (its owner stopped,
  or is slower than the lease: then the request is made twice, which
  costs, never misleads). Keeping a reply and removing its claim are one
  transaction.
- Clearing the cache is deleting rows (or the file).

A cache may also be any object with `get(key)` and `put(key, reply)`,
and optionally `claim(key)`/`unclaim(key)` and `discard(key)`.
