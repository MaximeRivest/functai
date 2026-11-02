# MemoryCalls { #functai.MemoryCalls }

```{.python .no-run}
MemoryCalls()
```

A call log kept in this process's memory (``log_calls=MemoryCalls()``):
for tests, and to see what a store of one's own is given. ``records``
holds every record appended, in order; ``read()`` gives them back.

A store of one's own is any object with these two methods: ``append(record)``
keeps one record (a dict: a call, or a rating) and ``read()`` gives back
every record kept, in the order appended. Put it in front of a database,
a bucket or a queue when a machine keeps no files (a serverless function,
a container that goes away).