"""Can Inkling 2 Small's tokenizer (h200k) be rebuilt from what is published? Its vocabulary yes
(every token's bytes, by id); its merge rule no. Ranks = ids (tiktoken's rule) and the o200k
pre-tokenizer match on most texts, not all: runs of punctuation merge differently."""
import tiktoken, glob, random
from tml_renderers import tokenizers
t = tokenizers.H200kBaseChatTokenizer(); o = tiktoken.get_encoding("o200k_base")
ranks = {}
for i in range(201088):
    if t.is_special_token(i): continue
    try: ranks[t.decode_bytes([i])] = i
    except Exception: pass
enc = tiktoken.Encoding("h200k_rebuilt", pat_str=o._pat_str, mergeable_ranks=ranks, special_tokens={})
texts = [open(f, errors="ignore").read() for pat in ("../../python/functai/**/*.py", "../../docs/**/*.md", "../../ts/src/**/*.ts")
         for f in glob.glob(pat, recursive=True)]
random.seed(1)
texts += ["".join(chr(random.choice([random.randint(32, 126), random.randint(0x4e00, 0x9fff), random.randint(0x400, 0x4ff),
          random.randint(0x1f300, 0x1f5ff), random.randint(0x900, 0x97f), 10, 32, 45, 9])) for _ in range(random.randint(1, 500)))
          for _ in range(2000)]
bad = sum(enc.encode_ordinary(s) != t.encode_ordinary(s) for s in texts)
print(f"{len(ranks)} tokens recovered; {bad} of {len(texts)} texts tokenize differently")
s = " ----------------"
print(repr(s), "h200k:", [t.decode_bytes([x]) for x in t.encode_ordinary(s)], "rebuilt:", [enc.decode_single_token_bytes(x) for x in enc.encode_ordinary(s)])
