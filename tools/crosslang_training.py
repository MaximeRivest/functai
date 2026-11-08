"""Python and TypeScript make the same tokens from a student's real published
files (contract/training.md §1 to §5): conversations of code, prose in several
scripts, emoji, a system message, several turns, and pictures of many sizes,
rendered by both and compared token for token, with the reply's start, the lead,
the end and the stop token. Students whose files are not in the Hugging Face
cache are skipped (offline), unless ``--download``.

    python/.venv/bin/python tools/crosslang_training.py [--download]
"""

import base64
import glob
import json
import os
import random
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

STUDENTS = ["Qwen/Qwen3.5-4B", "thinkingmachines/Inkling-Small"]


def png(w, h, shade):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([shade]) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def items(with_pictures):
    rnd = random.Random(11)
    texts = [open(f, errors="ignore").read()[:1200] for f in sorted(glob.glob(str(ROOT / "python/functai/*.py")))[:25]]
    texts += ["Où est la bibliothèque? 東京は日本の首都です。 Здравствуйте! مرحبا 🤖🧠 tabs\t\tand\n\n\nlines  ----------"]
    texts += ["".join(chr(rnd.choice([rnd.randint(32, 126), rnd.randint(0x4E00, 0x9FFF), rnd.randint(0x400, 0x4FF),
                                       rnd.randint(0x1F300, 0x1F5FF), 10, 32, 45, 9])) for _ in range(rnd.randint(1, 300)))
              for _ in range(30)]
    out = []
    for i, t in enumerate(texts):
        msgs = [{"role": "system", "content": "Function: f\n\nDo it."}] if i % 2 else []
        pics = []
        if with_pictures and i % 3 == 0:
            sizes = [(81 + i, 40 + 3 * i), (120, 40), (1280, 800), (390, 844), (40, 40)]
            pics = [png(*sizes[i % len(sizes)], i % 250)] + ([png(64, 48, 7)] if i % 6 == 0 else [])
            parts = [{"type": "text", "text": "<screen>\n"}]
            for _ in pics:
                parts += [{"type": "image"}, {"type": "text", "text": "\n"}]
            msgs.append({"role": "user", "content": parts + [{"type": "text", "text": t}]})
        else:
            msgs.append({"role": "user", "content": t})
        if i % 5 == 4:
            msgs += [{"role": "assistant", "content": "<result>\nfirst\n</result>"}, {"role": "user", "content": "again"}]
        reply = None if i % 7 == 6 else f"<result>\n{texts[(i + 1) % len(texts)][:80]}\n</result>"
        out.append({"messages": msgs, "pictures": [base64.b64encode(p).decode() for p in pics], "reply": reply})
    return out


def main(download=False):
    from huggingface_hub import hf_hub_download
    from functai.bake.tokens import Student, picture
    total = 0
    for sid in STUDENTS:
        try:
            st = Student.load(sid, local_files_only=not download)
        except Exception as err:  # noqa: BLE001
            print(f"{sid}: skipped (its files are not here: {type(err).__name__})")
            continue
        folder = Path(hf_hub_download(sid, "tokenizer.json", local_files_only=True)).parent
        its = items(st.picture is not None)
        node = subprocess.run(["node", "--conditions=functai-source", "tools/render.ts"], cwd=ROOT / "ts",
                              input=json.dumps({"student": sid, "dir": str(folder), "effort": 0, "items": its}),
                              capture_output=True, text=True, check=True)
        theirs = json.loads(node.stdout)
        ends = theirs["ends"]
        assert (ends["lead"], ends["end"], ends["stop"], ends["head"]) == (st.lead, st.end, st.stop_token, st.end_head), \
            (sid, ends)
        assert ends["kwargs"] == st.kwargs and ends["template"] == st.template_sha256 and \
            ends["tokenizer"] == st.tokenizer_sha256, (sid, ends)
        same = 0
        for it, t in zip(its, theirs["items"]):
            r = st.render(it["messages"], [picture(base64.b64decode(b)) for b in it["pictures"]], it["reply"])
            mine = {"prompt": r.prompt, "full": r.full, "replyStart": r.reply_start, "prefixFallback": r.prefix_fallback}
            assert mine == t, f"{sid}: Python and TypeScript differ on {json.dumps(it)[:300]}"
            same += 1
        n_pics = sum(len(it["pictures"]) for it in its)
        print(f"{sid}: {same} conversations ({n_pics} pictures) give the same tokens in Python and TypeScript")
        total += same
    if not total:
        print("no student's files are here: nothing compared (run with --download)")


if __name__ == "__main__":
    main(download="--download" in sys.argv)
