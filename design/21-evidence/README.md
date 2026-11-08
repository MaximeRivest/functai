# Evidence for design 21 (2026-11-08)

Run on lambda against tinker 0.39.0, tinker-cookbook 0.6.1, tml-renderers 0.4.1
(the Bigspin agent's environment, `~/Projects/gaze-models/tinker/.venv`), and Node 24 with
@huggingface/tokenizers 0.2.0 and @huggingface/jinja 0.5.11. Outputs are in `*.txt`.

| script | shows |
|---|---|
| `torch_need.sh` | what pulls PyTorch in (the cookbook, and tml-renderers' import of it, for audio only), and what it weighs |
| `image_tokens.py` | Inkling's picture tokens: the client computes (floor(w/40)+1)·ceil(h/40), the server expects ceil(w/40)·ceil(h/40); pictures are not resized |
| `make_cases.py`, `same_tokens.mjs` (run it where the two npm packages are installed: `node same_tokens.mjs HF_DIR cases.json`) | Inkling-Small's published tokenizer.json and chat_template.jinja, rendered in Node, give the cookbook's exact training tokens, prompt tokens and loss mask (365 of 365) |
| `h200k.py` | Inkling 2 Small's tokenizer (h200k) is not reproducible from published data: its vocabulary is recoverable, its merge rule is not (best rebuild: 93 of 2,334 texts differ) |
