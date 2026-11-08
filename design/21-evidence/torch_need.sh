#!/usr/bin/env bash
# What needs PyTorch, and how much it weighs. Run from a folder with a tinker venv in .venv.
S=.venv/lib/python3.12/site-packages
for p in tinker tinker_cookbook; do echo "$p requires:"; grep -E "^Requires-Dist" $S/$p-*.dist-info/METADATA | grep -v "extra ==" | grep -i -E "torch|transformers"; done
echo "tml_renderers imports torch in:"; grep -l "^import torch" $S/tml_renderers/*.py
echo "and uses it for:"; grep -n "torch" $S/tml_renderers/chat.pyi | head -5
du -sh .venv $S/torch
