"""Reference tokens from tinker-cookbook's renderer for Inkling-Small (the model whose files are published).
Writes cases.json for same_tokens.mjs. Files: huggingface.co/thinkingmachines/Inkling-Small
(tokenizer.json, tokenizer_config.json, chat_template.jinja) in hf/."""
import json, glob, random
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer
M = "thinkingmachines/Inkling-Small"
ren = get_renderer(model_info.get_recommended_renderer_name(M), get_tokenizer(M))
flat = lambda mi: [t for c in mi.chunks for t in c.tokens]
random.seed(3)
docs = [open(f, errors="ignore").read()[:3000] for f in glob.glob("../../python/functai/**/*.py", recursive=True)][:120]
weird = ["".join(chr(random.choice([random.randint(32, 126), random.randint(0x4e00, 0x9fff), random.randint(0x400, 0x4ff),
         random.randint(0x1f300, 0x1f5ff), 10, 32, 45, 9])) for _ in range(random.randint(1, 300))) for _ in range(300)]
cases = []
for i, t in enumerate(docs + weird):
    msgs = ([{"role": "system", "content": "You are FunctAI."}] if i % 3 == 0 else []) + \
           [{"role": "user", "content": t}, {"role": "assistant", "content": weird[i % len(weird)]}]
    eff = [0.0, 0.2, 0.9][i % 3]
    mi, w = ren.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=eff)
    cases.append({"messages": msgs, "effort": eff, "train_ids": flat(mi), "weights": [float(x) for x in w],
                  "prompt_ids": flat(ren.build_generation_prompt(msgs[:-1], effort=eff))})
json.dump(cases, open("cases.json", "w"))
print(len(cases), "cases")
