"""Nothing trains: the examples and ready-to-run recipes for other tools.

    folder/
      examples.parquet     the conversations with bake's exact tokens (input_ids, answer_start)
      examples.jsonl       the same as chat messages (for tools that apply the template themselves)
      train_trl.py         TRL on the exact tokens, with bake's settings
      axolotl.yaml         Axolotl on the messages (loss on the reply only)
      README.md            what each is, and how the result comes back to functai

Then ``functai.bake.adopt(<trained folder>, fn, examples=<folder>/examples.parquet)``
checks the trained model's chat template against the examples' tokens and
uses it.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

from ..recipe import recipe
from . import Estimate, Job, Trainer


class Export(Trainer):
    name = "export"

    def estimate(self, job: Job) -> Estimate:
        r = recipe(job)
        return Estimate(True, [], 0.0, 0.0, {**r, "precision": "bf16", "packing": not job.student.hybrid},
                        ["nothing trains: the examples and recipes are written for another tool"],
                        summary="writes the examples and recipes for TRL and Axolotl; trains nothing")

    def supervise(self, run) -> None:
        import pyarrow.parquet as pq
        plan = run.plan
        s = plan["settings"]
        out = Path(plan["output"])
        out.mkdir(parents=True, exist_ok=True)
        if (out / "examples.parquet").resolve() != (run.folder / "examples.parquet").resolve():
            shutil.copy2(run.folder / "examples.parquet", out / "examples.parquet")
        table = pq.read_table(out / "examples.parquet", columns=["function", "messages", "weight", "split"])
        with open(out / "examples.jsonl", "w") as f:
            for r in table.to_pylist():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kwargs = (plan.get("template") or {}).get("kwargs") or {}
        student = plan["student"]
        (out / "train_trl.py").write_text(textwrap.dedent(f'''\
            """{plan["name"]} → {student}: TRL on functai's exact tokens (written by functai.bake, where="export").

                pip install "trl>=1.14" peft datasets      # then: python train_trl.py   (torchrun for several GPUs)
            """
            import datasets
            import peft
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer

            STUDENT = {student!r}
            data = datasets.load_dataset("parquet", data_files="examples.parquet")["train"]
            # the loss on the reply only: positions before answer_start are the prompt
            data = data.map(lambda r: {{"completion_mask": [0] * r["answer_start"]
                                       + [1] * (len(r["input_ids"]) - r["answer_start"])}})
            train = data.filter(lambda r: r["split"] != "validation").select_columns(["input_ids", "completion_mask"])
            val = data.filter(lambda r: r["split"] == "validation").select_columns(["input_ids", "completion_mask"])

            model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16)
            model = peft.get_peft_model(model, peft.LoraConfig(r={s.get("lora_rank") or 32}, lora_alpha=32,
                                                               lora_dropout=0.0, target_modules="all-linear",
                                                               task_type="CAUSAL_LM"))
            args = SFTConfig(
                output_dir="trained", num_train_epochs={s["epochs"]}, learning_rate={s["learning_rate"]!r},
                per_device_train_batch_size=1, gradient_accumulation_steps={s["examples_per_step"]},
                lr_scheduler_type="warmup_stable_decay", warmup_steps={s["schedule"]["warmup"]},
                lr_scheduler_kwargs={{"num_decay_steps": max(1, round({s["steps"]} * {s["schedule"]["decay"]})),
                                     "decay_type": "1-sqrt"}},
                bf16=True, gradient_checkpointing=True, completion_only_loss=True,
                # packing is only safe with flash attention and a model whose every layer is attention
                packing={bool(not plan.get("student_info", {}).get("layer_types") or
                              all(t == "full_attention" for t in plan["student_info"]["layer_types"]))},
                max_length=None, eval_strategy="steps" if len(val) else "no", eval_steps=0.05, save_steps=0.1,
                logging_steps=10, report_to=[])
            trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=val if len(val) else None,
                                 processing_class=AutoTokenizer.from_pretrained(STUDENT))
            trainer.train()
            trainer.save_model("trained/final")
            # back in functai: functai.bake.adopt("trained/final", <the function>, examples="examples.parquet")
        '''))
        (out / "axolotl.yaml").write_text(textwrap.dedent(f'''\
            # {plan["name"]} → {student} (written by functai.bake, where="export")
            # axolotl train axolotl.yaml ; then functai.bake.adopt("outputs/lora", fn, examples="examples.parquet")
            base_model: {student}
            chat_template: tokenizer_default
            {"chat_template_kwargs: " + json.dumps(kwargs) if kwargs else ""}
            datasets:
              - path: examples.jsonl
                ds_type: json
                type: chat_template
                field_messages: messages
                roles_to_train: [assistant]      # the loss on the reply only
            adapter: lora
            lora_r: {s.get("lora_rank") or 32}
            lora_alpha: 32
            lora_dropout: 0.0
            lora_target_linear: true
            sequence_len: {s["max_model_len"]}
            sample_packing: {str(not any(t != "full_attention" for t in
                                         plan.get("student_info", {}).get("layer_types") or [])).lower()}
            micro_batch_size: 1
            gradient_accumulation_steps: {s["examples_per_step"]}
            num_epochs: {s["epochs"]}
            learning_rate: {s["learning_rate"]!r}
            warmup_ratio: {s["schedule"]["warmup"]}
            lr_scheduler: cosine                  # Axolotl's; functai trains with warmup-stable-decay
            bf16: auto
            gradient_checkpointing: true
            output_dir: ./outputs/lora
        '''))
        (out / "README.md").write_text(textwrap.dedent(f'''\
            # {plan["name"]}: training examples for {student}

            Written by `functai.bake(..., where="export")`.

            - `examples.parquet`: one conversation per row, with `input_ids` (the exact tokens {student}'s chat
              template writes, as the function will call it) and `answer_start` (the loss is on the tokens from
              there on). Also `messages`, `function`, `weight`, `split`.
            - `examples.jsonl`: the same conversations as chat messages, for tools that apply the template
              themselves (then the tokens depend on that tool keeping the template's options:
              {json.dumps(kwargs) or "none"}).
            - `train_trl.py`: TRL on the exact tokens, with the settings bake chose.
            - `axolotl.yaml`: Axolotl on the messages.

            Settings bake chose: LoRA rank {s.get("lora_rank")}, learning rate {s["learning_rate"]:.2e},
            {s["epochs"]} pass(es), about {s["examples_per_step"]} examples per step, replies up to
            {s["max_new_tokens"]} tokens.

            Back in functai:

            ```python
            baked = functai.bake.adopt("<trained folder>", <the function>, examples="examples.parquet")
            fast = <the function>.using(lm=baked)
            ```
        '''))
        run.update(state="exported", phase=None, output=str(out))


__all__ = ["Export"]
