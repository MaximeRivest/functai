import { readFileSync } from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import { Template } from "@huggingface/jinja";
const dir = process.argv[2] ?? "hf/";
const tok = new Tokenizer(JSON.parse(readFileSync(dir + "tokenizer.json")), JSON.parse(readFileSync(dir + "tokenizer_config.json")));
const tpl = new Template(readFileSync(dir + "chat_template.jinja", "utf8"));
const cases = JSON.parse(readFileSync(process.argv[3] ?? "cases.json"));
const ids = (text) => tok.encode(text, { add_special_tokens: false }).ids;
let trainOk = 0, promptOk = 0, weightOk = 0, firstBad = null;
const t0 = performance.now();
for (const c of cases) {
  const full = tpl.render({ messages: c.messages, add_generation_prompt: false, reasoning_effort: c.effort });
  const prompt = tpl.render({ messages: c.messages.slice(0, -1), add_generation_prompt: false, reasoning_effort: c.effort });
  const a = ids(full), p = ids(prompt);
  if (JSON.stringify(a) === JSON.stringify(c.train_ids)) trainOk++; else firstBad ??= { full: full.slice(0, 200), a: a.slice(0, 30), b: c.train_ids.slice(0, 30) };
  if (JSON.stringify(p) === JSON.stringify(c.prompt_ids)) promptOk++;
  // the trained span: from the last <|message_model|> after the prompt to the end
  const w = a.map((_, i) => (i >= p.length ? 1 : 0));
  if (JSON.stringify(w) === JSON.stringify(c.weights)) weightOk++;
}
console.log(`${cases.length} cases in ${Math.round(performance.now() - t0)} ms: training tokens identical ${trainOk}, prompt tokens identical ${promptOk}, trained-token mask identical ${weightOk}`);
if (firstBad) console.log(firstBad);
