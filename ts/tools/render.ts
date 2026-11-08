/**
 * A student's tokens for conversations given on stdin (tools/crosslang_training.py
 * compares them with Python's): `{student, dir, effort, items: [{messages,
 * pictures: [base64], reply}]}` → one line of JSON per item:
 * `{prompt, full, replyStart, prefixFallback}`, then the student's ends.
 *
 *     node --conditions=functai-source tools/render.ts < items.json
 */
import { readFileSync } from "node:fs";
import { picture, Student } from "../src/students.ts";

const input = JSON.parse(readFileSync(0, "utf8"));
const st = await Student.load(input.student, { dir: input.dir, effort: input.effort });
const out = input.items.map((it: { messages: Record<string, unknown>[]; pictures: string[]; reply: string | null }) => {
  try {
    const r = st.render(it.messages, it.pictures.map((b) => picture(new Uint8Array(Buffer.from(b, "base64")))), it.reply);
    return { prompt: r.prompt, full: r.full, replyStart: r.replyStart, prefixFallback: r.prefixFallback };
  } catch (err) {
    return { refused: (err as Error).message };
  }
});
process.stdout.write(JSON.stringify({ ends: { lead: st.lead, end: st.end, stop: st.stopToken, head: st.endHead, kwargs: st.kwargs, template: st.templateSha256, tokenizer: st.tokenizerSha256 }, items: out }));
