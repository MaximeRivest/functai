"use strict";
// FunctAI Studio, the dashboard. Reads /api/state (refreshed every 1.5 s, only when
// something changed), and acts through the same functions as the command line.

// ------------------------------------------------------------ icons (drawn: they show on every computer)
const SV = (d, fill) => `<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="${d}" ${fill ? 'fill="currentColor"' : 'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'}/></svg>`;
const I = {
  ok: SV("M3 8.5l3 3 7-7"), no: SV("M4.5 4.5l7 7M11.5 4.5l-7 7"),
  cloud: SV("M4.8 12.5h7a3 3 0 0 0 .3-6 4 4 0 0 0-7.6 1.1A2.5 2.5 0 0 0 4.8 12.5z"),
  house: SV("M2.5 8.2L8 3.5l5.5 4.7M4.2 7v6h7.6V7"),
  link: SV("M7 9a3 3 0 0 0 4.2 0l2-2A3 3 0 0 0 9 2.8l-1 1M9 7a3 3 0 0 0-4.2 0l-2 2A3 3 0 0 0 7 13.2l1-1"),
  down: SV("M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10"),
  agent: SV("M3 3.5h10v7H7l-3 2.5v-2.5H3z"),
  clock: SV("M8 2.5a5.5 5.5 0 1 0 0 11a5.5 5.5 0 0 0 0-11zM8 5v3.3l2.2 1.4"),
  loop: SV("M12.5 6A4.8 4.8 0 0 0 4 4.6L3 5.8M3 2.8v3h3M3.5 10a4.8 4.8 0 0 0 8.5 1.4l1-1.2M13 13.2v-3h-3"),
};

// ------------------------------------------------------------ state
let S = null, ETAG = "", pending = false;
const T = { view: "map", n: null, sel: null, drawer: null, storyAt: 0, wrongFor: null, watch: null, result: null,
  trial: null, check: null, request: null, what: "" };
const VIEWS = ["map", "review", "table", "summary", "code"];

// ------------------------------------------------------------ helpers
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const txt = v => v === null || v === undefined ? "" : typeof v === "string" ? v
  : Array.isArray(v) && v.every(x => typeof x === "string") ? v.map(x => "• " + x).join("\n")
  : typeof v === "boolean" ? (v ? "yes" : "no") : JSON.stringify(v, null, Array.isArray(v) || typeof v === "object" ? 1 : 0);
const words = v => { const t = txt(v).trim(); return t ? t.split(/\s+/).length : 0; };
function short(v, n = 70) { const t = txt(v).replace(/\s+/g, " ").trim(); return t.length > n ? t.slice(0, n - 1) + "…" : t; }
function cents(c) {
  if (c === null || c === undefined) return "unknown";
  if (c === 0) return "0¢";
  if (c < 0.1) return (+c.toFixed(3)) + "¢";
  if (c < 1) return (+c.toFixed(2)) + "¢";
  if (c < 100) return (+c.toFixed(1)) + "¢";
  return "$" + (c / 100).toFixed(2);
}
const pct = x => Math.round(x * 100) + "%";
const costTag = c => c === undefined || c === null ? "" : c === 0 ? " · free" : ` · about ${cents(c)}`;
function toast(t) { const e = $("#toast"); e.innerHTML = t; e.classList.add("on"); clearTimeout(toast.h); toast.h = setTimeout(() => e.classList.remove("on"), 4200); }
function provider(m) {
  m = (m || "").toLowerCase(); if (m.includes(":")) m = m.split(":")[1];
  if (m.startsWith("claude")) return "Anthropic"; if (/^(gpt|o1|o3|o4)/.test(m)) return "OpenAI";
  if (m.startsWith("gemini")) return "Google"; if (m.startsWith("grok")) return "xAI"; if (m.startsWith("llama") || m.startsWith("qwen") || m.startsWith("ollama")) return "this computer";
  return m || "?";
}
const where = m => provider(m) === "this computer" ? `${I.house} this computer` : `${I.cloud} ${esc(provider(m))}`;
function humanName(n) { n = (n || "").replace(/_/g, " "); return n.charAt(0).toUpperCase() + n.slice(1); }
function when(t) {
  if (!t) return "";
  let m = t.match(/^(\w+) == ['"](.*)['"]$/); if (m) return `if ${m[1]} is “${m[2]}”`;
  m = t.match(/^not \((\w+) == ['"](.*)['"]\)$/); if (m) return `if ${m[1]} isn't “${m[2]}”`;
  m = t.match(/^not \((.*)\)$/); if (m) return `otherwise (not ${m[1]})`;
  return `if ${t}`;
}
const stateClass = s => s === "right" ? "good-t" : s === "wrong" ? "bad-t" : s === "failed" ? "fail-t" : "mute";

window.addEventListener("unhandledrejection", e => { if (e.reason && e.reason.shown) e.preventDefault(); });   // already told on screen
async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Studio": "1" }, body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({ error: "no answer" }));
  if (r.status === 409) return { needsYes: true, ...j };
  if (!r.ok) { toast(esc(j.error || "something went wrong")); if (r.status === 400) ETAG = ""; const err = new Error(j.error); err.shown = true; throw err; }
  return j;
}

// ------------------------------------------------------------ addresses: every view has one; Back works
function urlFor() { return `#/${T.view}` + (T.n ? `?example=${encodeURIComponent(T.n)}` : ""); }
function go(push) { const u = urlFor(); if (location.hash !== u) { push ? history.pushState(null, "", u) : history.replaceState(null, "", u); } }
function readUrl() {
  const m = location.hash.match(/^#\/(\w+)(?:\?example=([^&]+))?/);
  if (m && VIEWS.includes(m[1])) T.view = m[1];
  if (m && m[2]) T.n = decodeURIComponent(m[2]);
}
window.addEventListener("popstate", () => { readUrl(); render(); });

// ------------------------------------------------------------ the data, live
async function poll() {
  try {
    const r = await fetch("/api/state?since=" + ETAG);
    const j = await r.json();
    if (r.status === 403) { $("#root").innerHTML = `<div class="loading">${esc(j.error)}</div>`; return; }
    if (j.unchanged) { if (S) { S.jobs = j.jobs; jobsChanged(); } }
    else { S = j; ETAG = j.etag; jobsChanged(); safeRender(); }
  } catch (e) { /* the server restarts: try again */ }
  setTimeout(poll, 1500);
}
// Live updates never get in the way of typing: held back while someone types (the last 2 s),
// and the field being typed in keeps its focus and cursor across a refresh.
let lastTyped = 0;
document.addEventListener("input", () => { lastTyped = Date.now(); }, true);
function safeRender() {
  const a = document.activeElement;
  if (a && /INPUT|TEXTAREA/.test(a.tagName) && a.closest("#root") && Date.now() - lastTyped < 2000) {
    pending = true; setTimeout(() => { if (pending) { pending = false; safeRender(); } }, 2100); return;
  }
  render();
}
document.addEventListener("focusout", () => setTimeout(() => { if (pending) { pending = false; render(); } }, 50));

function jobsChanged() {
  if (!S || !T.watch) { renderJobBar(); return; }
  const j = (S.jobs || []).find(j => j.id === T.watch);
  if (j && (j.state === "done" || j.state === "failed")) {
    T.watch = null;
    if (j.state === "done" && j.kind === "draft") { T.plan = j.result; T.planFromAI = true; ETAG = ""; render(); return; }
    if (j.state === "done" && j.kind === "examples") { toast(esc(j.result.what)); ETAG = ""; return; }
    if (j.state === "failed" && (j.kind === "draft" || j.kind === "examples")) { toast(esc(j.error)); ETAG = ""; renderJobBar(); return; }
    T.result = j; T.drawer = "result"; render();
    ETAG = ""; // fetch everything again: the job changed the records
  } else renderJobBar();
}
function renderJobBar() {
  const el = $("#jobbar"); if (!el || !S) return;
  const j = (S.jobs || []).find(j => j.state === "running" || j.state === "queued");
  el.innerHTML = j ? `<div class="banner job"><b>${esc(j.label)}…</b>${j.total ? `<div class="bar"><i style="width:${Math.round(100 * j.done / j.total)}%"></i></div><span>${j.done}/${j.total}</span>` : ""}</div>` : "";
}
async function startJob(path, body, label) {
  const r = await api(path, body);
  if (r.needsYes) {
    if (confirm(`${r.error}.\n\nUp to ${r.estimate.requests} model requests, about ${cents(r.estimate.cents)}. Go ahead?`))
      return startJob(path, { ...body, yes: true }, label);
    return;
  }
  T.watch = r.id; toast(esc(label || r.label) + "…"); ETAG = "";
}

// ------------------------------------------------------------ rows and paths
const rows = () => (S && S.rows) || [];
function curRow() {
  const rs = rows(); if (!rs.length) return null;
  let r = rs.find(r => String(r.n) === String(T.n));
  if (!r) { r = rs.find(r => r.ran) || rs[0]; T.n = String(r.n); }
  return r;
}
function moveRow(d) { const rs = rows(); if (!rs.length) return; const i = rs.findIndex(r => String(r.n) === String(T.n)); const r = rs[(i + d + rs.length) % rs.length]; T.n = String(r.n); T.storyAt = 0; render(); go(false); }
function pathOf(row) {
  const g = S.graph, used = new Set(), called = new Set(Object.keys((row && row.steps) || {}));
  if (!row || !row.ran) return null;
  const out = id => g.edges.filter(e => e.from === id).map(e => e.to);
  for (const n of g.nodes) {
    if (n.kind === "input" || n.kind === "output") used.add(n.id);
    else if (n.kind === "step") { if (called.has(n.id)) used.add(n.id); }
    else if (n.kind === "const") { if (out(n.id).some(x => called.has(x))) used.add(n.id); }
    else if (n.kind === "value") {
      if (n.returns) { if (!row.end.error && JSON.stringify(row.end.value) === JSON.stringify(n.value)) used.add(n.id); }
      else if (out(n.id).some(x => called.has(x)) && !(n.alt || []).some(a => called.has(a))) used.add(n.id);
    }
  }
  return used;
}
function stepDot(id) {
  const s = S.scores.steps[id]; if (!s || !s.checked) return "";
  const p = s.right / s.checked; return p === 1 ? "good" : p >= .7 ? "mid" : "bad";
}
const jword = j => !j ? "" : j.verdict === "right" ? `<span class="good-t">${I.ok}</span>` : `<span class="bad-t">${I.no}</span>`;
function judgedBy(j) {
  if (!j) return "Not checked yet.";
  const v = j.verdict === "right" ? "right" : "wrong";
  if (j.source === "person") return `Judged ${v} by ${esc(j.by)}.`;
  if (j.source === "step") return `Counted wrong because ${esc(j.because)} was judged wrong on the way.`;
  return `Counted ${v}: ${esc(j.why)}.`;
}

// ------------------------------------------------------------ the start (no program yet)
const STARTERS = [
  ["For work", "Sort wholesale inquiries", "Sort our wholesale inquiry emails into new orders, reorders and questions, and draft a reply to each for me to approve"],
  ["For work", "Reply to café reviews", "Draft a short, kind reply to each online review of our café, and flag every complaint for me"],
  ["For school", "Quiz from my notes", "Make a 5-question quiz from my biology notes, and check each answer is really in the notes"],
  ["For a club", "Sort messages", "Sort my robotics club's Discord messages into questions, ideas and spam, and answer the questions from our FAQ"],
  ["For research", "Extract from papers", "Read each paper's methods section and extract the sample size, the species and the statistical test"],
];
function agentPrompt(what) {
  return `Make me a FunctAI pipeline: ${what || "…describe it here…"}

Work in ${S.folder || "this folder"}. Use FunctAI (Python) and its Studio command line: read \`functai studio --help\` first. Before writing code, tell me your plan in plain words and ask me. Then write the pipeline, write example inputs in examples/<name>.csv (never their right answers), run them with \`functai studio run\`, and tell me to open the dashboard to judge the answers.`;
}
// ------------------------------------------------------------ the map
function viewMap() {
  const g = S.graph, row = curRow(), used = pathOf(row);
  let h = picker(row);
  const W = g.width, H = g.height;
  h += `<div class="mapwrap" id="mapwrap" style="width:${W}px;height:${H}px"><svg width="${W}" height="${H}">`;
  const N = Object.fromEntries(g.nodes.map(n => [n.id, n]));
  for (const e of g.edges) {
    const a = N[e.from], b = N[e.to]; if (!a || !b) continue;
    const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2, m = (x1 + x2) / 2;
    const on = used && used.has(a.id) && used.has(b.id);
    h += `<path class="wire ${on ? "on" : ""}" d="M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}"/>`;
  }
  h += `</svg>`;
  for (const n of g.nodes) {
    const off = used && !used.has(n.id) ? "off" : "", pos = `left:${n.x}px;top:${n.y}px;width:${n.w}px;height:${n.h}px`;
    const sel = T.sel === n.id ? "sel" : "";
    if (n.kind === "input") {
      h += `<div class="node input ${sel}" style="${pos}" data-node="${esc(n.id)}"><div class="h">${esc(n.label)}</div><div class="val">${row ? esc(short(row.inputs[n.label], 110)) : '<span class="mute">an input</span>'}</div></div>`;
    } else if (n.kind === "const") {
      h += `<div class="node const ${off} ${sel}" style="${pos}" data-node="${esc(n.id)}"><div class="h" style="font-size:13.5px">${esc(n.label)}</div><div class="small">${esc(n.sub || "")}</div></div>`;
    } else if (n.kind === "value") {
      h += `<div class="node value ${off} ${sel}" style="${pos}" data-node="${esc(n.id)}"><div class="h" style="font-weight:600;font-size:13.5px">${n.returns ? "returns " : ""}${esc(n.label)}</div><div class="when">${esc(when(n.when))}</div></div>`;
    } else if (n.kind === "output") {
      const st = row && row.ran ? row.state : null;
      h += `<div class="node output ${off} ${sel}" style="${pos}" data-node="out"><div class="h" style="font-weight:600;color:var(--mute);font-size:13px">end result</div>${st ? `<div class="big ${stateClass(st)}">${esc(st)}</div>` : `<div class="val none">not run yet</div>`}</div>`;
    } else {
      const c = row && row.steps ? row.steps[n.id] : null, info = S.steps[n.id] || {};
      const model = (c && c.model) || info.lm;
      let cls = "", val;
      if (!row || !row.ran) val = `<div class="val none">${row ? "not run yet" : "no example yet"}</div>`;
      else if (!c) val = `<div class="val none">not used here</div>`;
      else if (c.error) { cls = "failed"; val = `<div class="val fail-t">couldn't run: ${esc(short(c.error, 80))}</div>`; }
      else {
        const j = c.judgement; cls = j ? "ok-" + j.verdict : "";
        val = `<div class="val" title="${words(c.value)} words">${jword(j)} ${esc(txt(c.value) === "" ? "(nothing)" : txt(c.value))}</div>`;
      }
      const what = (info.instruction || "").split(/(?<=[.?!])\s/)[0];
      h += `<div class="node ${cls} ${off} ${sel}" style="${pos}" data-node="${esc(n.id)}"><div class="h"><span class="dot ${stepDot(n.id)}"></span>${esc(n.label)}${n.loop ? ` <span class="loopb">${I.loop} repeats</span>` : ""}<span class="where" title="${esc(model)}">${where(model)}</span></div>${what ? `<div class="what" title="${esc(info.instruction)}">${esc(what)}</div>` : ""}${n.when ? `<div class="when">${esc(when(n.when))}</div>` : ""}${val}</div>`;
    }
  }
  h += `</div>`;
  if (g.notes && g.notes.length) h += `<div class="notes small">${g.notes.map(esc).join("<br>")}</div>`;
  return h;
}
function picker(row) {
  const rs = rows();
  if (!rs.length) return `<div class="summary">No examples yet. <button class="btn small primary" data-act="toexamples">Add examples</button> (or write <code>${esc(S.examples_file)}</code>, a CSV with a column for ${S.program.inputs.map(i => `<code>${esc(i)}</code>`).join(", ")})</div>`;
  const i = rs.findIndex(r => r === row);
  return `<div class="picker"><button data-act="prev" title="Previous (k)">‹</button><span>${row.source === "use" ? "Run from elsewhere" : "Example"} ${i + 1} of ${rs.length}</span><button data-act="next" title="Next (j)">›</button>${row.ran && !row.current ? `<span class="small">· answered by version ${row.version}, not the current one</span>` : ""}${!row.ran ? `<button class="btn small" data-act="runone">Run it</button>` : ""}</div>`;
}

// ------------------------------------------------------------ the table
function viewTable() {
  const sc = S.scores.end, steps = Object.keys(S.steps);
  const ins = S.program.inputs.slice(0, 2);
  let h = `<div class="summary"><span>${rows().length} rows · <span class="good-t">${sc.right} right</span> · <span class="bad-t">${sc.wrong} wrong</span> · <span class="fail-t">${sc.failed} failed</span> · ${sc.unchecked} not checked${sc.not_run ? ` · ${sc.not_run} not run` : ""}</span><span class="small">Click any cell to see it whole and judge it.</span></div>`;
  h = examplesBar() + h;
  if (!rows().length) return h + `<div class="card" style="text-align:center">No examples yet. Add a few: typical ones, hard ones, odd ones. The AI can write some too.</div>`;
  h += `<table class="t"><thead><tr><th>#</th>${ins.map(i => `<th>${esc(i)}</th>`).join("")}${steps.map(s => `<th><span class="dot ${stepDot(s)}"></span> ${esc(s)}</th>`).join("")}<th>end</th></tr></thead><tbody>`;
  for (const r of rows()) {
    const sel = String(r.n) === String(T.n) ? "sel" : "", old = r.ran && !r.current ? "old" : "";
    h += `<tr class="${sel} ${old}"><td class="small"><a data-act="toreview" data-n="${esc(r.n)}" title="Read everything, and judge">${esc(r.n)}</a></td>`;
    for (const i of ins) h += `<td data-cell="${esc(r.n)}:in:${esc(i)}"><div class="c">${esc(short(r.inputs[i], 140))}</div></td>`;
    for (const s of steps) {
      const c = (r.steps || {})[s];
      if (!r.ran) { h += `<td class="small" data-cell="${esc(r.n)}:${esc(s)}"></td>`; continue; }
      if (!c) { h += `<td class="small" style="color:var(--faint)" data-cell="${esc(r.n)}:${esc(s)}">—</td>`; continue; }
      if (c.error) { h += `<td class="failed" data-cell="${esc(r.n)}:${esc(s)}"><div class="c">failed</div></td>`; continue; }
      const v = (c.judgement || {}).verdict || "";
      h += `<td class="${v}" data-cell="${esc(r.n)}:${esc(s)}"><div class="cell"><div class="c">${esc(short(c.value, 140) || "(nothing)")}</div>${jword(c.judgement)}</div>${words(c.value) > 18 ? `<span class="len">${words(c.value)} words</span>` : ""}</td>`;
    }
    h += `<td data-cell="${esc(r.n)}:out"><b class="${stateClass(r.state)}">${esc(r.state)}</b></td></tr>`;
  }
  return h + `</tbody></table>`;
}

// ------------------------------------------------------------ the code
function hilite(line, st) {
  const t = line.trim();
  if (st.doc || /^[rbuf]*"""/.test(t) || /^[rbuf]*'''/.test(t)) {
    const n = (line.match(/"""|'''/g) || []).length;
    if (st.doc ? n >= 1 : n === 1) st.doc = !st.doc;
    return `<span class="str">${esc(line)}</span>`;
  }
  const re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(from|import|def|return|if|elif|else|for|in|while|with|as|class|try|except|not|and|or|is|None|True|False|lambda|yield|async|await)\b|\b(str|int|float|bool|list|dict|Literal|Optional)\b|(@\w+)/g;
  let o = "", i = 0, m;
  while ((m = re.exec(line))) { o += esc(line.slice(i, m.index)); o += `<span class="${m[1] ? "cm" : m[2] ? "str" : m[3] ? "kw" : m[4] ? "ty" : "fn"}">${esc(m[0])}</span>`; i = m.index + m[0].length; }
  return o + esc(line.slice(i));
}
function stepLines(lines, start) {
  if (!start) return [-1, -1];
  let i = start - 1;
  while (i < lines.length && /^\s*@|^(async\s+)?def\s/.test(lines[i])) i++;
  while (i < lines.length && (lines[i].trim() === "" || /^\s/.test(lines[i]))) i++;
  while (i > start && lines[i - 1].trim() === "") i--;
  return [start, i];
}
function viewCode() {
  const code = S.code || { text: "" }, lines = code.text.split("\n"), st = { doc: false };
  const changed = new Set(code.changed || []);
  const info = S.steps[T.sel] || {}; const [a, b] = stepLines(lines, info.def_line);
  if (T.codeEdit) {
    const cur = draft("code", code.text);
    return `<div class="codehead"><b style="color:var(--ink)">${esc(S.program.file)}</b> · editing<span style="margin-left:auto"></span><button class="btn primary" data-act="savecode" ${cur === code.text ? "disabled" : ""}>Save</button><button class="btn" data-act="cancelcode">Cancel</button></div>
      <textarea class="in codeedit" data-draft="code" spellcheck="false">${esc(cur)}</textarea>`;
  }
  return `<div class="codehead"><button class="btn small" data-act="editcode">Edit the code</button><b style="color:var(--ink)">${esc(S.program.file)}</b> · version ${S.program.version}${changed.size ? ` · <span class="good-t">green: changed since the last commit</span>` : code.tracked ? " · no change since the last commit" : ""}<span style="margin-left:auto">This file is the program. Studio reads it; your agent and you change it.</span></div>
  <pre class="code">${lines.map((l, i) => `<span class="ln ${changed.has(i + 1) ? "add" : (i + 1 >= a && i + 1 <= b) ? "hl" : ""}"><span class="no">${i + 1}</span>${hilite(l, st) || " "}</span>`).join("")}</pre>`;
}

// ------------------------------------------------------------ the summary
function viewSummary() {
  const sw = S.score_words || ["", false];
  let h = `<div class="story"><div class="card"><h2>${esc(humanName(S.program.name))}, in one page</h2><div class="facts">${(S.facts || []).map(([k, v]) => `<div class="k">${esc(k)}</div><div>${esc(v)}</div>`).join("")}</div>
    <div class="row" style="margin-top:14px"><a class="btn primary" href="/api/page" download>${I.down} Download this summary</a><button class="btn" data-act="copylink">${I.link} Copy link</button>${sw[1] ? `<button class="btn" data-act="check">Check 10 more for a reliable score</button>` : ""}</div>
    <div class="privacy">One HTML file with these facts, a drawing of the pipeline and three examples: it opens in any browser, needs no account, and can be attached to an e-mail or printed to PDF. No AI wrote it: it comes from the records.</div></div>`;
  const row = curRow();
  if (!row || !row.ran) return h + `</div>`;
  const fr = [{ q: "It was given", t: Object.entries(row.inputs).map(([k, v]) => `${k}: ${txt(v)}`).join("\n") }];
  const order = S.graph.nodes.filter(n => n.kind === "step").map(n => n.id);
  for (const s of order) {
    const c = row.steps[s]; if (!c) continue;
    fr.push({ q: `Step “${s}” was asked: ${(S.steps[s].instruction || "").replace(/\s+/g, " ")}`, t: c.error ? "It couldn't: " + c.error : (txt(c.value) || "(nothing)"),
      cls: c.error ? "failed" : (c.judgement || {}).verdict || "", fix: (c.judgement || {}).answer });
  }
  fr.push({ q: "End result", t: row.end.error ? "It failed." : (txt(row.end.value) || "(nothing)"), cls: row.state, fix: ((row.end || {}).judgement || {}).answer });
  const at = Math.min(T.storyAt, fr.length - 1);
  h += `<h3 style="margin:22px 0 4px;font-size:15px">Follow one example <span class="small">· #${esc(row.n)}</span></h3>`;
  fr.slice(0, at + 1).forEach(f => h += `<div class="frame ${f.cls || ""}"><div class="q">${esc(f.q)}</div><div style="white-space:pre-wrap">${esc(f.t)}</div>${f.cls === "wrong" && f.fix !== undefined && f.fix !== null ? `<div class="should">Someone said it should be: ${esc(txt(f.fix))}</div>` : ""}</div>`);
  h += `<div class="row"><button class="btn" data-act="sback" ${at ? "" : "disabled"}>← Back</button><button class="btn primary" data-act="snext" ${at < fr.length - 1 ? "" : "disabled"}>Next step →</button><button class="btn" data-act="next">Another example</button></div></div>`;
  return h;
}

// ------------------------------------------------------------ the drawer
function judgeBlock(cell, choices) {
  if (!cell || !cell.call) return "";
  const j = cell.judgement, mine = j && j.source === "person";
  let h = `<div class="acts"><button class="btn ${j && j.verdict === "right" ? "on-good" : ""}" data-act="right" data-call="${cell.call}">${I.ok} Right</button><button class="btn ${j && j.verdict === "wrong" ? "on-bad" : ""}" data-act="wrong" data-call="${cell.call}">${I.no} Wrong</button>${mine ? `<button class="btn small" data-act="withdraw" data-call="${cell.call}">Withdraw</button>` : ""}</div>
    <div class="small" style="margin-top:6px">${judgedBy(j)}</div>`;
  if (j && j.verdict === "wrong" && j.answer !== undefined && j.answer !== null) h += `<div class="should">Should be: ${esc(txt(j.answer))}</div>`;
  if (T.wrongFor === cell.call) {
    h += `<div class="sec"><div class="lbl">What should it have said?</div>`;
    if (choices && choices.length) h += `<div class="acts">${choices.map(c => `<button class="btn" data-act="answer" data-call="${cell.call}" data-answer="${esc(JSON.stringify(c))}">${esc(c)}</button>`).join("")}</div>`;
    else h += `<textarea class="in" id="answerbox" rows="3" placeholder="The right answer (optional)"></textarea><div class="acts"><button class="btn primary" data-act="answertext" data-call="${cell.call}">Save</button></div>`;
    h += `<div class="acts"><button class="btn small" data-act="answer" data-call="${cell.call}" data-answer="null">I don't know the right answer</button></div></div>`;
  }
  return h;
}
function drawerNode() {
  const row = curRow(), id = T.sel, n = S.graph.nodes.find(n => n.id === id) || { kind: id === "out" ? "output" : "step" };
  if (n.kind === "input") return exampleDrawer(row);
  if (n.kind === "const") {
    const file = ((S.wiring || {}).files || {})[n.label];
    if (file === undefined) return [esc(n.label), "", `<div class="sec"><div class="small">A value set in ${esc(S.program.file)} (${esc(n.sub)}), given to the steps it feeds. Change it in the Code view.</div></div>`];
    const orig = (S.files || {})[n.label] || "", empty = orig.startsWith("(Write "), cur = draft("file:" + n.label, empty ? "" : orig);
    return [esc(n.label), `read from ${esc(file)}`, `<div class="sec"><div class="small">Given to the steps it feeds, as it is written in <code>${esc(file)}</code>.</div>
      <textarea class="in" rows="14" data-draft="file:${esc(n.label)}" ${empty ? `placeholder="${esc(orig.replace(/^\(|\)$/g, "").split(". Until")[0])}."` : ""} style="margin-top:6px;font-family:ui-monospace,Menlo,monospace;font-size:13px">${esc(cur)}</textarea>
      ${cur.trim() && cur !== orig ? `<div class="acts"><button class="btn primary" data-act="savefile" data-name="${esc(n.label)}">Save ${esc(file)}</button><button class="btn small" data-act="dropdraft" data-key="file:${esc(n.label)}">Cancel</button></div>` : ""}</div>`];
  }
  if (n.kind === "value") {
    const orig = txt(n.value), cur = draft("val:" + n.id, orig);
    return ["A fixed value", "", `<div class="sec"><div class="small">${esc(when(n.when))}: the code ${n.returns ? "returns this" : "uses this instead of asking a model"}.</div>
      ${n.editable ? `<textarea class="in" rows="3" data-draft="val:${esc(n.id)}" style="margin-top:6px">${esc(cur)}</textarea>${cur !== orig ? `<div class="acts"><button class="btn primary" data-act="savevalue" data-id="${esc(n.id)}">Save</button><button class="btn small" data-act="dropdraft" data-key="val:${esc(n.id)}">Cancel</button></div>` : ""}` : `<div class="full">${esc(orig)}</div>`}</div>`];
  }
  if (n.kind === "output") {
    if (!row || !row.ran) return ["End result", "", `<div class="small">Not run yet.</div><div class="sec"><div class="acts"><button class="btn small" data-act="addopen" data-after="(end)">Add a step at the end</button></div>${T.addAfter === "(end)" ? addForm("(end)") : ""}</div>`];
    const e = row.end;
    let h = `<div class="sec"><div class="full">${e.error ? "Failed: " + esc(e.error) : esc(txt(e.value) || "(nothing)")}</div><p class="${stateClass(row.state)}"><b>${esc(row.state)}</b></p>`;
    if (!e.error) h += judgeBlock({ call: row.call, judgement: e.judgement }, null);
    if (row.state === "wrong") h += `<div class="acts"><button class="btn primary" data-act="why">Which step is at fault?${costTag(((S.estimates || {}).why || {})[String(row.n)])}</button></div><div class="small">Studio gives each step judged wrong its right answer, reruns only what comes after, and sees whether the end becomes right.</div>`;
    h += `<div class="sec"><div class="acts"><button class="btn small" data-act="addopen" data-after="(end)">Add a step at the end</button></div>${T.addAfter === "(end)" ? addForm("(end)") : ""}</div>`;
    return ["End result", `example ${esc(row.n)}`, h + `</div>`];
  }
  const info = S.steps[id] || {}, c = row && row.steps ? row.steps[id] : null, model = (c && c.model) || info.lm, who = provider(model);
  let h = editBlocks(id) + `<div class="privacy">${who === "this computer" ? "Runs on this computer: nothing leaves it." : `Sends what it reads to ${esc(who)} to get an answer. What ${esc(who)} keeps depends on its rules and your account with it.`}</div>`;
  h += `<div class="sec"><div class="lbl">On example ${row ? esc(row.n) : ""}</div>`;
  if (!row || !row.ran) h += `<div class="small">Not run yet.</div>`;
  else if (!c) h += `<div class="small">This example didn't go through ${esc(id)}.</div>`;
  else if (c.error) h += `<div class="full fail-t">Couldn't run: ${esc(c.error)}</div>`;
  else h += (c.model && info.lm && c.model.split(":").pop() !== info.lm.split(":").pop() ? `<div class="small" style="color:var(--warn);margin-bottom:4px">Answered by ${esc(c.model)}; the code now names ${esc(info.lm)}. Run it again to see what that gives.</div>` : "") + `<div class="full">${esc(txt(c.value) || "(nothing)")}</div><div class="small" style="margin-top:4px">${words(c.value) > 18 ? words(c.value) + " words · " : ""}${esc(c.model || "")} · ${(c.seconds || 0).toFixed(1)} s · ${c.cached ? "from the cache, free" : cents(c.cents)}${c.more ? ` · asked ${c.more + 1} times here` : ""}</div>` + judgeBlock(c, info.choices);
  if (row && row.ran) h += `<div class="acts"><button class="btn small" data-act="toreview">Read everything it saw, in Review</button></div>`;
  const sc = S.scores.steps[id] || { right: 0, checked: 0 };
  h += `</div><div class="sec"><div class="lbl">How often it's right</div><div>${!sc.checked ? "Nobody has checked it yet." : sc.checked < 10 ? `Right ${sc.right} of the ${sc.checked} times it was checked. Check ${10 - sc.checked} more for a reliable score.` : `Right ${pct(sc.right / sc.checked)} of the time (likely ${pct(sc.low)}–${pct(sc.high)}).`}</div></div>
    <div class="row"><a data-act="incode">Show in code</a>${c && c.call ? `<a data-act="request" data-call="${c.call}">The exact request</a>` : ""}</div>`;
  if (T.request && T.request.id === (c || {}).call) h += requestBlock(T.request);
  return [esc(id), where(model), h];
}
function priceLine(m) { const p = S.models.prices[m]; return p ? `List price: $${p.input} in / $${p.output} out per million tokens.` : "No price is known for it: Studio will count its tokens but not its cost."; }
function modelOptions(cur) {
  const seen = new Set(), opts = [];
  for (const m of [cur, ...S.models.suggested, ...S.models.priced]) if (m && !seen.has(m)) { seen.add(m); opts.push(m); }
  return opts.map(m => `<option ${m === cur ? "selected" : ""}>${esc(m)}</option>`).join("") + `<option value="__other">another model…</option>`;
}
function requestBlock(rec) {
  const ex = (rec.exchanges || [])[0] || {}, req = ex.request || {}, res = ex.response || {};
  const parts = m => (m.parts || []).map(p => p.text || `[${p.type}]`).join("");
  const resText = res.message ? parts(res.message) : (res.parts ? parts(res) : JSON.stringify(res).slice(0, 2000));
  return `<div class="sec"><div class="lbl">Sent to ${esc(ex.model || "")}</div><div class="full">${req.system ? `<b>system</b>\n${esc(req.system)}\n\n` : ""}${(req.messages || []).map(m => `<b>${esc(m.role)}</b>\n${esc(parts(m))}`).join("\n\n")}</div>
    <div class="lbl" style="margin-top:8px">It replied</div><div class="full">${esc(resText)}</div><div class="small">${ex.usage ? `${ex.usage.input_tokens} tokens in, ${ex.usage.output_tokens} out` : ""}${ex.cached ? " · from the cache" : ""}</div></div>`;
}
function drawerCosts() {
  const m = S.money, pm = m.per_message_cents, st = S.settings;
  const models = {}; for (const [k, v] of Object.entries(S.steps)) models[k] = v.lm;
  return ["What it costs", "", `<div class="sec"><div class="lbl">So far, in this project</div><div style="font-size:22px;font-weight:700">${cents(m.spent_cents)}</div>
    <div class="small">Paid to the AI companies with your own accounts. Studio itself is free and open source: no plans, no subscription. Replies already asked come back from the cache, free.</div>
    ${Object.keys(m.spent_by || {}).length ? `<table class="cmp">${Object.entries(m.spent_by).filter(([, c]) => c > 0).map(([k, c]) => `<tr><td>${esc(k)}</td><td>${cents(c)}</td></tr>`).join("")}</table>` : ""}</div>
    ${accountsBlock()}
    <div class="sec"><div class="lbl">Each message</div><div>${pm === null ? "Not known yet: run the examples first." : `About <b>${cents(pm)}</b> with today's AI for each step.`}</div>
    <table class="cmp">${Object.entries(models).map(([k, v]) => { const p = S.models.prices[v]; return `<tr><td>${esc(k)}</td><td>${esc(v)}</td><td>${p ? `$${p.input} / $${p.output}` : "price unknown"}</td></tr>`; }).join("")}</table>
    <div class="small">List prices per million tokens in / out, of ${esc(m.prices_date)} (${esc(m.prices_source)}).</div></div>
    <div class="sec"><div class="lbl">A month</div><div>If it runs on <input class="num" id="monthly" type="number" min="0" step="50" value="${st.monthly}"> messages a month: about <b id="monthcost">${pm === null ? "unknown" : cents(pm * st.monthly)}</b>.</div>
    <div class="small" style="margin-top:4px">From the length of the examples and their answers so far. Longer messages cost more; messages that stop early (spam, say) cost less.</div></div>
    <div class="sec"><div class="lbl">Your limit</div><div>Ask me before any run that would cost more than $<input class="num" id="cap" type="number" min="0" step="0.5" value="${st.cap_dollars}">.</div>
    <div class="small" style="margin-top:4px">Every button that asks the AI shows its cost; anything above this waits for your yes. (The command line has its own budget for agents: --budget.)</div></div>
    ${m.unknown_prices.length ? `<div class="small">No price is known for ${esc(m.unknown_prices.join(", "))}: their cost is not counted.</div>` : ""}`];
}
function drawerCheck() {
  const c = T.check;
  const dots = `<div class="dots">${c.queue.map((_, i) => `<i class="${i < c.done.length ? (c.done[i] === "right" ? "r" : c.done[i] === "wrong" ? "w" : "") : i === c.i ? "now" : ""}"></i>`).join("")}</div>`;
  if (c.i >= c.queue.length) {
    const sw = S.score_words || [""];
    return ["Checking", "", `${dots}<p><b>Done.</b> ${esc(sw[0])}</p><p class="small">Each check is recorded under your name, as one random sample. The score uses only what people judged, never an AI's opinion of itself.</p><div class="acts"><button class="btn primary" data-act="closedrawer">Back to the pipeline</button>${sw[1] ? `<button class="btn" data-act="check">Check 10 more</button>` : ""}</div>`];
  }
  const r = c.queue[c.i];
  return ["Checking", `${c.i + 1} of ${c.queue.length}`, `<div class="small">One at a time: is the end result right for this input? Keys: 1 right, 2 wrong, s skip.</div>${dots}
    <div class="lbl">It was given</div><div class="full">${esc(Object.entries(r.inputs).map(([k, v]) => (S.program.inputs.length > 1 ? k + ": " : "") + txt(v)).join("\n"))}</div>
    <div class="lbl" style="margin-top:12px">It answered</div><div class="full">${esc(txt(r.end.value) || "(nothing)")}</div>
    <div class="bigbtns"><button class="btn" data-act="cright">${I.ok} Right</button><button class="btn" data-act="cwrong">${I.no} Wrong</button></div><div class="acts"><button class="btn small" data-act="cskip">Skip</button></div>`];
}
function drawerResult() {
  const j = T.result; if (!j) return ["", "", ""];
  if (j.state === "failed") return ["It didn't work", "", `<div class="full bad-t">${esc(j.error)}</div>`];
  const r = j.result || {};
  if (j.kind === "run") {
    const s = S.scores.end;
    return ["Run finished", "", `<p class="say">Ran ${r.examples} example${r.examples === 1 ? "" : "s"} in ${(r.seconds || 0).toFixed(1)} s: ${r.failed} failed.${r.trial ? ` (A trial: ${esc(r.trial.lm)} for ${esc(r.trial.step)}; the code did not change.)` : ""}</p>
      <p>${s.right} right, ${s.wrong} wrong, ${s.failed} failed, ${s.unchecked} not checked yet.</p>${s.unchecked ? `<div class="acts"><button class="btn primary" data-act="check">Check the answers</button></div><div class="small">Only a person can say which answers are right.</div>` : ""}`];
  }
  if (j.kind === "why") {
    let h = `<p class="say">${esc(r.say)}</p>`;
    for (const t of r.tried || []) if (t.result === "someone has to look") {
      h += `<div class="sec"><div class="small">When “${esc(t.step)}” was given “${esc(short(t.given, 80))}”, the end became:</div><div class="full">${esc(txt(t.end))}</div>
        <div class="acts"><button class="btn" data-act="wifright" data-end="${esc(JSON.stringify(t.end))}" data-n="${esc(r.n)}">${I.ok} That would be right</button><button class="btn" data-act="wifwrong" data-end="${esc(JSON.stringify(t.end))}" data-n="${esc(r.n)}">${I.no} Still wrong</button></div></div>`;
    }
    return ["Which step is at fault", `#${esc(r.n)}`, h];
  }
  if (j.kind === "blame") {
    const max = Math.max(1, ...Object.values(r.counts || {}));
    return ["Which step to fix first", "", `<p class="say">${esc(r.say)}</p>${Object.entries(r.counts || {}).map(([k, v]) => `<div class="row" style="margin:6px 0"><b style="width:90px">${esc(k)}</b><div class="bar" style="max-width:none;background:#eee"><i style="width:${100 * v / max}%;background:var(--bad)"></i></div><span class="small">${v} of ${r.wrong}</span></div>`).join("")}
      <div class="small" style="margin-top:10px">To fix it, ask your coding agent: “${esc(S.program.file)}: ${esc(Object.keys(r.counts || {})[0] || "a step")} is the step that fails most (functai studio blame). Improve it and measure the change with functai studio compare --before HEAD.”</div>`];
  }
  if (j.kind === "compare") {
    const verdict = v => v || "not judged";
    const settled = (r.rows || []).filter(x => x.before && x.after && x.before !== x.after);
    const open = (r.rows || []).filter(x => !x.before || !x.after);
    let h = `<p class="say">${esc(r.say)}</p>`;
    if (settled.length) h += `<table class="cmp"><tr><th>#</th><th>before</th><th>after</th></tr>${settled.map(x => `<tr><td>${esc(x.n)}</td><td class="${stateClass(x.before)}">${esc(x.before)}</td><td class="${stateClass(x.after)}">${esc(x.after)}</td></tr>`).join("")}</table>`;
    if (open.length) {
      h += `<div class="sec"><div class="lbl">New answers nobody has judged (${open.length})</div><div class="small">Judge them, then compare again: only people say what is right.</div>`;
      for (const x of open) {
        const row = rows().find(z => String(z.n) === String(x.n)) || { inputs: {} };
        const side = (label, v, val, which) => `<div style="flex:1;min-width:0"><div class="small">${label} · <span class="${stateClass(v)}">${esc(verdict(v))}</span></div><div class="full" style="max-height:140px">${esc(txt(val) || "(nothing)")}</div>${!v ? `<div class="acts"><button class="btn small" data-act="cmpright" data-n="${esc(x.n)}" data-end="${esc(JSON.stringify(val))}">${I.ok} Right</button><button class="btn small" data-act="cmpwrong" data-n="${esc(x.n)}" data-end="${esc(JSON.stringify(val))}">${I.no} Wrong</button></div>` : ""}</div>`;
        h += `<div class="frame"><div class="q">#${esc(x.n)} · ${esc(short(Object.values(row.inputs)[0], 80))}</div><div class="row" style="align-items:flex-start">${side("before", x.before, x.before_value)}${side("after", x.after, x.after_value)}</div></div>`;
      }
      h += `<div class="acts"><button class="btn" data-act="recompare">Compare again</button></div></div>`;
    }
    if (T.trial) h += `<div class="acts"><button class="btn primary" data-act="usemodel">Use ${esc(T.trial.model)} for ${esc(T.trial.step)} (changes the code)</button></div>`;
    T.lastCompare = j.result && j.result.request;
    return ["Comparison", "", h];
  }
  return [esc(j.label), "", `<pre class="full">${esc(JSON.stringify(r, null, 1))}</pre>`];
}
function drawerVersions() {
  const vs = [...S.program.versions].reverse();
  let h = `<p class="small">Each change to the code that changes what the AI is asked is a new version. Studio keeps each version's code the first time it sees it, so you can go back.</p>`;
  for (const v of vs) {
    const score = v.ran ? `${v.right} right, ${v.wrong} wrong${v.failed ? `, ${v.failed} failed` : ""}${v.unchecked ? `, ${v.unchecked} not checked` : ""} (of ${v.ran} run)` : "not run";
    h += `<div class="frame ${v.current ? "right" : ""}"><div class="row"><b>Version ${v.label}</b>${v.current ? `<span class="small">· the code now</span>` : ""}<span class="small" style="margin-left:auto">${esc((v.first_run || "").slice(0, 16).replace("T", " "))}</span></div>
      <div class="small" style="margin-top:2px">${esc(score)}</div>
      ${v.current ? "" : v.code_kept ? `<div class="acts"><button class="btn small" data-act="seeversion" data-hash="${esc(v.hash)}">What's different</button><button class="btn small" data-act="restore" data-hash="${esc(v.hash)}">Go back to this one</button></div>` : `<div class="small">Its code was not kept (it ran before Studio kept versions); git may have it.</div>`}
      ${T.version && T.version.hash === v.hash ? (T.version.error ? `<div class="small bad-t">${esc(T.version.error)}</div>` : `<div class="diff" style="margin-top:8px">${(T.version.diff || []).map(l => `<div class="${l.startsWith("+") ? "a" : l.startsWith("-") ? "d" : ""}">${esc(l)}</div>`).join("") || `<div>Same code as now (only a model or a value outside the AI steps differs).</div>`}</div><div class="small">− that version · + the code now</div>`) : ""}</div>`;
  }
  return ["Versions", "", h];
}
function drawerActivity() {
  return ["Who did what", "", `<div class="act-list">${(S.activity || []).map(a => `<div><b>${esc(a.who)}</b> ${esc(a.what)} <span class="small">· ${esc((a.at || "").replace("T", " ").slice(0, 16))}</span></div>`).join("") || `<div class="small">Nothing yet.</div>`}</div>`];
}
function drawerAgent() {
  const f = S.program.file, top = S.blame && Object.keys(S.blame.counts || {})[0];
  const asks = [
    top ? `${f}: ${top} is the step that fails most (functai studio blame). Improve its instruction, then measure the change with functai studio compare --before HEAD and tell me the result in plain words.` : null,
    `Look at the examples judged wrong in ${f} (functai studio show --wrong) and tell me, in plain words, what goes wrong and what you would change. Don't change anything yet.`,
    `Add 10 more varied example inputs to ${S.examples_file} (hard and unusual ones too, never their right answers) and run them with functai studio run.`,
    `Try another model for each step of ${f} on the judged examples (functai studio compare --lm step=model) and tell me which is best for its cost.`,
  ].filter(Boolean);
  return ["Work with your coding agent", "", `<p>Your coding agent (Claude Code, Pi, Codex…) changes the code; Studio shows what each change does, live. It uses <code>functai studio</code> (it can read <code>functai studio --help</code>). Some things to ask it:</p>
    ${asks.map((a, i) => `<div class="prompt">${esc(a)}</div><div class="acts"><button class="btn small" data-act="copyask" data-i="${i}">Copy</button></div>`).join("")}
    <p class="small">Only you judge which answers are right. If you tell your agent a judgement, it records it under your name (functai studio rate … --by you).</p>`].concat([asks]);
}

// ------------------------------------------------------------ render
function render() {
  const a = document.activeElement, key = a && a.dataset && a.dataset.draft, sel = key ? [a.selectionStart, a.selectionEnd] : null;
  renderNow();
  if (key) {
    const el = document.querySelector(`[data-draft="${CSS.escape(key)}"]`);
    if (el && el !== document.activeElement) { el.focus(); try { el.setSelectionRange(sel[0], sel[1]); } catch (_) { } }
  }
}
function renderNow() {
  if (!S) return;
  if (!S.project) {
    $("#root").innerHTML = viewHome(); renderJobBar();
    if (T.draftEst === undefined && readyConns().length) { T.draftEst = null; api("/api/draft/estimate").then(r => { T.draftEst = r.cents; render(); }).catch(() => { }); }
    return;
  }
  if (!S.program) { $("#root").innerHTML = `<div class="loading">${esc(S.error || "Loading…")}</div>`; return; }
  curRow();
  const sw = S.score_words || ["", false], sc = S.scores.end;
  const status = !rows().some(r => r.ran) ? "not run yet" : sc.checked ? `${sc.right} of ${sc.checked} checked are right${sc.checked >= 15 ? ` (likely ${pct(sc.low)}–${pct(sc.high)})` : ""}` : "not checked yet";
  const views = { map: viewMap, review: viewReview, summary: viewSummary, table: viewTable, code: viewCode };
  let dr = null;
  if (T.drawer === "node" && T.sel) dr = drawerNode();
  else if (T.drawer === "costs") dr = drawerCosts();
  else if (T.drawer === "check" && T.check) dr = drawerCheck();
  else if (T.drawer === "result") dr = drawerResult();
  else if (T.drawer === "activity") dr = drawerActivity();
  else if (T.drawer === "agent") dr = drawerAgent();
  else if (T.drawer === "versions") dr = drawerVersions();
  else if (T.drawer === "models") dr = drawerModels();
  else if (T.drawer === "accounts") dr = drawerAccounts();
  const scroll = { stage: ($(".stage") || {}).scrollTop || 0, db: ($(".db") || {}).scrollTop || 0 };
  $("#root").innerHTML = `<div class="app">
    <div class="top"><button class="btn small" data-act="closeproj" title="All your pipelines, and making a new one">← Pipelines</button><span class="name">${esc(humanName(S.program.name))}</span><button class="ver" data-act="versions" title="Versions: their scores, and going back to one">Version ${S.program.version} ▾</button>
      <button class="btn small" data-act="copylink" title="Copy a link to this view">${I.link}</button>
      <span class="status">${esc(status)}${sw[1] && rows().some(r => r.ran) ? ` · <a data-act="check">check more</a>` : ""}</span>
      <button class="btn small" data-act="accounts">Accounts</button>
      <button class="btn small" data-act="models">Models</button>
      <button class="btn small" data-act="costs" title="What it costs, and your limit">${cents(S.money.spent_cents)} spent</button>
      ${lastEdit() ? `<button class="btn small" data-act="undo" title="Undo: ${esc(lastEdit().what)}">↶ Undo</button>` : ""}
      <button class="btn small" data-act="activity" title="Who did what">${I.clock}</button>
      <div class="seg">${[["map", "Map"], ["review", "Review"], ["table", "Table"], ["summary", "Summary"], ["code", "Code"]].map(([v, l]) => `<button data-v="${v}" class="${T.view === v ? "on" : ""}">${l}</button>`).join("")}</div></div>
    ${missingAccount()}${fillIn()}
    ${S.error ? `<div class="banner err">${esc(S.error)}${S.graph ? " Showing the last version that loaded." : ""}</div>` : ""}
    <div id="jobbar"></div>
    <div class="body"><div class="main"><div class="stage">${views[T.view]()}</div>
      <div class="ask">${suggestions().map(([t, a, cls]) => `<button class="btn ${cls || ""}" data-act="${a}">${t}</button>`).join("")}<button class="btn" data-act="agent" style="margin-left:auto">${I.agent} Work with your coding agent</button></div></div>
      ${dr ? `<aside class="drawer"><div class="dh"><b>${dr[0]}</b><span class="small">${dr[1]}</span><button class="x" data-act="closedrawer" title="Close (Esc)">×</button></div><div class="db">${dr[2]}</div></aside>` : ""}</div></div>`;
  const st = $(".stage"); if (st) st.scrollTop = scroll.stage; const db = $(".db"); if (db) db.scrollTop = scroll.db;
  fitMap(); renderJobBar(); go(false);
}
function fitMap() {
  const m = $("#mapwrap"), st = $(".stage"); if (!m || !st) return;
  const k = Math.min(1, (st.clientWidth - 40) / S.graph.width);
  m.style.transform = `scale(${k})`; m.style.marginBottom = `${(k - 1) * S.graph.height}px`; m.style.marginRight = `${(k - 1) * S.graph.width}px`;
}
window.addEventListener("resize", fitMap);
function stale(r) {
  if (!r.ran || !r.current) return true;
  return Object.entries(r.steps || {}).some(([s, c]) => c.model && S.steps[s] && S.steps[s].lm && c.model.split(":").pop() !== S.steps[s].lm.split(":").pop());
}
function suggestions() {
  const out = [], rs = rows(), sc = S.scores.end, row = curRow();
  if (!rs.length) return [["Add examples", "toexamples", "primary"]];
  if (!readyConns().length) out.push(["Connect an AI", "accounts", "primary"]);
  const notRun = rs.filter(r => r.source === "example" && stale(r));
  const E = S.estimates || {};
  const runCost = notRun.length ? (E.not_run || (E.per_example || 0) * notRun.length) : 0;
  if (rs.length && notRun.length) out.push([(notRun.length === rs.length ? `Run the ${rs.length} examples` : `Run the ${notRun.length} not run with the code as it is now`) + costTag(runCost), "runall", "primary"]);
  if (sc.unchecked) out.push([sc.checked < 15 ? "Check 10 answers" : "Check more answers", "check", out.length ? "" : "primary"]);
  if (row && row.ran && row.state === "wrong") out.push([`Why is #${row.n} wrong?` + costTag((E.why || {})[String(row.n)]), "why"]);
  if (sc.wrong) out.push(["Find the step to fix first" + costTag(E.blame), "blame"]);
  if (rs.some(r => r.ran)) out.push([`${I.down} Summary to share`, "download"]);
  return out.slice(0, 4);
}

// ------------------------------------------------------------ actions
async function rate(call, verdict, answer, sample) {
  const body = { call, verdict }; if (answer !== undefined && answer !== null) body.answer = answer; if (sample) body.sample = sample;
  await api("/api/rate", body); ETAG = "";
}
function startCheck() {
  const pool = rows().filter(r => r.ran && r.current && r.state === "not checked");
  if (!pool.length) { toast("Every answer of this version is checked. Run more examples to check more."); return; }
  for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  T.check = { queue: pool.slice(0, 10), i: 0, done: [], sample: Math.random().toString(16).slice(2, 12) };
  T.drawer = "check"; render();
}
async function checkAnswer(v) {
  const c = T.check; if (!c || c.i >= c.queue.length) return;
  const r = c.queue[c.i];
  if (v !== "skip") await rate(r.call, v, null, c.sample);
  c.done.push(v); c.i++; render();
}
async function copy(text, msg) {
  try { await navigator.clipboard.writeText(text); toast(msg || "Copied."); }
  catch (e) { prompt("Copy this:", text); }
}

document.addEventListener("toggle", e => { const k = e.target.dataset && e.target.dataset.key; if (k) T.open[k] = e.target.open; }, true);
document.addEventListener("click", async e => {
  const f = e.target.closest("[data-focus]"); if (f && T.view === "review" && !e.target.closest("button,a,input,textarea,select,summary")) { T.focus = +f.dataset.focus; render(); return; }
  const t = e.target.closest("[data-act],[data-v],[data-node],[data-cell],[data-fill]"); if (!t) return;
  if (t.tagName === "A" && !t.hasAttribute("download")) e.preventDefault();
  if (t.dataset.fill) { T.drafts.what = t.dataset.fill; render(); return; }
  if (t.dataset.v) { T.view = t.dataset.v; T.storyAt = 0; go(true); render(); return; }
  if (t.dataset.node) { T.sel = t.dataset.node; T.drawer = "node"; T.wrongFor = null; T.request = null; render(); return; }
  if (t.dataset.cell) { const [n, s, i] = t.dataset.cell.split(":"); T.n = n; T.sel = s === "in" ? (i ? "in:" + i : null) : s; T.drawer = T.sel ? "node" : null; T.wrongFor = null; render(); go(false); return; }
  const a = t.dataset.act, call = t.dataset.call;
  switch (a) {
    case "prev": return moveRow(-1);
    case "next": return moveRow(1);
    case "snext": T.storyAt++; return render();
    case "sback": T.storyAt--; return render();
    case "closedrawer": T.drawer = null; T.sel = null; return render();
    case "costs": case "activity": case "agent": T.drawer = a; return render();
    case "copylink": return copy(location.href, "Link copied: it opens this view on this computer.");
    case "copyprompt": return copy(agentPrompt(draft("what", T.what || "")), "Copied: paste it to your coding agent.");
    case "copyask": return copy(drawerAgent()[3][+t.dataset.i], "Copied: paste it to your coding agent.");
    case "example": await api("/api/example"); ETAG = ""; toast("Made the club example. Run it to see it work."); return;
    case "right": await rate(call, "right"); T.wrongFor = null; return;
    case "wrong": T.wrongFor = call; await rate(call, "wrong"); return render();
    case "withdraw": await rate(call, null); return;
    case "answer": { const v = JSON.parse(t.dataset.answer); await rate(call, "wrong", v); T.wrongFor = null; return; }
    case "answertext": { const v = ($("#answerbox") || {}).value; await rate(call, "wrong", v ? v : null); T.wrongFor = null; return; }
    case "incode": T.view = "code"; go(true); render(); return;
    case "request": { const r = await fetch("/api/call?id=" + encodeURIComponent(call)); T.request = await r.json(); return render(); }
    case "runall": { const ns = rows().filter(r => r.source === "example" && stale(r)).map(r => r.n); return startJob("/api/run", { only: ns }, "Running"); }
    case "runone": return startJob("/api/run", { only: [curRow().n] }, "Running");
    case "check": return startCheck();
    case "cright": return checkAnswer("right");
    case "cwrong": return checkAnswer("wrong");
    case "cskip": return checkAnswer("skip");
    case "why": return startJob("/api/why", { n: curRow().n }, "Finding the step at fault");
    case "blame": return startJob("/api/blame", {}, "Finding the step to fix first");
    case "download": location.href = "/api/page"; return;
    case "trial": T.compareBody = { lm: { [T.trial.step]: T.trial.model }, all: true }; return startJob("/api/compare", T.compareBody, `Trying ${T.trial.model}`);
    case "recompare": return startJob("/api/compare", T.compareBody || { all: true }, "Comparing again");
    case "versions": T.drawer = "versions"; T.version = null; return render();
    case "models": T.drawer = "models"; return render();
    case "accounts": T.drawer = "accounts"; return render();
    case "homeacc": T.homeAcc = !T.homeAcc; return render();
    case "signin": { await api("/api/accounts/login", { provider: t.dataset.provider }); T.loginSeen = null; if (S.project) T.drawer = "accounts"; else T.homeAcc = true; ETAG = ""; return; }
    case "signout": { if (!confirm("Sign out? Studio and your scripts will no longer use this account (the account itself is untouched).")) return; const r = await api("/api/accounts/signout", { provider: t.dataset.provider }); toast(esc(r.say)); ETAG = ""; return; }
    case "savekey": return saveKey();
    case "loginanswer": return loginAnswer(t.dataset.value);
    case "logincancel": await api("/api/accounts/login/cancel", { id: t.dataset.id }); ETAG = ""; return;
    case "loginok": T.loginSeen = t.dataset.id; return render();
    case "copytext": return copy(t.dataset.text, "Copied.");
    case "openproj": { await api("/api/projects/open", { path: t.dataset.path }); resetView(); ETAG = ""; return; }
    case "closeproj": { await api("/api/projects/close"); resetView(); ETAG = ""; return; }
    case "fillin": T.view = "map"; T.sel = "const:" + t.dataset.name; T.drawer = "node"; go(true); return render();
    case "draftplan": { const what = draft("what", ""); if (!what.trim()) return; T.what = what; return startJob("/api/draft", { request: what }, "Planning it"); }
    case "blankplan": T.plan = blankPlan(draft("what", "").trim()); T.planFromAI = false; return render();
    case "dropplan": T.plan = null; return render();
    case "buildplan": { const r = await api("/api/new", { plan: T.plan }); T.plan = null; delete T.drafts.what; resetView(); toast(`Made ${esc(r.made)}. Run the examples to see it work.`); ETAG = ""; return; }
    case "planadd": { const l = t.dataset.list, P = T.plan;
      if (l === "inputs") P.inputs.push({ name: "", about: "" });
      if (l === "facts") P.facts.push({ name: "", about: "" });
      if (l === "steps") { const prev = P.steps[P.steps.length - 1]; P.steps.push({ name: "", instruction: "", takes: prev && prev.name ? [prev.name] : [], answer: "text", choices: [] }); }
      if (l === "examples") P.examples.push({});
      return render(); }
    case "plandel": T.plan[t.dataset.list].splice(+t.dataset.i, 1); return render();
    case "planmove": { const st = T.plan.steps, i = +t.dataset.i, j = i + +t.dataset.d; [st[i], st[j]] = [st[j], st[i]]; return render(); }
    case "toexamples": T.view = "table"; T.exForm = "add"; go(true); return render();
    case "exform": T.exForm = t.dataset.f || null; return render();
    case "exadd": { const values = {}; for (const n of S.program.inputs) values[n] = draft("exnew:" + n, ""); const r = await api("/api/examples", { op: "add", values }); for (const n of S.program.inputs) delete T.drafts["exnew:" + n]; toast(esc(r.what)); ETAG = ""; return; }
    case "expaste": { const r = await api("/api/examples", { op: "paste", text: draft("expaste", "") }); delete T.drafts.expaste; T.exForm = null; toast(esc(r.what) + (r.skipped ? ` (${r.skipped} already there)` : "")); ETAG = ""; return; }
    case "exwrite": return startJob("/api/examples/write", { count: 10 }, "Writing examples");
    case "exsave": { const n = t.dataset.n, values = {}; const row = rows().find(r => String(r.n) === n); for (const k of S.program.inputs) values[k] = draft(`ex:${n}:${k}`, txt(row.inputs[k])); await api("/api/examples", { op: "update", n: +n, values }); for (const k of S.program.inputs) delete T.drafts[`ex:${n}:${k}`]; toast(`Changed example ${esc(n)}.`); ETAG = ""; return; }
    case "exdelete": { if (!confirm(`Remove example ${t.dataset.n}? (Undo brings it back.)`)) return; await api("/api/examples", { op: "delete", n: +t.dataset.n }); T.drawer = null; T.sel = null; toast(`Removed example ${esc(t.dataset.n)}.`); ETAG = ""; return; }
    case "exkeep": { const row = curRow(); await api("/api/examples", { op: "add", values: row.inputs }); toast("Kept as an example."); ETAG = ""; return; }
    case "undo": { const r = await api("/api/undo"); toast(`Undone: ${esc(r.undone)}`); ETAG = ""; return; }
    case "toreview": if (t.dataset.n) T.n = t.dataset.n; T.view = "review"; T.focus = 0; T.detail = null; go(true); render(); return;
    case "rvprev": return moveReview(-1);
    case "rvnext": return moveReview(1);
    case "rvright": return rvJudge(+t.dataset.i, "right");
    case "rvwrong": return rvJudge(+t.dataset.i, "wrong");
    case "rvshould": return rvJudge(+t.dataset.i, "wrong", { answer: JSON.parse(t.dataset.v) });
    case "toggleopen": T.open[t.dataset.key] = !T.open[t.dataset.key]; return render();
    case "dropdraft": delete T.drafts[t.dataset.key]; if (t.dataset.key2) delete T.drafts[t.dataset.key2]; return render();
    case "trialopen": T.trialOpen = T.trialOpen === T.sel ? null : T.sel; return render();
    case "saveinstr": { const st = t.dataset.step; await edit({ op: "instruction", step: st, text: T.drafts["instr:" + st] }); delete T.drafts["instr:" + st]; return; }
    case "saveanswer": { const st = t.dataset.step, kind = draft("kind:" + st, answerKind(S.steps[st]));
      await edit({ op: "answer", step: st, kind, choices: kind === "choices" ? draft("choices:" + st, "").split(",").map(x => x.trim()).filter(Boolean) : null });
      delete T.drafts["kind:" + st]; delete T.drafts["choices:" + st]; return; }
    case "savevalue": { const n = S.graph.nodes.find(n => n.id === t.dataset.id); await edit({ op: "value", line: n.line, col: n.col, value: T.drafts["val:" + n.id] }); delete T.drafts["val:" + n.id]; return; }
    case "savefile": { const nm = t.dataset.name; await edit({ op: "file", name: nm, text: T.drafts["file:" + nm] }); delete T.drafts["file:" + nm]; return; }
    case "addopen": T.addAfter = T.addAfter === t.dataset.after ? null : t.dataset.after; return render();
    case "addclose": T.addAfter = null; return render();
    case "addstep": { const af = t.dataset.after, k = "add:" + af, kind = draft(k + ":kind", "same");
      const extra = [...document.querySelectorAll("[data-addextra]")].filter(x => x.checked).map(x => x.dataset.addextra);
      await edit({ op: "add", after: af, name: draft(k + ":name").trim(), instruction: draft(k + ":instr"), answer: kind, extra,
        choices: kind === "choices" ? draft(k + ":choices").split(",").map(x => x.trim()).filter(Boolean) : null });
      for (const x of ["name", "instr", "kind", "choices"]) delete T.drafts[k + ":" + x]; T.addAfter = null; return; }
    case "removestep": { if (!confirm(`Take ${t.dataset.step} out of the pipeline? (Undo brings it back.)`)) return; await edit({ op: "remove", step: t.dataset.step }); T.drawer = null; T.sel = null; return; }
    case "editcode": T.codeEdit = true; return render();
    case "cancelcode": T.codeEdit = false; delete T.drafts.code; return render();
    case "savecode": { await edit({ op: "source", text: T.drafts.code }); T.codeEdit = false; delete T.drafts.code; return; }
    case "seeversion": { const r = await fetch("/api/version?hash=" + encodeURIComponent(t.dataset.hash)); const j = await r.json(); T.version = { hash: t.dataset.hash, ...j }; return render(); }
    case "restore": {
      const v = S.program.versions.find(v => v.hash === t.dataset.hash);
      if (!confirm(`Put version ${v.label}'s code back in ${S.program.file}?\n\nThe code as it is now is kept too: you can come back to it from this list.`)) return;
      await api("/api/restore", { hash: t.dataset.hash }); ETAG = ""; T.version = null;
      toast(`${esc(S.program.file)} holds version ${v.label}'s code again. Your coding agent and git see it as an ordinary change.`); return;
    }
    case "cmpright": case "cmpwrong": {
      const row = rows().find(x => String(x.n) === String(t.dataset.n));
      await api("/api/whatif-judge", { inputs: row.inputs, end: JSON.parse(t.dataset.end), verdict: a === "cmpright" ? "right" : "wrong" });
      t.closest(".acts").innerHTML = `<span class="small">${a === "cmpright" ? "judged right" : "judged wrong"}</span>`; return;
    }
    case "usemodel": { const r = await api("/api/use-model", { step: T.trial.step, model: T.trial.model }); toast(`${esc(S.program.file)}, line ${r.line}: <code>${esc(r.after)}</code>`); T.trial = null; ETAG = ""; return; }
    case "wifright": case "wifwrong": {
      const row = rows().find(x => String(x.n) === String(t.dataset.n));
      await api("/api/whatif-judge", { inputs: row.inputs, end: JSON.parse(t.dataset.end), verdict: a === "wifright" ? "right" : "wrong" });
      toast("Noted. Run “Find the step to fix first” again to count it."); return;
    }
  }
});
document.addEventListener("change", async e => {
  const ac = e.target.dataset && e.target.dataset.actChange;
  if (e.target.id === "modelpick" || ac === "model" || ac === "allmodels") {
    let m = e.target.value; if (!m) return;
    if (m === "__other") { m = prompt("Model name (as FunctAI writes it, e.g. claude-sonnet-5-5 or ollama:llama3.2):"); if (!m) return render(); }
    if (ac === "allmodels") return edit({ op: "models", model: m });
    return edit({ op: "model", step: e.target.dataset.step, model: m });
  }
  if (e.target.dataset && e.target.dataset.planRender && T.plan) { setPath(T.plan, e.target.dataset.plan, e.target.value); return render(); }
  if (e.target.dataset && e.target.dataset.plantake !== undefined && T.plan) { const st = T.plan.steps[+e.target.dataset.plantake]; st.takes = st.takes || []; if (e.target.checked) { if (!st.takes.includes(e.target.value)) st.takes.push(e.target.value); } else st.takes = st.takes.filter(x => x !== e.target.value); return; }
  if (e.target.id === "exfile" && e.target.files && e.target.files[0]) { const text = await e.target.files[0].text(); const r = await api("/api/examples", { op: "paste", text }); toast(esc(r.what) + (r.skipped ? ` (${r.skipped} already there)` : "")); ETAG = ""; return; }
  if (e.target.dataset && e.target.dataset.draft === "keyprov") { T.drafts.keyprov = e.target.value; return render(); }
  if (ac === "wire") return edit({ op: "input", step: e.target.dataset.step, param: e.target.dataset.param, source: e.target.value, line: +e.target.dataset.line });
  if (e.target.id === "reviewfilter") { T.reviewFilter = e.target.value; const rs = reviewRows(); if (rs.length && !rs.some(r => String(r.n) === String(T.n))) T.n = String(rs[0].n); T.detail = null; return render(); }
  if (e.target.id === "advance") { T.advance = e.target.checked; localStorage.setItem("studio-advance", T.advance ? "1" : "0"); return; }
  if (e.target.dataset && e.target.dataset.draft && /^(kind:|add:.*:kind)/.test(e.target.dataset.draft)) { T.drafts[e.target.dataset.draft] = e.target.value; return render(); }
  if (e.target.id === "lmpick") {
    let m = e.target.value;
    if (m === "__other") { m = prompt("Model name (as FunctAI writes it, e.g. claude-sonnet-5-5 or ollama:llama3.2):"); if (!m) return render(); }
    T.trial = { step: e.target.dataset.step, model: m }; T.trialEst = null; render();
    if (m !== (S.steps[T.trial.step] || {}).lm) {
      const est = await api("/api/estimate", { action: "compare", lm: { [T.trial.step]: m }, all: true });
      if (T.trial && T.trial.model === m) { T.trialEst = est; render(); }
    }
    return;
  }
  if (e.target.id === "monthly") { await api("/api/settings", { monthly: Math.max(0, +e.target.value || 0) }); ETAG = ""; }
  if (e.target.id === "cap") { await api("/api/settings", { cap_dollars: Math.max(0, +e.target.value || 0) }); toast("Runs above your limit will wait for your yes."); ETAG = ""; }
});
document.addEventListener("input", e => {
  if (e.target.id === "what") { T.what = e.target.value; const p = $("#prompt"); if (p) p.textContent = agentPrompt(T.what); }
  if (e.target.id === "monthly") { const pm = S.money.per_message_cents, el = $("#monthcost"); if (el && pm !== null) el.textContent = cents(pm * (+e.target.value || 0)); }
});
function scrollFocus() { const el = document.querySelector(".rv.focus"); if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && e.target.dataset && e.target.dataset.enter && !e.shiftKey) {
    e.preventDefault(); const i = +e.target.dataset.i, v = e.target.value;
    if (e.target.dataset.enter === "savekey") return saveKey();
    if (e.target.dataset.enter === "loginanswer") return loginAnswer();
    if (e.target.dataset.enter === "rvsavenote") { delete T.drafts[e.target.dataset.draft]; e.target.blur(); return rvJudge(i, undefined, { note: v }); }
    if (e.target.dataset.enter === "rvsaveshould") { delete T.drafts[e.target.dataset.draft]; e.target.blur(); let a = v; try { a = JSON.parse(v); } catch (_) { } return rvJudge(i, "wrong", { answer: a }); }
  }
  if (!S || !S.program) return;
  if (e.key === "Escape") { T.drawer = null; T.sel = null; return render(); }
  if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  if (T.drawer === "check" && T.check && T.check.i < T.check.queue.length) {
    if (e.key === "1") return checkAnswer("right"); if (e.key === "2") return checkAnswer("wrong"); if (e.key === "s") return checkAnswer("skip");
  }
  if (T.view === "review" && T.detail && !T.detail.error) {
    const last = T.detail.steps.length;
    if (e.key === "1") { e.preventDefault(); return rvJudge(T.focus, "right"); }
    if (e.key === "2") { e.preventDefault(); return rvJudge(T.focus, "wrong"); }
    if (e.key === "ArrowDown") { e.preventDefault(); T.focus = Math.min(last, T.focus + 1); render(); scrollFocus(); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); T.focus = Math.max(0, T.focus - 1); render(); scrollFocus(); return; }
    if (e.key === "n") { const el = $("#note-" + T.focus); if (el) { e.preventDefault(); el.focus(); } return; }
    if (e.key === "j" || e.key === "ArrowRight") return moveReview(1);
    if (e.key === "k" || e.key === "ArrowLeft") return moveReview(-1);
    return;
  }
  const v = { "1": "map", "2": "review", "3": "table", "4": "summary", "5": "code" }[e.key];
  if (v) { T.view = v; go(true); render(); return; }
  if (e.key === "j") moveRow(1);
  if (e.key === "k") moveRow(-1);
});

readUrl();
poll();

// ============================================================ drafts: what a person types survives the live refresh
T.drafts = {};
const draft = (key, fallback) => (key in T.drafts ? T.drafts[key] : (fallback ?? ""));
document.addEventListener("input", e => {
  const d = e.target.dataset || {};
  if (d.plan && T.plan && !d.planRender) { setPath(T.plan, d.plan, e.target.value); return; }
  if (d.planchoices !== undefined && T.plan) { T.plan.steps[+d.planchoices].choices = e.target.value.split(",").map(x => x.trim()).filter(Boolean); return; }
  const k = d.draft; if (!k) return;
  const had = k in T.drafts; T.drafts[k] = e.target.value;
  if (!had && /^(instr:|choices:|file:|val:|code$|what$|ex:)/.test(k)) render();
});

// ============================================================ editing: every change goes into the code
async function edit(body, okText) {
  const r = await api("/api/edit", { base: S.code.sha, ...body });
  const a = document.activeElement; if (a && /INPUT|TEXTAREA|SELECT/.test(a.tagName)) a.blur();   // saved: refresh freely
  if (r.unchanged) { toast("Nothing changed."); return r; }
  toast(esc(okText || r.what) + ` · <a data-act="undo" style="color:#c9cdff">undo</a>`);
  ETAG = ""; return r;
}
function lastEdit() { const e = (S.edits || []); return e.length ? e[e.length - 1] : null; }
const ANSWER_KINDS = [["text", "text"], ["choices", "one of a few choices"], ["number", "a number"], ["whole number", "a whole number"], ["yes/no", "yes or no"], ["list of texts", "a list of texts"]];
function answerKind(info) {
  if (info.choices && info.choices.length) return "choices";
  const g = String(info.gives || "");
  return g === "int" || g === "integer" ? "whole number" : g === "float" || g === "number" ? "number" : g === "bool" || g === "boolean" ? "yes/no" : /^list|^array/.test(g) ? "list of texts" : "text";
}
function editBlocks(id) {
  const info = S.steps[id] || {}, wires = ((S.wiring || {}).steps || {})[id] || [], labels = (S.wiring || {}).labels || {};
  const kind = draft(`kind:${id}`, answerKind(info)), instr = draft(`instr:${id}`, info.instruction || "");
  let h = `<div class="sec"><div class="lbl">Its AI</div>${modelSelect("modelpick", info.lm, `data-step="${esc(id)}"`)}
    <div class="small" style="margin-top:4px">${info.own_lm ? "Changing it writes the new model into the code." : `The code names no model, so it uses the project's default (${esc(S.settings.lm)}); choosing one writes it into the code.`} <a data-act="trialopen">Try another first, without changing the code</a></div>
    ${T.trialOpen === id ? `<div style="margin-top:6px">${modelSelect("lmpick", T.trial && T.trial.step === id ? T.trial.model : info.lm, `data-step="${esc(id)}"`)}</div>` : ""}`;
  if (T.trialOpen === id && T.trial && T.trial.step === id && T.trial.model !== info.lm)
    h += `<div class="acts"><button class="btn primary" data-act="trial">Try ${esc(T.trial.model)} on the judged examples${T.trialEst ? costTag(T.trialEst.cents) : ""}</button></div>`;
  h += `</div><div class="sec"><div class="lbl">It is asked</div><textarea class="in" rows="${Math.min(10, Math.max(3, instr.split("\n").length + 1))}" data-draft="instr:${esc(id)}" id="instr">${esc(instr)}</textarea>
    ${instr !== (info.instruction || "") ? `<div class="acts"><button class="btn primary" data-act="saveinstr" data-step="${esc(id)}">Save the instruction</button><button class="btn small" data-act="dropdraft" data-key="instr:${esc(id)}">Cancel</button></div>` : ""}
    <div class="small" style="margin-top:4px">These words are the step's prompt (with its name, its inputs' names and its kind of answer).</div></div>`;
  const choices = draft(`choices:${id}`, (info.choices || []).join(", "));
  h += `<div class="sec"><div class="lbl">It answers with</div><select class="in" data-draft="kind:${esc(id)}" id="answerkind">${ANSWER_KINDS.map(([k, l]) => `<option value="${k}" ${k === kind ? "selected" : ""}>${l}</option>`).join("")}</select>
    ${kind === "choices" ? `<input class="in" style="margin-top:6px" data-draft="choices:${esc(id)}" id="choices" value="${esc(choices)}" placeholder="question, idea, spam">` : ""}
    ${kind !== answerKind(info) || (kind === "choices" && choices !== (info.choices || []).join(", ")) ? `<div class="acts"><button class="btn primary" data-act="saveanswer" data-step="${esc(id)}">Save</button><button class="btn small" data-act="dropdraft" data-key="kind:${esc(id)}" data-key2="choices:${esc(id)}">Cancel</button></div>` : ""}</div>`;
  if (wires.length) {
    const w = wires[0];
    h += `<div class="sec"><div class="lbl">It is given</div>${w.inputs.map(inp => `<div class="row" style="margin:4px 0"><span style="min-width:80px"><b>${esc(inp.param)}</b></span>${inp.simple ? `<select class="in" style="flex:1" data-act-change="wire" data-step="${esc(id)}" data-param="${esc(inp.param)}" data-line="${w.line}">${w.sources.map(s => `<option value="${esc(s)}" ${s === inp.given ? "selected" : ""}>${esc(s)} · ${esc(labels[s] || "")}</option>`).join("")}</select>` : `<code style="flex:1">${esc(inp.given)}</code>`}</div>`).join("")}
      ${wires.length > 1 ? `<div class="small">${esc(id)} is called in ${wires.length} places; this changes the first (line ${w.line}).</div>` : ""}</div>`;
  }
  h += `<div class="sec"><div class="lbl">Change the pipeline</div><div class="acts"><button class="btn small" data-act="addopen" data-after="${esc(id)}">Add a step after ${esc(id)}</button><button class="btn small" data-act="removestep" data-step="${esc(id)}">Take ${esc(id)} out</button></div>${T.addAfter === id ? addForm(id) : ""}</div>`;
  return h;
}
function addForm(after) {
  const k = `add:${after}`, kind = draft(k + ":kind", "same"), ins = S.program.inputs;
  return `<div class="frame" style="margin-top:8px"><div class="small">The new step gets ${after === "(end)" ? "the pipeline's result" : `what ${esc(after)} gave`}${ins.length ? " (and, if you tick them, the pipeline's inputs)" : ""}; its answer goes on in place of ${after === "(end)" ? "the result" : esc(after) + "'s"}.</div>
    <input class="in" style="margin-top:6px" placeholder="its name, e.g. shorten" data-draft="${k}:name" id="addname" value="${esc(draft(k + ":name"))}">
    <textarea class="in" style="margin-top:6px" rows="3" placeholder="what it is asked, e.g. Make it one friendly sentence." data-draft="${k}:instr" id="addinstr">${esc(draft(k + ":instr"))}</textarea>
    <select class="in" style="margin-top:6px" data-draft="${k}:kind" id="addkind"><option value="same" ${kind === "same" ? "selected" : ""}>it answers with the same kind of value</option>${ANSWER_KINDS.map(([v, l]) => `<option value="${v}" ${v === kind ? "selected" : ""}>it answers with ${l}</option>`).join("")}</select>
    ${kind === "choices" ? `<input class="in" style="margin-top:6px" placeholder="choices, separated by commas" data-draft="${k}:choices" value="${esc(draft(k + ":choices"))}">` : ""}
    ${ins.length ? `<div class="small" style="margin-top:6px">Also give it: ${ins.map(i => `<label><input type="checkbox" data-addextra="${esc(i)}"> ${esc(i)}</label>`).join(" ")}</div>` : ""}
    <div class="acts"><button class="btn primary" data-act="addstep" data-after="${esc(after)}">Add the step</button><button class="btn small" data-act="addclose">Cancel</button></div></div>`;
}
function drawerModels() {
  const steps = Object.keys(S.steps);
  return ["Models", "", `<p class="small">Each step asks its own AI. Choosing one writes it into the code (undo is one click). Prices are per million tokens in / out.</p>
    ${steps.map(s => `<div class="row" style="margin:8px 0"><b style="min-width:90px">${esc(s)}</b><div style="flex:1">${modelSelect("m-" + s, S.steps[s].lm, `data-act-change="model" data-step="${esc(s)}"`)}</div></div>`).join("")}
    <div class="sec"><div class="lbl">One model for every step</div>${modelSelect("allmodels", "", `data-act-change="allmodels"`, "choose one…")}</div>
    ${accountsBlock()}`];
}

// ============================================================ Review: what each model was given and answered, judged fast
T.detail = null; T.focus = 0; T.reviewFilter = "all"; T.open = {};
T.advance = localStorage.getItem("studio-advance") !== "0";
function reviewRows() {
  const f = T.reviewFilter;
  return rows().filter(r => f === "all" ? true : f === "unchecked" ? r.ran && r.state === "not checked" : r.state === f);
}
async function loadDetail(force) {
  const n = T.n, key = n + "@" + ETAG;
  if (!force && T.detail && T.detailKey === key) return;
  T.detailKey = key;
  const r = await fetch("/api/example?n=" + encodeURIComponent(n));
  const j = await r.json();
  if (T.detailKey !== key) return;
  T.detail = r.ok ? j : { error: j.error };
  if (T.view === "review") render();
}
function fieldBox(key, label, v) {
  const t = txt(v), lines = t.split("\n"), long = lines.length > 8 || t.length > 700, open = T.open[key];
  const shown = long && !open ? lines.slice(0, 6).join("\n").slice(0, 600) + "…" : t;
  return `<div class="fld"><div class="fl">${esc(label)}${long ? ` <span class="small">· ${lines.length} lines</span>` : ""}</div><div class="fv">${esc(shown) || '<span class="mute">(nothing)</span>'}</div>${long ? `<a class="small" data-act="toggleopen" data-key="${esc(key)}">${open ? "show less" : "show all"}</a>` : ""}</div>`;
}
function judgeBar(c, idx, choices) {
  if (!c || !c.call || c.error) return "";
  const j = c.judgement, mine = j && j.source === "person";
  const note = draft(`note:${c.call}`, (mine && j.note) || ""), should = draft(`should:${c.call}`, mine && j.answer != null ? txt(j.answer) : "");
  let h = `<div class="jb"><button class="btn ${j && j.verdict === "right" ? "on-good" : ""}" data-act="rvright" data-i="${idx}">${I.ok} Right <kbd>1</kbd></button><button class="btn ${j && j.verdict === "wrong" ? "on-bad" : ""}" data-act="rvwrong" data-i="${idx}">${I.no} Wrong <kbd>2</kbd></button>
    <span class="small">${judgedBy(j)}</span></div>`;
  if (j && j.verdict === "wrong" && mine) {
    h += `<div class="row" style="margin-top:6px"><span class="small" style="min-width:90px">It should be</span>${choices && choices.length
      ? choices.map(ch => `<button class="btn small ${txt(j.answer) === ch ? "on-good" : ""}" data-act="rvshould" data-i="${idx}" data-v="${esc(JSON.stringify(ch))}">${esc(ch)}</button>`).join("")
      : `<input class="in" style="flex:1" data-draft="should:${c.call}" data-i="${idx}" data-enter="rvsaveshould" value="${esc(should)}" placeholder="the right answer (Enter saves)">`}</div>`;
  }
  if (mine) h += `<div class="row" style="margin-top:6px"><span class="small" style="min-width:90px">Note</span><input class="in" style="flex:1" data-draft="note:${c.call}" data-i="${idx}" data-enter="rvsavenote" id="note-${idx}" value="${esc(note)}" placeholder="why (Enter saves) · n"></div>`;
  return h;
}
function stepCard(c, idx, title, sub) {
  const sent = c.sent, rep = c.reply || {}, key = c.call;
  const outs = c.outputs && typeof c.outputs === "object" ? Object.entries(c.outputs) : [];
  const answer = c.error ? `<div class="fv fail-t">couldn't run: ${esc(c.error)}</div>` : outs.length > 1
    ? outs.map(([k, v]) => fieldBox(key + ":o:" + k, k, v)).join("") : `<div class="fv big">${esc(txt(c.value)) || '<span class="mute">(nothing)</span>'}</div>`;
  const meta = [c.model, c.seconds != null ? c.seconds.toFixed(1) + " s" : null, c.cached ? "from the cache" : c.cents ? cents(c.cents) : null, c.attempts > 1 ? `${c.attempts} tries` : null].filter(Boolean).join(" · ");
  return `<div class="rv ${T.focus === idx ? "focus" : ""} ${(c.judgement || {}).verdict || ""}" data-focus="${idx}">
    <div class="rvh"><b>${esc(title)}</b><span class="small">${esc(sub || "")}</span><span class="small" style="margin-left:auto">${esc(meta)}</span></div>
    <div class="rvb"><div><div class="lbl">It was given</div>${Object.entries(c.inputs || {}).map(([k, v]) => fieldBox(key + ":i:" + k, k, v)).join("") || '<div class="mute small">nothing</div>'}</div>
      <div><div class="lbl">It answered</div>${answer}${rep.reasoning ? `<details><summary>Its reasoning</summary><div class="fv">${esc(rep.reasoning)}</div></details>` : ""}</div></div>
    ${sent ? `<details ${T.open[key + ":sent"] ? "open" : ""} data-key="${esc(key)}:sent"><summary>Exactly what was sent, and the reply</summary><div class="sent">${sent.system ? `<div class="role">instructions</div><div class="fv">${esc(sent.system)}</div>` : ""}${sent.messages.map(m => `<div class="role">${esc(m.role)}</div><div class="fv">${esc(m.text)}</div>`).join("")}<div class="role">reply</div><div class="fv">${esc(rep.text)}</div></div></details>` : ""}
    ${judgeBar(c, idx, c.choices)}</div>`;
}
function viewReview() {
  const rs = reviewRows(), row = curRow();
  const filter = `<select class="in" style="width:auto" id="reviewfilter">${[["all", "every example"], ["unchecked", "not checked yet"], ["wrong", "wrong"], ["failed", "failed"], ["right", "right"]].map(([v, l]) => `<option value="${v}" ${T.reviewFilter === v ? "selected" : ""}>${l} (${v === "all" ? rows().length : rows().filter(r => v === "unchecked" ? r.ran && r.state === "not checked" : r.state === v).length})</option>`).join("")}</select>`;
  let h = `<div class="picker"><button data-act="rvprev" title="Previous (k)">‹</button><span>${row ? `#${esc(row.n)}` : ""} · ${Math.max(0, rs.indexOf(row)) + 1} of ${rs.length}</span><button data-act="rvnext" title="Next (j)">›</button>${filter}
    <label class="small"><input type="checkbox" id="advance" ${T.advance ? "checked" : ""}> after judging the end, go to the next</label><span class="small" style="margin-left:auto">keys: j/k examples · ↑/↓ steps · 1 right · 2 wrong · n note</span></div>`;
  if (!row) return h + `<div class="summary">No examples yet.</div>`;
  if (!row.ran) return h + `<div class="card">#${esc(row.n)} has not run with this code yet. <button class="btn primary" data-act="runone">Run it${costTag((S.estimates || {}).per_example)}</button></div>`;
  loadDetail();                                   // again when the records changed (it keeps showing the last copy)
  const d = T.detail;
  if (!d || String(d.n) !== String(row.n)) return h + `<div class="summary">Loading…</div>`;
  if (d.error) return h + `<div class="banner err">${esc(d.error)}</div>`;
  h += `<div class="rvin"><div class="lbl">The pipeline was given</div>${Object.entries(d.inputs).map(([k, v]) => fieldBox("in:" + k, k, v)).join("")}${!d.current ? `<div class="small" style="color:var(--warn)">Answered by version ${d.version}, not the code as it is now.</div>` : ""}</div>`;
  d.steps.forEach((c, i) => h += stepCard(c, i, c.name, (c.instruction || "").split(/(?<=[.?!])\s/)[0]));
  const ei = d.steps.length;
  h += stepCard({ ...d.end, inputs: {}, sent: null, reply: null }, ei, "End result", "what the pipeline gave back").replace('<div class="rvb"><div><div class="lbl">It was given</div><div class="mute small">nothing</div></div>', '<div class="rvb one">');
  return h;
}
function reviewTarget(i) { const d = T.detail; if (!d) return null; return i < d.steps.length ? d.steps[i] : d.end; }
async function rvJudge(i, verdict, extra) {
  const c = reviewTarget(i); if (!c || !c.call) return;
  const j = c.judgement && c.judgement.source === "person" ? c.judgement : {};
  const body = { call: c.call, verdict: verdict === undefined ? j.verdict : verdict };
  const note = extra && "note" in extra ? extra.note : j.note, answer = extra && "answer" in extra ? extra.answer : (verdict === "right" ? null : j.answer);
  if (note) body.note = note; if (answer !== null && answer !== undefined && answer !== "") body.answer = answer;
  await api("/api/rate", body); ETAG = ""; T.focus = i;
  const isEnd = i === T.detail.steps.length;
  if (verdict === "right" && isEnd && T.advance) { setTimeout(() => moveReview(1), 250); return; }
  if (verdict === "right" && !isEnd) T.focus = i + 1;
  loadDetail(true);
}
function moveReview(d) {
  const rs = reviewRows(); if (!rs.length) return;
  const i = rs.findIndex(r => String(r.n) === String(T.n));
  const r = rs[(i + d + rs.length) % rs.length]; T.n = String(r.n); T.focus = 0; T.detail = null; render(); go(false);
}

// ============================================================ AI accounts: connect, sign in, keys, sign out
const SECURE = location.protocol === "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
function acc() { return (S && S.accounts) || { connections: [], can_add: { subscriptions: [], keys: [] }, catalogue: [], default_model: null, login: null }; }
function readyConns() { return acc().connections.filter(c => c.status === "ready"); }
function priceTxt(m) { const p = S && S.models && S.models.prices[m]; return p ? ` · $${p.input}/$${p.output}` : ""; }
function modelSelect(id, cur, attrs, placeholder) {
  const cat = acc().catalogue, known = new Set(cat.map(c => c.model));
  const opt = c => `<option value="${esc(c.model)}" ${c.model === cur && !placeholder ? "selected" : ""}>${esc(c.model)} · ${esc(c.label)}${c.subscription ? " · your subscription" : priceTxt(c.model)}</option>`;
  const ready = cat.filter(c => c.ready), off = cat.filter(c => !c.ready);
  return `<select class="in" id="${id}" ${attrs || ""}>${placeholder ? `<option value="" selected>${esc(placeholder)}</option>` : ""}
    ${cur && !known.has(cur) && !placeholder ? `<option value="${esc(cur)}" selected>${esc(cur)} (as the code says)</option>` : ""}
    ${ready.length ? `<optgroup label="Your accounts">${ready.map(opt).join("")}</optgroup>` : ""}
    ${off.length ? `<optgroup label="Not connected (connect it in Accounts)">${off.map(opt).join("")}</optgroup>` : ""}
    <option value="__other">another model…</option></select>`;
}
function loginBlock(l) {
  if (!l || (["done", "failed", "cancelled"].includes(l.state) && T.loginSeen === l.id)) return "";
  const word = { starting: "starting…", working: "waiting for you to finish on the company's page…", waiting: "your turn", done: "done", failed: "it didn't work", cancelled: "cancelled" }[l.state] || l.state;
  let h = `<div class="frame loginbox"><div class="row"><b>Signing in to ${esc(l.label)}</b><span class="small">${esc(word)}</span></div>`;
  for (const n of l.notices || []) {
    if (n.type === "auth_url") h += `<p>${esc(n.instructions || "Open the company's page and sign in.")}</p><div class="acts"><a class="btn primary" href="${esc(n.url)}" target="_blank" rel="noopener">Open the sign-in page</a></div>`;
    else if (n.type === "device_code") h += `<p>Open <a href="${esc(n.verification_url)}" target="_blank" rel="noopener">${esc(n.verification_url)}</a> and enter this code:</p><div class="devcode">${esc(n.user_code)}</div><div class="acts"><button class="btn small" data-act="copytext" data-text="${esc(n.user_code)}">Copy the code</button></div>`;
    else if (n.message) h += `<div class="small">${esc(n.message)}</div>`;
  }
  const p = l.prompt;
  if (p) {
    h += `<div class="sec"><div class="lbl">${esc(p.label)}</div>`;
    if (p.type === "select") h += `<div class="acts">${p.options.map(o => `<button class="btn" data-act="loginanswer" data-value="${esc(o.id)}" title="${esc(o.description || "")}">${esc(o.label)}</button>`).join("")}</div>`;
    else if (!SECURE) h += `<div class="banner err">Answer this from the computer Studio runs on, or through its encrypted (https) link: this page came over plain http.</div>`;
    else {
      h += `<div class="row"><input class="in" style="flex:1" ${p.type === "secret" ? 'type="password"' : ""} data-draft="loginanswer" placeholder="${esc(p.placeholder || p.accepted || "")}" data-enter="loginanswer"><button class="btn primary" data-act="loginanswer">Continue</button></div>`;
      if (p.type === "manual_code") h += `<div class="small" style="margin-top:4px">After you sign in, the company's page sends your browser to an address on the computer Studio runs on. If that page doesn't load (it is on another computer), copy its whole address from the address bar and paste it here.</div>`;
    }
    h += `</div>`;
  }
  if (l.state === "done") h += `<p class="good-t">${esc(l.result || "Signed in.")}</p><div class="acts"><button class="btn primary" data-act="loginok" data-id="${l.id}">OK</button></div>`;
  else if (l.state === "failed") h += `<p class="bad-t">${esc(l.error || "The sign-in failed.")}</p><div class="acts"><button class="btn" data-act="loginok" data-id="${l.id}">Close</button></div>`;
  else if (l.state === "cancelled") h += `<div class="acts"><button class="btn" data-act="loginok" data-id="${l.id}">Close</button></div>`;
  else h += `<div class="acts"><button class="btn small" data-act="logincancel" data-id="${l.id}">Cancel</button></div>`;
  return h + `</div>`;
}
function accountsPanel() {
  const a = acc(), keys = a.can_add.keys, prov = draft("keyprov", (keys[0] || {}).provider), kp = keys.find(k => k.provider === prov) || {};
  let h = loginBlock(a.login);
  h += `<div class="sec"><div class="lbl">Connected</div>${a.connections.length ? a.connections.map(c => `<div class="acc"><span class="dot ${c.status === "ready" ? "good" : "bad"}"></span><div class="grow"><b>${esc(c.label)}</b><div class="small">${esc(c.how)}${c.status !== "ready" ? " · " + esc(c.status) : ""}${c.expires && !["never", "unknown"].includes(c.expires) ? " · until " + esc(c.expires) : ""}</div></div>${c.status !== "ready" && c.kind === "subscription" ? `<button class="btn small" data-act="signin" data-provider="${esc(c.provider)}">Sign in again</button>` : ""}${c.can_sign_out ? `<button class="btn small" data-act="signout" data-provider="${esc(c.provider)}">Sign out</button>` : ""}</div>`).join("") : `<div class="small">Nothing yet. Connect one below: Studio needs one AI account to run anything.</div>`}</div>`;
  h += `<div class="sec"><div class="lbl">Sign in with a subscription you already have</div><div class="small" style="margin-bottom:6px">No key needed; it runs within your plan's limits and costs nothing more per use.</div>${a.can_add.subscriptions.map(x => `<div class="acc"><div class="grow"><b>${esc(x.label)}</b><div class="small">${esc(x.about)}${x.certified ? "" : " · not yet certified by lm15"}</div></div><button class="btn small" data-act="signin" data-provider="${esc(x.provider)}">Sign in</button></div>`).join("")}</div>`;
  h += `<div class="sec"><div class="lbl">Or add an API key</div><div class="small" style="margin-bottom:6px">Pay per use, to the company, from your account with it: usually a fraction of a cent per message.</div>`;
  if (!SECURE) h += `<div class="banner err">Keys are accepted only on the computer Studio runs on, or through its encrypted (https) link: this page came over plain http, where anyone on the network could read a key.</div>`;
  else h += `<div class="row"><select class="in" style="width:auto" data-draft="keyprov">${keys.map(k => `<option value="${esc(k.provider)}" ${k.provider === prov ? "selected" : ""}>${esc(k.label)}</option>`).join("")}</select><input class="in" style="flex:1" type="password" autocomplete="off" data-draft="key" data-enter="savekey" placeholder="paste the key"><button class="btn primary" data-act="savekey">Save</button></div>
    ${kp.get_one ? `<div class="small" style="margin-top:4px">No key yet? <a href="${esc(kp.get_one)}" target="_blank" rel="noopener">Get one from ${esc(kp.label)}</a> (you will need an account with them, and some credit).</div>` : ""}`;
  h += `<div class="small" style="margin-top:8px">Keys and sign-ins are kept by lm15 in one file on this computer, private to you, and used by your FunctAI scripts too. Never in the project, never shown again, never sent anywhere but to the company.</div></div>`;
  return h;
}
function drawerAccounts() { return ["AI accounts", "", accountsPanel()]; }
function missingAccount() {
  const miss = ((S.money || {}).accounts || []).filter(a => a.found === false);
  if (!miss.length) return "";
  return `<div class="banner err">${miss.map(a => `${a.steps.map(esc).join(", ")} ${a.steps.length > 1 ? "use" : "uses"} ${esc(a.company)}, which is not connected`).join("; ")}. <a data-act="accounts">Connect it</a>, or <a data-act="models">choose another AI</a>.</div>`;
}
function fillIn() {
  const f = S.fill_in || [];
  const file = n => ((S.wiring || {}).files || {})[n] || n;
  return f.length ? `<div class="banner job">${f.map(n => `<b>${esc(file(n))}</b>`).join(" and ")} ${f.length > 1 ? "are" : "is"} still empty: the AI only sees “Write … here”, so its answers will guess. ${f.map(n => `<a data-act="fillin" data-name="${esc(n)}">Write ${esc(file(n))}</a>`).join(" · ")}</div>` : "";
}
function accountsBlock() {
  const accs = (S.money || {}).accounts || [];
  if (!accs.length) return "";
  return `<div class="sec"><div class="lbl">Who pays</div>${accs.map(a => `<div>${a.local ? I.house : I.cloud} <b>${esc(a.company)}</b> for ${a.steps.map(esc).join(", ")}: ${a.local ? "this computer, free, nothing leaves it." : !a.found ? `<span class="bad-t">not connected</span>, so ${a.steps.length > 1 ? "these steps" : "this step"} cannot run.` : a.subscription ? `your subscription (${esc(a.how || "")}): no cost per use.` : `your account (${esc(a.how || "")}), per use.`}</div>`).join("")}<div class="small" style="margin-top:4px"><a data-act="accounts">Accounts</a>: connect, sign in, add a key.</div></div>`;
}

// ============================================================ home: connect, make, open
function viewHome() {
  const ready = readyConns().length > 0, projs = S.projects || [];
  return `<div class="home"><div id="jobbar"></div>
    <h1>FunctAI Studio</h1>
    <p class="lede">Make small AI helpers you can trust: say what you want, try it on examples, see how often it is right, and fix what goes wrong. It stays on your computer, as ordinary code that is yours.</p>
    <section class="card"><h2><span class="num ${ready ? "done" : ""}">${ready ? I.ok : "1"}</span> Connect an AI</h2>
      ${ready && !T.homeAcc ? `<p>${readyConns().map(c => `<span class="pill">${esc(c.label)}</span>`).join(" ")} <a data-act="homeacc">Manage</a></p>` : accountsPanel()}
      ${ready && T.homeAcc ? `<div class="acts"><button class="btn small" data-act="homeacc">Done</button></div>` : ""}</section>
    <section class="card"><h2><span class="num">2</span> Make something</h2>${T.plan ? planEditor() : makeBox(ready)}</section>
    ${projs.length ? `<section class="card"><h2>Your pipelines</h2>${projs.map(p => `<div class="acc proj" data-act="openproj" data-path="${esc(p.path)}"><b>${esc(p.name)}</b><span class="small">${esc(p.about)}${p.about ? " · " : ""}${p.examples} example${p.examples === 1 ? "" : "s"}</span><span class="small" style="margin-left:auto">${esc(p.path)}</span></div>`).join("")}<div class="small" style="margin-top:6px">In ${esc(S.folder)}</div></section>` : ""}
  </div>`;
}
function makeBox(ready) {
  const what = draft("what", T.what || "");
  return `<textarea class="in" rows="3" data-draft="what" placeholder="What should it do? For example: sort my club's messages into questions, ideas and spam, and answer the questions from our FAQ.">${esc(what)}</textarea>
    <div class="chips" style="justify-content:flex-start;margin-top:8px">${STARTERS.map(([a, b, c]) => `<button class="chip" data-fill="${esc(c)}"><span class="aud">${a}</span>${b}</button>`).join("")}</div>
    <div class="acts" style="margin-top:12px"><button class="btn primary" data-act="draftplan" ${ready && what.trim() ? "" : "disabled"}>Plan it with AI${costTag(T.draftEst)}</button><button class="btn" data-act="blankplan">Plan it myself</button><button class="btn" data-act="example">Try a ready example</button></div>
    <div class="small" style="margin-top:6px">${ready ? "The AI writes a plan (the steps, what each is asked, a few example inputs); you check and change it; Studio writes the code." : "Connect an AI first to plan with it; you can always plan it yourself."} Have a coding agent? <a data-act="copyprompt">Copy a request for it</a>.</div>`;
}

// ---- the plan editor: plain data, turned into code by Studio
function setPath(o, path, v) { const ks = path.split("."); let x = o; for (const k of ks.slice(0, -1)) x = x[k]; x[ks[ks.length - 1]] = v; }
const ANSWER_OPTS = [["text", "text"], ["choices", "one of a few choices"], ["number", "a number"], ["whole number", "a whole number"], ["yes/no", "yes or no"], ["list of texts", "a list of texts"]];
function planEditor() {
  const P = T.plan, inp = (path, v, ph, extra) => `<input class="in" data-plan="${path}" value="${esc(v || "")}" placeholder="${esc(ph || "")}" ${extra || ""}>`;
  let h = `<div class="small" style="margin-bottom:8px">${T.planFromAI ? "The AI's plan: check it, change anything." : "Your plan."} Studio turns it into code when you build it.</div>
    <div class="planrow"><span>Name</span>${inp("name", P.name, "my_helper")}</div>
    <div class="planrow"><span>What it does</span>${inp("description", P.description, "one sentence")}</div>
    <div class="lbl" style="margin-top:12px">It is given</div>`;
  P.inputs.forEach((x, i) => h += `<div class="planrow"><span></span>${inp(`inputs.${i}.name`, x.name, "name, e.g. message", 'style="max-width:180px"')}${inp(`inputs.${i}.about`, x.about, "what it is")}${P.inputs.length > 1 ? `<button class="btn small" data-act="plandel" data-list="inputs" data-i="${i}">×</button>` : ""}</div>`);
  h += `<a class="small" data-act="planadd" data-list="inputs">+ another input</a>
    <div class="lbl" style="margin-top:12px">Facts you provide <span class="small" style="text-transform:none">(the AI cannot know them: an FAQ, prices, rules; each becomes a file you fill in)</span></div>`;
  P.facts.forEach((x, i) => h += `<div class="planrow"><span></span>${inp(`facts.${i}.name`, x.name, "name, e.g. faq", 'style="max-width:180px"')}${inp(`facts.${i}.about`, x.about, "what you will write in it")}<button class="btn small" data-act="plandel" data-list="facts" data-i="${i}">×</button></div>`);
  h += `<a class="small" data-act="planadd" data-list="facts">+ a fact</a>
    <div class="lbl" style="margin-top:12px">Steps, in order <span class="small" style="text-transform:none">(the last step's answer is the result)</span></div>`;
  P.steps.forEach((x, i) => {
    const avail = [...P.inputs.map(z => z.name), ...P.facts.map(z => z.name), ...P.steps.slice(0, i).map(z => z.name)].filter(Boolean);
    h += `<div class="frame"><div class="row">${inp(`steps.${i}.name`, x.name, "step name, e.g. sort", 'style="max-width:200px"')}<span style="margin-left:auto"></span>${i ? `<button class="btn small" data-act="planmove" data-i="${i}" data-d="-1">↑</button>` : ""}${i < P.steps.length - 1 ? `<button class="btn small" data-act="planmove" data-i="${i}" data-d="1">↓</button>` : ""}${P.steps.length > 1 ? `<button class="btn small" data-act="plandel" data-list="steps" data-i="${i}">×</button>` : ""}</div>
      <textarea class="in" rows="3" style="margin-top:6px" data-plan="steps.${i}.instruction" placeholder="What it is asked, as you would ask a careful assistant">${esc(x.instruction || "")}</textarea>
      <div class="row" style="margin-top:6px"><span class="small">It reads</span>${avail.map(a => `<label class="small"><input type="checkbox" data-plantake="${i}" value="${esc(a)}" ${(x.takes || []).includes(a) ? "checked" : ""}> ${esc(a)}</label>`).join(" ")}</div>
      <div class="row" style="margin-top:6px"><span class="small">It answers with</span><select class="in" style="width:auto" data-plan="steps.${i}.answer" data-plan-render="1">${ANSWER_OPTS.map(([v, l]) => `<option value="${v}" ${x.answer === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      ${x.answer === "choices" ? `<input class="in" style="flex:1" data-planchoices="${i}" value="${esc((x.choices || []).join(", "))}" placeholder="the choices, separated by commas">` : ""}</div></div>`;
  });
  h += `<a class="small" data-act="planadd" data-list="steps">+ a step</a>
    <div class="lbl" style="margin-top:12px">Example inputs <span class="small" style="text-transform:none">(never their answers: you judge those later)</span></div>`;
  const names = P.inputs.map(z => z.name);
  P.examples.forEach((e, i) => h += `<div class="planrow"><span class="small">${i + 1}</span>${names.map(n => `<input class="in" data-plan="examples.${i}.${n}" value="${esc(e[n] || "")}" placeholder="${esc(n)}">`).join("")}<button class="btn small" data-act="plandel" data-list="examples" data-i="${i}">×</button></div>`);
  h += `<a class="small" data-act="planadd" data-list="examples">+ an example</a>
    <div class="acts" style="margin-top:14px"><button class="btn primary" data-act="buildplan">Build it</button><button class="btn" data-act="dropplan">Start over</button></div>`;
  return h;
}
function blankPlan(what) {
  return { name: "my_helper", description: what || "", inputs: [{ name: "text", about: "what it works on" }], facts: [],
    steps: [{ name: "answer", instruction: what ? `${what}` : "Say what this text is about, in one sentence.", takes: ["text"], answer: "text", choices: [] }], examples: [] };
}

// ============================================================ examples: add, paste, upload, write with AI, change, remove
function examplesBar() {
  const ins = S.program.inputs, E = S.estimates || {};
  let h = `<div class="row exbar"><button class="btn small ${rows().length ? "" : "primary"}" data-act="exform" data-f="add">+ Add an example</button><button class="btn small" data-act="exform" data-f="paste">Paste many</button><label class="btn small">Upload a CSV<input type="file" accept=".csv,.tsv,.txt,text/csv" id="exfile" hidden></label><button class="btn small" data-act="exwrite" ${readyConns().length ? "" : "disabled"}>Write 10 with AI${costTag(E.examples)}</button><span class="small" style="margin-left:auto">${esc(S.examples_file)}</span></div>`;
  if (T.exForm === "add") h += `<div class="frame">${ins.map(n => `<div class="planrow"><span>${esc(n)}</span><textarea class="in" rows="2" data-draft="exnew:${esc(n)}">${esc(draft("exnew:" + n))}</textarea></div>`).join("")}<div class="acts"><button class="btn primary" data-act="exadd">Add it</button><button class="btn small" data-act="exform" data-f="">Close</button></div><div class="small">Inputs only: you judge its answers after it runs.</div></div>`;
  if (T.exForm === "paste") h += `<div class="frame"><textarea class="in" rows="6" data-draft="expaste" placeholder="${ins.length === 1 ? "One example per line" : `A table with a first line naming the inputs (${ins.join(", ")}), separated by commas or tabs; copied from a spreadsheet works`}">${esc(draft("expaste"))}</textarea><div class="acts"><button class="btn primary" data-act="expaste">Add them</button><button class="btn small" data-act="exform" data-f="">Close</button></div><div class="small">${ins.length === 1 ? "One example per line. A table with a first line saying " + esc(ins[0]) + " works too." : "A spreadsheet's columns copied together paste as a table."}</div></div>`;
  return h;
}
function exampleDrawer(row) {
  if (!row) return ["Example", "", `<div class="small">No example.</div>`];
  if (row.source === "use") return [`A run from elsewhere`, "", `<div class="sec">${Object.entries(row.inputs || {}).map(([k, v]) => fieldBox("u:" + k, k, v)).join("")}</div><div class="acts"><button class="btn primary" data-act="exkeep">Keep it as an example</button></div><div class="small">It was run outside the examples (a script, your agent, a served program).</div>`];
  const n = row.n, ins = S.program.inputs, changed = ins.some(k => draft(`ex:${n}:${k}`, txt(row.inputs[k])) !== txt(row.inputs[k]));
  return [`Example ${esc(n)}`, row.ran ? esc(row.state) : "not run", `<div class="sec">${ins.map(k => `<div class="lbl">${esc(k)}</div><textarea class="in" rows="${Math.min(10, Math.max(2, txt(row.inputs[k]).split("\n").length + 1))}" data-draft="ex:${esc(n)}:${esc(k)}">${esc(draft(`ex:${n}:${k}`, txt(row.inputs[k])))}</textarea>`).join("")}
    <div class="acts">${changed ? `<button class="btn primary" data-act="exsave" data-n="${esc(n)}">Save</button>` : ""}<button class="btn small" data-act="exdelete" data-n="${esc(n)}">Remove this example</button>${row.ran ? `<button class="btn small" data-act="toreview" data-n="${esc(n)}">Read everything, in Review</button>` : `<button class="btn small" data-act="runone">Run it${costTag((S.estimates || {}).per_example)}</button>`}</div>
    <div class="small">Changing an example makes it a new one: its earlier answers and judgements stay with the old text.</div></div>`];
}

function resetView() { T.view = "map"; T.n = null; T.sel = null; T.drawer = null; T.detail = null; T.exForm = null; T.addAfter = null; T.trial = null; history.replaceState(null, "", "#/map"); }
async function saveKey() {
  const key = draft("key", ""), prov = draft("keyprov", (acc().can_add.keys[0] || {}).provider);
  if (!key.trim()) return toast("Paste the key first.");
  const r = await api("/api/accounts/key", { provider: prov, key });
  delete T.drafts.key; const el = document.querySelector('[data-draft="key"]'); if (el) { el.value = ""; el.blur(); }
  toast(`Saved the ${esc(r.label)} key (…${esc(r.ends_with)})${r.checked ? ", and it works" : ""}.`); ETAG = "";
}
async function loginAnswer(value) {
  const l = acc().login; if (!l) return;
  const v = value !== undefined ? value : draft("loginanswer", "");
  await api("/api/accounts/login/answer", { id: l.id, value: v }); delete T.drafts.loginanswer; ETAG = "";
  const a = document.activeElement; if (a && /INPUT|TEXTAREA/.test(a.tagName)) a.blur();
}
