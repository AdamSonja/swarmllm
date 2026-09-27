// The Tabby agent loop: ask the model, run the tools it calls, hand back the results, repeat
// until it answers without calling a tool. The model is any function
//   generate({ system, turns, signal }) -> async iterable of text deltas
// (the room, a single engine, or a script in tests), so the loop knows nothing about GPUs.
//
// turns: [{ role: "user" | "assistant", text }]. Tool results go back as one user turn of
// <tool_response> blocks, as the Qwen templates expect. Assistant turns keep the model's raw
// text (tool-call markup included) so the history the model sees is exactly what it wrote.
// Turns also carry bookkeeping the model ignores: `req` (which request they belong to) and, on a
// tool-result turn, `calls` (what each result was, for the compaction stubs).
//
// Tool: { name, description, parameters, mutates, run(args, { signal, step }) -> string,
//         preview?(args) -> { path, before, after } }   (shown to approve() for mutating tools)
// Events (onEvent): step, delta (raw streamed text), text (visible text), tool-start, tool,
// usage, compacted, trimmed, stopped, done, limit.
import { toolsSystemPrompt, toolResponses, ToolCallParser, parseCallBody } from "./tools.js";

const STOPPED = "(stopped by the user)";
export const CONTEXT_FULL = "context full: start a new task (the files are kept)";

// head and tail of a long tool result (errors are usually at the end, headers at the start)
export function capResult(s, max) {
  if (s.length <= max) return s;
  const cut = s.length - max, head = Math.ceil(max * 0.6), tail = max - head;
  return s.slice(0, head) + `\n…(${cut} chars cut)…\n` + s.slice(s.length - tail);
}
// one line naming a call, e.g. "read_file game.js 1 200"
export function briefCall(c) {
  const vals = Object.values(c.arguments || {}).map((v) => {
    const s = typeof v === "string" ? v : JSON.stringify(v);
    return s.includes("\n") || s.length > 40 ? null : s;
  }).filter((v) => v != null && v !== "");
  return [c.name || "?", ...vals].join(" ");
}

export class Agent {
  // budget: tokens the conversation may take (a number, or a function when the context can change,
  // e.g. roomModel().budget); count: text -> tokens (default ~3.5 characters per token).
  // approve(call, info) -> true | false | { ok: false, reason }: info is tool.preview(args).
  // usage() -> { prompt, reused, generated, tps, reason } after each step (roomModel's stats.last),
  // optional; reason "max" / "ctx" means the answer hit the length cap.
  // idsFor(text) / adopt(text, ids): the model's exact sampled ids, for toJSON / from; idsTag() names
  // the tokenizer they belong to, so a session saved under another model re-encodes its text.
  constructor({ generate, tools, style = "xml", system = "", maxSteps = 24, approve = async () => true, onEvent = () => {},
    budget = Infinity, count = null, maxResultChars = 6000, usage = null, idsFor = null, adopt = null, idsTag = null }) {
    this.generate = generate; this.tools = tools; this.style = style; this.maxSteps = maxSteps;
    this.budget = typeof budget === "function" ? budget : () => budget;
    this.count = count || ((t) => Math.ceil(t.length / 3.5));
    this.approve = approve; this.onEvent = onEvent; this.maxResultChars = maxResultChars;
    this.usage = usage; this.idsFor = idsFor; this.adopt = adopt; this.idsTag = idsTag;
    this.system = toolsSystemPrompt(tools.map(({ name, description, parameters }) => ({ name, description, parameters })), { style, system });
    this.byName = new Map(tools.map((t) => [t.name, t]));
    this.reset();
  }
  reset() { this.turns = []; this.reqs = {}; this.req = 0; }

  // Run one user request to the end. -> { text, steps, calls, reason: "done"|"stopped"|"limit"|"context" }
  async run(userText, { signal } = {}) {
    const req = ++this.req;
    const R = this.reqs[req] = { calls: [], done: false, answer: "" };
    this.turns.push({ role: "user", text: userText, req });
    let calls = 0, shown = "";
    // never leave two user turns in a row: an untouched request is taken back, anything else gets
    // a closing assistant turn
    const close = () => {
      const last = this.turns[this.turns.length - 1];
      if (last?.role !== "user") return;
      if (last.req === req && !last.calls) { this.turns.pop(); delete this.reqs[req]; }
      else this.turns.push({ role: "assistant", text: STOPPED, req });
    };
    // a request that ended any way but "done" is still finished history: compaction may fold it
    const finish = () => { R.done = true; R.answer ||= shown.trim(); };
    const full = (step) => {
      finish(); close();
      this.onEvent({ type: "done", step, reason: "context" });
      return { text: CONTEXT_FULL, steps: step, calls, reason: "context" };
    };
    const stopped = (step) => {
      finish(); close();
      this.onEvent({ type: "stopped", step });
      return { text: shown.trim(), steps: step, calls, reason: "stopped" };
    };
    for (let step = 1; step <= this.maxSteps; step++) {
      if (signal?.aborted) return stopped(step - 1);
      this.onEvent({ type: "step", step });
      if (this._compact(req) === "full") return full(step);
      const P = new ToolCallParser({ schemaFor: (n) => this.byName.get(n)?.parameters });
      let raw = "";
      shown = "";
      const found = [];
      try {
        for await (const d of this.generate({ system: this.system, turns: this.turns, signal })) {
          raw += d;
          this.onEvent({ type: "delta", text: d, step });
          const r = P.feed(d);
          shown += r.text; found.push(...r.calls);
          if (r.text) this.onEvent({ type: "text", text: r.text, step });
          // the tool call being typed, so the UI can show code as it is written (null once it is complete)
          if (P.inCall) this.onEvent({ type: "call-live", raw: P.buf, step });
          else if (r.calls.length) this.onEvent({ type: "call-live", raw: null, step });
        }
      } catch (err) {
        if (err?.name === "ContextFull") {
          if (raw) this.turns.push({ role: "assistant", text: raw, req });
          return full(step);
        }
        if (!signal?.aborted) {
          // the model failed (a device left): keep what it said and close the turn, so the next
          // request does not follow a dangling user turn
          if (raw) this.turns.push({ role: "assistant", text: raw, req });
          finish(); close();
          throw err;
        }
      }
      if (signal?.aborted) {
        // keep what was said (the tool calls in it do not run)
        if (raw) this.turns.push({ role: "assistant", text: raw, req });
        return stopped(step);
      }
      const e = P.end();
      const u = this.usage?.();
      // a call left open by the length cap: its last value is a fragment, so say why instead of running it
      if (u && (u.reason === "max" || u.reason === "ctx")) {
        for (let i = 0; i < e.calls.length; i++) {
          const c = e.calls[i];
          if (!c.open) continue;
          // a write_file cut mid-content: keep its complete lines instead of losing the whole answer,
          // and tell the model exactly where to pick up (otherwise it retries the same long write and
          // is cut at the same place again)
          const part = salvageWrite(c.raw, (n) => this.byName.get(n)?.parameters);
          if (part) { e.calls[i] = part; continue; }
          c.error = `your answer was cut at ${u.generated} tokens before the call was complete; write long files in parts (write_file with append: true)`;
        }
      }
      shown += e.text; found.push(...e.calls);
      if (e.text) this.onEvent({ type: "text", text: e.text, step });
      this.turns.push({ role: "assistant", text: raw, req });
      if (u) this.onEvent({ type: "usage", step, prompt: u.prompt, reused: u.reused, generated: u.generated, tps: u.tps });
      if (!found.length) {
        R.done = true; R.answer = shown.trim();
        this.onEvent({ type: "done", step });
        return { text: shown.trim(), steps: step, calls, reason: "done" };
      }
      const results = [], briefs = [];
      for (const c of found) {
        calls++;
        R.calls.push({ name: c.name, arguments: c.arguments });
        briefs.push(briefCall(c));
        results.push(signal?.aborted ? STOPPED : await this._runCall(c, step, signal));
      }
      this.turns.push({ role: "user", text: toolResponses(results), req, calls: briefs });
      if (signal?.aborted) return stopped(step);
    }
    finish(); close();
    this.onEvent({ type: "limit", steps: this.maxSteps });
    return { text: `(stopped after ${this.maxSteps} steps)`, steps: this.maxSteps, calls, reason: "limit" };
  }

  async _runCall(c, step, signal) {
    let result;
    const t0 = Date.now();
    this.onEvent({ type: "tool-start", call: c, step });
    if (c.error) result = c.open && !/^unterminated/.test(c.error) ? `error: ${c.error}` : `error: ${c.error}. Write the call again in the format the system prompt shows.`;
    else {
      const t = this.byName.get(c.name);
      if (!t) result = `error: there is no tool called ${c.name}; the tools are ${[...this.byName.keys()].join(", ")}`;
      else {
        let verdict = true;
        if (t.mutates) {
          let info = null;
          try { info = (await t.preview?.(c.arguments || {})) ?? null; } catch { info = null; }
          verdict = await this.approve(c, info);
        }
        const ok = verdict === true || (verdict && typeof verdict === "object" && verdict.ok !== false);
        if (!ok) result = "declined by the user" + (verdict?.reason ? `: ${verdict.reason}` : "");
        else {
          try { result = String(await t.run(c.arguments || {}, { signal, step })); }
          catch (err) { result = `error: ${err.message}`; }
          if (c.salvage && !/^error/.test(result)) result += `\nYour answer hit the length limit, so only the first ${c.salvage.lines} lines of ${c.arguments.path} were saved. The last saved line is:\n${c.salvage.last}\nContinue with write_file(path: ${c.arguments.path}, append: true) starting with the line after it. Keep each part under ~120 lines.`;
        }
      }
    }
    result = capResult(result, this.maxResultChars);
    this.onEvent({ type: "tool", call: c, result, step, ms: Date.now() - t0 });
    return result;
  }

  _size() { return this.count(this.system) + this.turns.reduce((n, t) => n + this.count(t.text) + 4, 0); }

  // Past the budget, compact down to 60% of it. Every compaction changes the middle of the prompt
  // (a full re-prefill across the room), so it should be rare and free a lot each time:
  // 1. stub old tool results, 2. fold finished requests into one line each, 3. drop the oldest
  // folded requests, 4. give up ("full"). The current request's own turns are only ever stubbed.
  _compact(cur) {
    const B = this.budget();
    const before = this._size();
    if (before <= B) return null;
    const target = B * 0.6;
    let tier = 0;
    // 1. tool results older than the last 2 steps, oldest first
    const res = this.turns.map((t, i) => i).filter((i) => this.turns[i].role === "user" && this.turns[i].text.startsWith("<tool_response>"));
    let cut = 0;
    for (const i of res.slice(0, -2)) {
      if (this._size() <= target) break;
      const t = this.turns[i];
      let k = 0;
      const text = t.text.replace(/<tool_response>\n([\s\S]*?)\n<\/tool_response>/g, (m, body) => {
        const name = t.calls?.[k++];
        if (body.length <= 200) return m;
        return `<tool_response>\n(output of ${name || "this call"} dropped; run it again if needed)\n</tool_response>`;
      });
      if (text !== t.text) { t.text = text; cut++; tier = 1; }
    }
    if (cut) this.onEvent({ type: "trimmed", turns: cut });
    // 2. fold earlier finished requests, oldest first
    const earlier = [...new Set(this.turns.map((t) => t.req))].filter((r) => r !== cur && this.reqs[r]);
    for (const r of earlier) {
      if (this._size() <= target) break;
      const idx = this.turns.map((t, i) => (t.req === r ? i : -1)).filter((i) => i >= 0);
      if (idx.length <= 2 && this.turns[idx[idx.length - 1]]?.folded) continue;
      const user = this.turns[idx[0]];
      const line = { role: "assistant", text: this._foldLine(r), req: r, folded: true };
      this.turns.splice(idx[0], idx.length, { ...user }, line);
      tier = 2;
    }
    // 3. drop the oldest earlier requests whole
    for (const r of earlier) {
      if (this._size() <= target) break;
      this.turns = this.turns.filter((t) => t.req !== r);
      delete this.reqs[r];
      tier = 3;
    }
    const after = this._size();
    if (tier) this.onEvent({ type: "compacted", tier, before, after });
    if (after > B) { this.onEvent({ type: "compacted", tier: 4, before, after }); return "full"; }
    return tier;
  }
  // "[earlier: wrote index.html, game.js; edited game.js; served :5173; 3 other calls]" + the answer
  _foldLine(r) {
    const R = this.reqs[r], wrote = [], edited = [], served = [];
    let other = 0;
    for (const c of R.calls) {
      const p = c.arguments?.path;
      if (c.name === "write_file" && p) { if (!wrote.includes(p)) wrote.push(p); }
      else if (c.name === "edit_file" && p) { if (!edited.includes(p)) edited.push(p); }
      else if (c.name === "serve") { const s = ":" + (c.arguments?.port || 5173); if (!served.includes(s)) served.push(s); }
      else other++;
    }
    const parts = [];
    if (wrote.length) parts.push("wrote " + wrote.slice(0, 12).join(", ") + (wrote.length > 12 ? ` (+${wrote.length - 12})` : ""));
    if (edited.length) parts.push("edited " + edited.slice(0, 12).join(", ") + (edited.length > 12 ? ` (+${edited.length - 12})` : ""));
    if (served.length) parts.push("served " + served.join(", "));
    if (other) parts.push(`${other} other call${other > 1 ? "s" : ""}`);
    return `[earlier: ${parts.join("; ") || "no tool calls"}]\n` + (R.answer || "").slice(0, 600);
  }

  // Session state. Assistant turns carry the ids they were sampled as when the model knows them.
  toJSON() {
    const tag = this._tag();
    return {
      v: 1, req: this.req, reqs: this.reqs, ...(tag ? { tok: tag } : {}),
      turns: this.turns.map((t) => {
        const o = { ...t };
        if (t.role === "assistant") { const ids = this.idsFor?.(t.text); if (ids) o.ids = Array.from(ids); }
        return o;
      }),
    };
  }
  _tag() { try { return this.idsTag?.() ?? null; } catch { return null; } }
  static from(json, opts) {
    const a = new Agent(opts);
    if (json?.v !== 1) return a;
    a.req = json.req || 0; a.reqs = json.reqs || {};
    const same = (json.tok ?? null) === a._tag();   // ids from another tokenizer (or an untagged save) are noise
    a.turns = (json.turns || []).map(({ ids, ...t }) => {
      if (same && ids && t.role === "assistant") a.adopt?.(t.text, ids);
      return t;
    });
    return a;
  }
}

// A write_file call cut by the length cap: its content up to the last complete line, as a call that
// can run (appending when the model asked to append), with where it stopped. null when the cut
// call is anything else or has no complete line yet.
export function salvageWrite(raw, schemaFor = () => null) {
  if (!raw) return null;
  const c = parseCallBody(raw, schemaFor);
  const a = c?.arguments;
  if (c?.name !== "write_file" || !a || typeof a.path !== "string" || !a.path.trim() || typeof a.content !== "string") return null;
  const cut = a.content.lastIndexOf("\n");
  if (cut < 0) return null;
  const content = a.content.slice(0, cut + 1), lines = content.split("\n").length - 1;
  const last = content.slice(0, -1).split("\n").pop();
  return { name: "write_file", arguments: { path: a.path.trim(), content, append: a.append === true }, salvage: { lines, last } };
}
