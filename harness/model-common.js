// Pieces the agent's model adapters share (harness/engine-model.js over one engine,
// harness/room-model.js over the room): the tool-name constraint as a sampler wrapper, streaming
// decode with UTF-8 holdback, an async queue from callbacks to an async iterator, and the map
// from an assistant turn's text to the exact ids it was sampled as. DOM-free.
import { ToolCallConstraint } from "./constrain.js";

// id -> decoded text of that one token, cached (the constraint scans the vocabulary with it)
export function tokenTexts(tok) {
  const texts = [];
  return (id) => (texts[id] ??= tok.decode([id]));
}

// Wrap a sampler with the tool-name constraint. setText(t) tells it the answer so far (call it
// after each emitted token); within one speculative step every sampled column is appended to it
// in order, so each verified position is masked against the text before it, and accepted tokens
// always satisfy the constraint. Without tools it is the base sampler.
export function constrainedSampler(base, tools, { tokenText, vocabSize, style = "xml" }) {
  if (!tools?.length) return { sample: base, setText() {}, constraint: null };
  const C = new ToolCallConstraint(tools, { vocabSize, tokenText, style });
  let text = "", pend = "";
  return {
    sample(lg) {
      C.text = text + pend;
      C.mask(lg);
      const t = base(lg);
      pend += tokenText(t);
      return t;
    },
    setText(t) { text = t; pend = ""; },
    constraint: C,
  };
}

// Streaming decode: push(id) -> the new text, holding back while the tail is an incomplete UTF-8
// sequence (U+FFFD). Decodes only the held-back ids each time, so a long answer stays O(n).
export function deltaDecoder(tok) {
  let held = [];
  const d = {
    ids: [], text: "",
    push(id) {
      d.ids.push(id); held.push(id);
      const s = tok.decode(held);
      if (s.endsWith("�") && held.length < 8) return "";
      held = [];
      d.text += s;
      return s;
    },
    end() {
      if (!held.length) return "";
      const s = tok.decode(held);
      held = [];
      d.text += s;
      return s;
    },
  };
  return d;
}

// Callbacks in, async iterator out: push(x) queues, end(err?) finishes (or fails) the iterator.
export function asyncQueue() {
  const items = [];
  let done = false, error = null, wake = null;
  const poke = () => { const w = wake; wake = null; w?.(); };
  return {
    push(x) { items.push(x); poke(); },
    end(err) { done = true; error = err || null; poke(); },
    async *[Symbol.asyncIterator]() {
      for (;;) {
        if (items.length) { yield items.shift(); continue; }
        if (done) { if (error) throw error; return; }
        await new Promise((r) => (wake = r));
      }
    },
  };
}

// assistant text -> the ids it was sampled as. Re-tokenizing the text can split it differently,
// and then the prompt no longer extends what the caches hold. prune(texts) keeps only the turns
// still in the conversation, so the map does not grow without bound.
export class OwnIds {
  constructor() { this.map = new Map(); }
  get(text) { return this.map.get(text); }
  set(text, ids) { this.map.set(text, Array.from(ids)); }
  prune(texts) { const keep = new Set(texts); for (const k of this.map.keys()) if (!keep.has(k)) this.map.delete(k); }
  clear() { this.map.clear(); }
  get size() { return this.map.size; }
}
