// room/working.js: the words of the "working" line shown before a model's first token.
import { verbs, elapsed } from "../../room/working.js";

Deno.test("verbs: one word at a time, never the same twice in a row, every verb before a repeat", () => {
  const next = verbs();
  const seen = [];
  for (let i = 0; i < 64; i++) seen.push(next());
  for (let i = 1; i < seen.length; i++) if (seen[i] === seen[i - 1]) throw new Error(`repeat at ${i}: ${seen[i]}`);
  for (const v of seen) if (!/^[A-Z][a-z]+ing$/.test(v)) throw new Error(`not a single -ing word: ${v}`);
  const first = new Set(seen.slice(0, 16));
  if (first.size !== 16) throw new Error(`the first 16 are not all different: ${[...first]}`);
});

Deno.test("elapsed: seconds, then minutes and padded seconds", () => {
  const cases = [[0, "0s"], [999, "0s"], [4200, "4s"], [59999, "59s"], [60000, "1m 00s"], [65000, "1m 05s"], [-5, "0s"]];
  for (const [ms, want] of cases) if (elapsed(ms) !== want) throw new Error(`${ms}: ${elapsed(ms)} != ${want}`);
});
