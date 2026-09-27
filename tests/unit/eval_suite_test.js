// tests/eval/: the task table is well formed (each golden run only calls real tools, with valid
// arguments), and harness/run-js.js builds the probe document the checks run in. The checks
// themselves run in a browser: node tests/eval/run.mjs --model mock (golden runs pass, bad files fail).
import { TASKS } from "../eval/tasks/index.js";
import { probeDoc, formatProbe } from "../../harness/run-js.js";
import { ToolCallParser } from "../../harness/tools.js";
import { codingTools } from "../../harness/codetools.js";
import { previewTools } from "../../harness/preview-tools.js";
import { MemoryWorkspace } from "../../harness/workspace.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const TOOLS = [...codingTools(new MemoryWorkspace()), ...previewTools({})];
const byName = new Map(TOOLS.map((t) => [t.name, t]));

Deno.test("eval tasks: 12, unique ids, every field a runner needs", () => {
  ok(TASKS.length === 12, `${TASKS.length} tasks`);
  ok(new Set(TASKS.map((t) => t.id)).size === 12, "duplicate ids");
  for (const t of TASKS) {
    ok(typeof t.prompt === "string" && t.prompt.length > 20, `${t.id}: prompt`);
    ok(typeof t.check === "string" && /ok\(/.test(t.check), `${t.id}: check asserts something`);
    ok(Array.isArray(t.mock) && t.mock.length >= 2, `${t.id}: mock run`);
    ok(t.bad && typeof t.bad === "object", `${t.id}: bad files for the self-test`);
    ok(!t.check.includes("${"), `${t.id}: check has an unexpanded template`);
  }
});

Deno.test("eval tasks: the golden runs call only known tools with their required arguments, and end with an answer", async () => {
  for (const t of TASKS) {
    const replies = t.mock.map((r) => (typeof r === "function" ? null : r));
    ok(typeof replies[replies.length - 1] === "string" && !replies[replies.length - 1].includes("<tool_call>"), `${t.id}: last reply is the answer`);
    for (const r of replies.slice(0, -1)) {
      if (r == null) continue;   // a function reply (looks at the conversation)
      const P = new ToolCallParser({ schemaFor: (n) => byName.get(n)?.parameters });
      const calls = [...P.feed(r).calls, ...P.end().calls];
      ok(calls.length === 1, `${t.id}: one call per reply (${calls.length})`);
      const c = calls[0], tool = byName.get(c.name);
      ok(tool, `${t.id}: unknown tool ${c.name}`);
      for (const k of tool.parameters.required || []) ok(k in c.arguments, `${t.id}: ${c.name} without ${k}`);
    }
  }
});

Deno.test("probeDoc: __run.js is imported by a loader at the end of the page, relative to it", () => {
  const enc = new TextEncoder(), f = (s, type = "text/html") => ({ type, bytes: enc.encode(s), hash: s.length + "" });
  const files = new Map([["index.html", f("<!doctype html><body><p>hi</p></body></html>")], ["sub/p.html", f("<body>x</body>")], ["lib.js", f("export const a = 1;", "text/javascript")]]);
  const html = probeDoc({ files }, { code: "import { a } from './lib.js'; console.log(a);", page: "index.html", nonce: "n1" });
  ok(/<p>hi<\/p><script>addEventListener\("load"/.test(html), "loader right before </body>");
  ok(!html.includes('"./__run.js"') && /import\("data:text\/javascript/.test(html), "the import became a data: URL");
  ok(html.includes('pv:"n1"'), "done carries the nonce");
  const sub = probeDoc({ files }, { code: "1", page: "sub/p.html", nonce: "n" });
  ok(/import\("data:text\/javascript/.test(sub), "a page in a folder imports ../__run.js");
  const blank = probeDoc({ files }, { code: "1", page: null, nonce: "n" });
  ok(/import\("data:text\/javascript/.test(blank) && !blank.includes("<p>hi</p>"), "no page: a blank one");
});

Deno.test("formatProbe: status line, log lines, caps", () => {
  ok(formatProbe({ status: "ok", ms: 12, logs: [] }) === "ok in 12 ms (no output; print results with console.log)");
  ok(formatProbe({ status: "timeout", ms: 3004, logs: [] }).startsWith("timed out after 3 s"));
  const r = formatProbe({ status: "error", ms: 5, logs: [{ level: "error", text: "boom", src: "a.js", line: 3, col: 1, ms: 0 }] });
  ok(r === "error in 5 ms\n[0.0s] error a.js:3:1 boom", r);
  const many = formatProbe({ status: "ok", ms: 1, logs: Array.from({ length: 50 }, (_, i) => ({ level: "log", text: "x" + i, src: "", line: 0, col: 0, ms: 0 })) });
  ok(many.split("\n").length === 32 && many.endsWith("(20 more lines)"), many.slice(-40));
});
