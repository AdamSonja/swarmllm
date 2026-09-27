// harness/run-js.js: the run_js document and result format, with a fake runner (the real frame is
// tests/e2e/preview_browser.mjs).
import { MemoryWorkspace, watch } from "../../harness/workspace.js";
import { PreviewServer } from "../../harness/preview.js";
import { buildPreviewDoc } from "../../harness/preview-build.js";
import { runJsTool, runSnapshot, formatRun, RUN_LINES } from "../../harness/run-js.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const dec = new TextDecoder();
const project = () => watch(new MemoryWorkspace({
  "index.html": "<!doctype html><body><canvas id=c></canvas><script type=module src=game.js></script></body>",
  "game.js": "export const add = (a, b) => a + b;\n",
  "sub/page.html": "<p>x</p>",
}));

Deno.test("runSnapshot: a blank page or the given page, with the loader and __run.js next to it", () => {
  const files = new Map([["index.html", { bytes: new TextEncoder().encode("<body><p>hi</p></body>") }], ["game.js", { bytes: new TextEncoder().encode("export const x = 1;") }]]);
  const a = runSnapshot({ files }, { code: "import { x } from './game.js'; console.log(x);" });
  eq(a.entry, "__run.html");
  ok(dec.decode(a.files.get("__run.js").bytes).includes("console.log(x)"));
  const b = runSnapshot({ files }, { code: "1", page: "index.html" });
  const html = dec.decode(b.files.get("index.html").bytes);
  ok(/<p>hi<\/p><script type="module">[\s\S]*import\("\.\/__run\.js"\)[\s\S]*<\/script><\/body>/.test(html), html);
  ok(!dec.decode(files.get("index.html").bytes).includes("__run"), "the project's own snapshot is untouched");
  // built like any preview: the snippet's import of game.js and the loader's import are data URLs
  const { html: doc, missing } = buildPreviewDoc(a, { nonce: "n" });
  eq(missing, []);
  ok(!/"\.\/__run\.js"/.test(doc) && !/from '\.\/game\.js'/.test(doc), "imports rewritten");
  ok(doc.includes("__pvDone"), "the capture script offers the done hook");
});

Deno.test("run_js: runs through the runner with the project's current files; result format", async () => {
  const ws = project(), s = new PreviewServer(ws);
  let got = null;
  const runner = async (snap, { timeout }) => { got = { snap, timeout }; return { logs: [{ level: "log", text: "3", src: "__run.js", line: 1, col: 9, ms: 12 }], done: { ok: true, ms: 4 } }; };
  const t = runJsTool(s, { runner });
  eq(await t.run({ code: "import { add } from './game.js'; console.log(add(1, 2));" }), "ok in 4 ms\n[0.0s] log __run.js:1:9 3");
  eq(got.timeout, 3000);
  ok(got.snap.files.has("game.js") && got.snap.files.has("__run.js"));
  eq(await t.run({ code: "1", page: "nope.html" }), "error: no nope.html in the project root");
  eq(await t.run({ code: " " }), "error: code is empty");
  await t.run({ code: "1", page: "./sub/page.html" });
  ok(got.snap.files.has("sub/__run.js") && got.snap.entry === "sub/page.html", "next to the page");
  s.close();
});

Deno.test("formatRun: ok without output, errors, timeouts, caps", () => {
  eq(formatRun({ logs: [], done: { ok: true, ms: 3 } }), "ok in 3 ms (no output; print results with console.log)");
  eq(formatRun({ logs: [{ level: "error", text: "Error: bad\n    at f (lib.js:4:3)", src: "__run.js", line: 2, col: 1, ms: 5 }], done: { ok: false, ms: 5 } }),
    "error in 5 ms\n[0.0s] error __run.js:2:1 Error: bad\n  at f (lib.js:4:3)");
  eq(formatRun({ logs: [{ level: "error", text: "Error: no board\n    at bad (lib/t.js:5:9)\n    at __run.js:2:1", src: "__run.html", line: 127, col: 5, ms: 1 }], done: { ok: false, ms: 1 } }),
    "error in 1 ms\n[0.0s] error lib/t.js:5:9 Error: no board\n  at bad (lib/t.js:5:9)\n  at __run.js:2:1", "a throw is placed at its first stack frame, not the loader");
  eq(formatRun({ logs: [{ level: "error", text: "late", ms: 1 }], done: { ok: true, ms: 1 } }).split("\n")[0], "error in 1 ms", "an error logged by the page counts");
  eq(formatRun({ logs: [], done: null }), "timed out after 3 s (a loop that never ends, or an await that never resolves?)");
  eq(formatRun({ logs: [], done: null, hung: true }), "timed out after 3 s (the page hung: a loop that never ends?)");
  const many = formatRun({ logs: Array.from({ length: 100 }, (_, i) => ({ level: "log", text: "line " + i, ms: 1 })), done: { ok: true, ms: 1 } }).split("\n");
  eq(many.length, RUN_LINES + 2); eq(many.at(-1), `(${100 - RUN_LINES} more lines)`);
  ok(formatRun({ logs: Array.from({ length: 10 }, () => ({ level: "log", text: "x".repeat(900), ms: 1 })), done: { ok: true, ms: 1 } }).length < 2100, "char cap");
});
