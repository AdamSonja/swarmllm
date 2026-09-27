// run_js: the agent runs a JS module in the preview sandbox and reads what it printed
// (docs/design/harness-light.md B.2). It checks logic ("does clearLines() drop a full row?"),
// not only "the page has no console errors".
//
// The project is snapshotted like serve does (without registering a port or touching the visible
// preview), the snippet is added as __run.js next to the page, and a loader module that imports it
// goes at the end of that page (or of a blank page). The document is built by buildPreviewDoc
// (imports rewritten, capture script, CSP) and mounted by mountPreview in a hidden, fresh,
// sandboxed frame: through the relay when there is one, so `while (true) {}` cannot freeze the
// room. Console output and errors are collected until the loader says done, or for TIMEOUT_MS.
//
//   runJsTool(server, { runner }) -> the tool; runner(snapshot, { timeout }) ->
//     { logs: [{ level, text, src, line, col, ms }], done: { ok, ms } | null, hung }
//   runSnapshot(snapshot, { code, page }) -> the snapshot to build (pure; unit-tested)
//
// The eval suite's checks (tests/eval/) run through the same runner:
//   probeSnapshot(ws, dir?) -> { dir, entry, files }   the project as serve would see it
//   runProbe(snap, { code, page, timeout, width, height, doc, runner })
//     -> { status: "ok"|"error"|"timeout", ms, logs, done, hung, timeout }
//   formatProbe(result) -> formatRun's text
import { mountPreview } from "./preview-frame.js";
import { fold } from "./preview-tools.js";
import { mimeFor } from "./preview-build.js";
import { PreviewServer } from "./preview.js";

export const TIMEOUT_MS = 3000, RUN_LINES = 30, RUN_CHARS = 2000;
const enc = new TextEncoder(), dec = new TextDecoder();
const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/") + 1) : "");

// Waits for the page's load (at most 1 s: a page can hang its own load), imports the snippet, and
// reports done through the capture script (window.__pvDone, which knows the mount's nonce).
const LOADER = `<script type="module">
const done = (ok, t0) => window.__pvDone?.(ok, Math.round(performance.now() - t0));
if (document.readyState !== "complete") await Promise.race([new Promise((r) => addEventListener("load", r, { once: true })), new Promise((r) => setTimeout(r, 1000))]);
await new Promise((r) => setTimeout(r, 0));
const t0 = performance.now();
try { await import("./__run.js"); done(true, t0); } catch (e) { console.error(e); done(false, t0); }
</script>`;

export function runSnapshot(snap, { code, page = "" }) {
  const files = new Map(snap.files);
  let entry = page;
  let html;
  if (page) html = dec.decode(files.get(page).bytes);
  else { entry = "__run.html"; html = "<!doctype html><html><head><meta charset=\"utf-8\"></head><body></body></html>"; }
  const L = LOADER;
  const i = html.search(/<\/body\s*>(?![\s\S]*<\/body\s*>)/i);
  html = i >= 0 ? html.slice(0, i) + L + html.slice(i) : html + L;
  files.set(entry, { type: "text/html", bytes: enc.encode(html), hash: "run" });
  files.set(dirOf(entry) + "__run.js", { type: mimeFor("x.js"), bytes: enc.encode(code), hash: "run" });
  return Object.freeze({ port: 0, dir: snap.dir || "", entry, rev: 1, bytes: 0, files });
}

// The browser runner: a hidden frame through mountPreview, destroyed when done or after `timeout`.
export function browserRunner(doc = globalThis.document, { width = 480, height = 360 } = {}) {
  return (snap, { timeout = TIMEOUT_MS } = {}) => new Promise((resolve) => {
    const el = doc.createElement("div");
    // on screen (throttled rendering off-screen would stop requestAnimationFrame) but invisible
    el.style.cssText = `position:fixed;left:0;top:0;width:${width}px;height:${height}px;opacity:0;pointer-events:none;z-index:-1;overflow:hidden`;
    doc.body.append(el);
    const logs = [];
    let view = null, timer = 0, fin = false;
    const end = (r) => {
      if (fin) return;
      fin = true; clearTimeout(timer);
      queueMicrotask(() => { view?.destroy(); el.remove(); });
      resolve({ logs, ...r });
    };
    const source = {
      snapshot: () => snap, onUpdate: () => () => {}, attach: () => () => {}, frameEvent() {},
      pushLog() {},
    };
    view = mountPreview(el, source, 0, {
      onLog: (e) => { if (e.src !== "(preview)") logs.push(e); else if (/hung/.test(e.text)) end({ done: null, hung: true }); },
      onStatus: (s) => { if (s.state === "hung") end({ done: null, hung: true }); },
      onShow: () => { if (!timer) timer = setTimeout(() => end({ done: null, hung: false }), timeout); },
      onDone: (d) => end({ done: { ok: !!d.ok, ms: Math.max(0, Number(d.ms) || 0) } }),
    });
  });
}

export function runJsTool(server, { runner = null, timeout = TIMEOUT_MS } = {}) {
  return {
    name: "run_js", mutates: false,
    description: "Run a JS module in the sandbox, after loading page if given; it can import project files. Returns console output.",
    parameters: { type: "object", properties: { code: { type: "string" }, page: { type: "string" } }, required: ["code"] },
    async run({ code = "", page = "" } = {}) {
      if (!String(code).trim()) return "error: code is empty";
      page = String(page || "").replace(/^\.?\//, "");
      const dir = server.ports()[0]?.dir || "";
      const files = await server._read(dir, null, null);
      if (page && !files.has(page)) return `error: no ${page} in ${dir || "the project root"}`;
      const r = await (runner || browserRunner())(runSnapshot({ dir, files }, { code: String(code), page }), { timeout });
      return formatRun(r, timeout);
    },
  };
}

export function formatRun({ logs = [], done = null, hung = false }, timeout = TIMEOUT_MS) {
  const errs = logs.some((e) => e.level === "error");
  const head = done ? `${done.ok && !errs ? "ok" : "error"} in ${done.ms} ms`
    : `timed out after ${timeout / 1000} s${hung ? " (the page hung: a loop that never ends?)" : " (a loop that never ends, or an await that never resolves?)"}`;
  if (!logs.length) return done?.ok ? `${head} (no output; print results with console.log)` : head;
  const rows = fold(logs.map((e) => ({ ...e, t: e.ms ?? 0, rev: 0, ...atThrow(e) })));
  const out = [head];
  let size = head.length, n = 0;
  for (const r of rows) {
    if (n >= RUN_LINES || size + r.text.length + 1 > RUN_CHARS) { out.push(`(${rows.length - n} more lines)`); break; }
    out.push(r.text); size += r.text.length + 1; n++;
  }
  return out.join("\n");
}

// the loader logs a thrown error from its own line: name the stack's first frame instead
function atThrow(e) {
  if (!/(^|\/)__run\.html$/.test(e.src || "")) return {};
  const m = /^\s+at .*?\(?([^\s()]+):(\d+):(\d+)\)?\s*$/m.exec(e.text || "");
  return m ? { src: m[1], line: +m[2], col: +m[3] } : { src: "", line: 0, col: 0 };
}

// ---- the eval suite's probe: the same runner, with a status for pass/fail
export async function probeSnapshot(ws, dir = "") {
  const s = new PreviewServer(ws);
  try { return { dir, entry: "index.html", files: await s._read(dir, null, null) }; } finally { s.close(); }
}

export async function runProbe(snap, { code, page = null, timeout = TIMEOUT_MS, width = 800, height = 600, doc = globalThis.document, runner = null } = {}) {
  const t0 = performance.now();
  page = page && snap.files.has(page) ? page : "";
  const r = await (runner || browserRunner(doc, { width, height }))(runSnapshot(snap, { code: String(code), page }), { timeout });
  const errs = r.logs.some((e) => e.level === "error");
  const status = r.done ? (r.done.ok && !errs ? "ok" : "error") : "timeout";
  return { ...r, status, ms: r.done ? r.done.ms : Math.round(performance.now() - t0), timeout };
}

export const formatProbe = (r) => formatRun(r, r.timeout ?? TIMEOUT_MS);
