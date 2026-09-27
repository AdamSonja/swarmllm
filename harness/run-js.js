// The run_js probe (docs/design/harness-light.md B.2): run a JS module against a snapshot of the
// project in a hidden, fresh, sandboxed frame, and collect what it printed. The eval suite's
// checks (tests/eval/) run through it today; the run_js tool (B.2) wraps it later.
//
//   probeSnapshot(ws) -> { entry, files }       the project as serve would see it (no port, no logs)
//   runProbe(snap, { code, page, timeout = 3000, width = 800, height = 600, relay, doc })
//     -> { status: "ok"|"error"|"timeout", ms, logs: [{ level, text, src, line, col, ms }] }
//   formatProbe(result) -> text for the model (capped, preview_logs' line format)
//
// The document is buildPreviewDoc's (same CSP, capture script, module rewriting). `code` becomes
// the virtual file __run.js; a classic loader appended to the page (or to a blank page) waits for
// load + one frame, imports it (relative imports resolve like any project module; top-level await
// works), then posts done. With a relay (another site: its own process) a snippet that never ends
// is cut at `timeout` without freezing this page; without one it runs in a blob: frame here.
// Status "error": the snippet threw, or anything (page or snippet) logged an error.
import { PreviewServer } from "./preview.js";
import { buildPreviewDoc } from "./preview-build.js";
import { relayUrl } from "./preview-frame.js";
import { logLine } from "./preview-tools.js";

const enc = new TextEncoder(), dec = new TextDecoder();
const RUN = "__run.js", BLANK = "__probe.html";

export async function probeSnapshot(ws, dir = "") {
  const s = new PreviewServer(ws);
  try { return { entry: "index.html", files: await s._read(dir, null, null) }; } finally { s.close(); }
}

// the snapshot plus __run.js, and the page with the loader: -> { snap, path }
export function probeDoc(snap, { code, page, nonce }) {
  const files = new Map(snap.files);
  files.set(RUN, { type: "text/javascript", bytes: enc.encode(String(code)), hash: "run" });
  const path = page && files.has(page) ? page : BLANK;
  const up = path.split("/").slice(1).map(() => "..").join("/");
  const loader = `<script>addEventListener("load",()=>requestAnimationFrame(()=>setTimeout(async()=>{const t=performance.now();let ok=1;`
    + `try{await import(${JSON.stringify((up ? up + "/" : "./") + RUN)})}catch(e){ok=0;console.error(e)}`
    + `parent.postMessage({pv:${JSON.stringify(nonce)},t:"done",ok,ms:Math.round(performance.now()-t)},"*")})))</script>`;
  const html = path === BLANK ? "<!doctype html><meta charset=utf-8><body>" : dec.decode(files.get(path).bytes);
  const i = html.search(/<\/body\s*>(?![\s\S]*<\/body)/i);
  files.set(path, { type: "text/html", bytes: enc.encode(i >= 0 ? html.slice(0, i) + loader + html.slice(i) : html + loader), hash: "page" });
  return buildPreviewDoc({ entry: path, files }, { path, nonce }).html;
}

export function runProbe(snap, { code, page = null, timeout = 3000, width = 800, height = 600, relay, doc = globalThis.document } = {}) {
  if (relay === undefined) relay = relayUrl(doc);
  const nonce = "p" + crypto.getRandomValues(new Uint32Array(2)).join("-");
  const html = probeDoc(snap, { code, page, nonce });
  const win = doc.defaultView, logs = [], t0 = performance.now();
  return new Promise((resolve) => {
    const frame = doc.createElement("iframe");
    frame.setAttribute("sandbox", "allow-scripts");
    frame.setAttribute("allow", "");
    frame.setAttribute("referrerpolicy", "no-referrer");
    frame.setAttribute("aria-hidden", "true");
    // on screen but invisible: an off-screen or display:none frame gets no animation frames
    frame.style.cssText = `position:fixed;left:0;top:0;width:${width}px;height:${height}px;border:0;opacity:0;pointer-events:none;z-index:-1;background:#fff`;
    let url = null, done = false;
    const end = (status) => {
      if (done) return;
      done = true;
      clearTimeout(timer); win.removeEventListener("message", onMsg); frame.remove();
      if (url) URL.revokeObjectURL(url);
      if (status === "ok" && logs.some((l) => l.level === "error")) status = "error";
      resolve({ status, ms: Math.round(performance.now() - t0), logs });
    };
    const onMsg = (e) => {
      if (e.source !== frame.contentWindow) return;
      const d = e.data;
      if (!d || typeof d !== "object") return;
      if (d.pvr === "hello") { frame.contentWindow.postMessage({ pvr: "doc", html }, "*"); return; }
      if (d.pv !== nonce) return;
      if (d.t === "log") logs.push({ level: String(d.level || "log"), text: String(d.text ?? "").slice(0, 1200), src: String(d.src || ""), line: d.line >>> 0, col: d.col >>> 0, ms: Math.max(0, +d.ms || 0) });
      else if (d.t === "done") setTimeout(() => end(d.ok ? "ok" : "error"), 30);   // let the last logs land
    };
    win.addEventListener("message", onMsg);
    const timer = setTimeout(() => end("timeout"), timeout);
    if (relay) frame.src = relay;
    else { url = URL.createObjectURL(new Blob([html], { type: "text/html" })); frame.src = url; }
    (doc.body || doc.documentElement).appendChild(frame);
  });
}

export function formatProbe(r, { maxLines = 30, maxChars = 2000 } = {}) {
  const head = r.status === "timeout" ? `timed out after ${+(r.ms / 1000).toFixed(1)} s (a loop that never ends?)` : `${r.status} in ${r.ms} ms`;
  if (!r.logs.length) return r.status === "ok" ? `${head} (no output; print results with console.log)` : head;
  const lines = [];
  let size = 0;
  for (const l of r.logs) {
    const t = logLine({ ...l, t: l.ms });
    if (lines.length >= maxLines || size + t.length > maxChars) { lines.push(`(${r.logs.length - lines.length} more lines)`); break; }
    lines.push(t); size += t.length + 1;
  }
  return [head, ...lines].join("\n");
}
