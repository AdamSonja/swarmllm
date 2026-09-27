// Mounts a port of a PreviewSource (PreviewServer on the host, PreviewSubscriber on a peer) in a
// sandboxed iframe (docs/design/harness-app.md B.1, B.4). The document comes from
// buildPreviewDoc and is loaded with srcdoc; sandbox="allow-scripts" and nothing else, so the
// agent's code runs in an opaque origin: no storage, cookies, OPFS or DOM of the room page.
// Messages from the frame are accepted only from its own window and with this mount's nonce, and
// are treated as untrusted text (typed and capped here, shown with textContent by the UI).
//
//   mountPreview(el, source, port, { onLog, onStatus, autorun = true }) -> { reload(), destroy(), frame, rev }
//   onLog({ level, text, src, line, col, ms, rev }); onStatus({ state: "idle"|"loading"|"ready"|"stopped"|"waiting", rev, path })
// autorun false (a peer's first view) shows a "Run preview :port" button instead of running it.
import { buildPreviewDoc } from "./preview-build.js";

const LEVELS = new Set(["log", "info", "warn", "error"]);
const str = (v, n) => String(v ?? "").slice(0, n);

export function mountPreview(el, source, port, { onLog = () => {}, onStatus = () => {}, autorun = true } = {}) {
  const doc = el.ownerDocument, win = doc.defaultView;
  const frame = doc.createElement("iframe");
  frame.setAttribute("sandbox", "allow-scripts");
  frame.setAttribute("allow", "");
  frame.setAttribute("referrerpolicy", "no-referrer");
  frame.setAttribute("title", `preview :${port}`);
  frame.className = "pv-frame";
  frame.style.cssText = "border:0;width:100%;height:100%;display:block;background:#fff";
  let nonce = "", rev = 0, path = null, detach = () => {}, running = autorun, gate = null;
  // the in-frame capture script rate-limits itself, but the app's code can post directly
  let win0 = 0, count = 0;
  const flood = () => { const now = Date.now(); if (now - win0 > 1000) { win0 = now; count = 0; } return ++count > 300; };
  const status = (state) => { try { onStatus({ state, rev, path: path || source.snapshot(port)?.entry }); } catch (e) { console.error(e); } };

  const load = () => {
    const snap = source.snapshot(port);
    if (!snap) { frame.removeAttribute("srcdoc"); rev = 0; status("stopped"); return; }
    if (!running) return;
    if (path && !snap.files.has(path)) path = null;   // the page went away: back to the entry
    nonce = crypto.getRandomValues(new Uint32Array(2)).join("-");
    rev = snap.rev;
    const { html } = buildPreviewDoc(snap, { path: path || snap.entry, nonce });
    frame.srcdoc = html;
    status("loading");
  };
  const onMessage = (e) => {
    if (e.source !== frame.contentWindow) return;
    const d = e.data;
    if (!d || typeof d !== "object" || d.pv !== nonce || flood()) return;
    const ms = Math.max(0, Number(d.ms) || 0);
    if (d.t === "log") {
      const entry = { level: LEVELS.has(d.level) ? d.level : "log", text: str(d.text, 1200), src: str(d.src, 200), line: d.line >>> 0, col: d.col >>> 0, ms, rev };
      source.pushLog?.(port, entry);
      onLog(entry);
    } else if (d.t === "ready" || d.t === "idle") {
      source.frameEvent?.(port, { t: d.t, rev, ms });
      if (d.t === "ready") status("ready");
    } else if (d.t === "nav") {
      const p = str(d.path, 300);
      const snap = source.snapshot(port);
      if (snap?.files.has(p) && /\.html?$/i.test(p)) { path = p; load(); }
      else {
        const entry = { level: "error", text: `404 ${p} (link)`, src: "", line: 0, col: 0, ms, rev };
        source.pushLog?.(port, entry); onLog(entry);
      }
    }
  };
  win.addEventListener("message", onMessage);
  const off = source.onUpdate((u) => {
    if (u.port !== port) return;
    if (u.stopped) { detach(); detach = () => {}; frame.removeAttribute("srcdoc"); rev = 0; status("stopped"); return; }
    if (running) { detach(); detach = source.attach?.(port) || (() => {}); }
    load();
  });

  const start = () => {
    running = true;
    gate?.remove(); gate = null;
    frame.style.display = "block";
    detach = source.attach?.(port) || (() => {});
    load();
  };
  el.appendChild(frame);
  if (autorun) start();
  else {
    frame.style.display = "none";
    gate = doc.createElement("button");
    gate.type = "button"; gate.className = "pv-run"; gate.textContent = `Run preview :${port}`;
    gate.addEventListener("click", start);
    el.appendChild(gate);
    status("waiting");
  }
  return {
    frame,
    get rev() { return rev; },
    get path() { return path; },
    run: start,
    reload() { if (running) load(); },
    navigate(p) { path = p || null; load(); },
    destroy() { off(); detach(); win.removeEventListener("message", onMessage); frame.remove(); gate?.remove(); },
  };
}
