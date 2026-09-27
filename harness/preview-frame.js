// Mounts a port of a PreviewSource (PreviewServer on the host, PreviewSubscriber on a peer) in a
// sandboxed iframe (docs/design/harness-app.md B.1, B.4). The document comes from buildPreviewDoc.
// sandbox="allow-scripts" and nothing else, so the agent's code runs in an opaque origin: no
// storage, cookies, OPFS or DOM of the room page.
//
// Two ways to run it:
//   relay  - the frame is harness/preview-relay.html on another site (relayUrl()), which renders the
//            document in its own sandboxed child. Site isolation gives it its own process, so an
//            infinite loop or a memory bomb in the agent's code hangs that process, not the room:
//            the relay's heartbeat stops, and after HANG_MS the frame is removed and the hang logged.
//   local  - no other site configured: the document is a blob: URL in a frame of this page. Same
//            renderer process as the room (a hang freezes the tab), but still an opaque origin.
// Either way the document is a blob: URL, not srcdoc: a srcdoc document inherits the room page's
// URL as its base, and that URL carries the room code. A frame that navigates itself away
// (location.href = ..., meta refresh) is put back on the current rev, and stopped if it keeps doing it.
// Messages from the frame are accepted only from its own window and with this mount's nonce, and
// are treated as untrusted text (typed and capped here, shown with textContent by the UI).
//
//   mountPreview(el, source, port, { onLog, onStatus, autorun = true, relay = relayUrl(), onShow, onDone })
//     -> { reload(), destroy(), frame, rev }
//   onLog({ level, text, src, line, col, ms, rev })
//   onStatus({ state: "idle"|"loading"|"ready"|"stopped"|"waiting"|"hung", rev, path })
// autorun false (a peer's first view) shows a "Run preview :port" button instead of running it.
import { buildPreviewDoc } from "./preview-build.js";

const LEVELS = new Set(["log", "info", "warn", "error"]);
const str = (v, n) => String(v ?? "").slice(0, n);
export const HANG_MS = 3000, HELLO_MS = 5000, MAX_NAV = 3;

// The relay's address: <meta name="preview-origin" content="https://..."> on the page (a second
// deployment of this site on another registrable domain), else in development the other loopback
// name (localhost <-> 127.0.0.1 are different sites), else null (local mode).
export function relayUrl(doc = globalThis.document) {
  const loc = doc?.defaultView?.location;
  if (!loc) return null;
  const meta = doc.querySelector?.('meta[name="preview-origin"]')?.content?.trim();
  const path = "/harness/preview-relay.html";
  if (meta) return meta.replace(/\/+$/, "") + path;
  const port = loc.port ? ":" + loc.port : "";
  if (loc.hostname === "localhost") return `${loc.protocol}//127.0.0.1${port}${path}`;
  if (loc.hostname === "127.0.0.1") return `${loc.protocol}//localhost${port}${path}`;
  return null;
}

export function mountPreview(el, source, port, { onLog = () => {}, onStatus = () => {}, autorun = true, relay = undefined, onShow = null, onDone = null } = {}) {
  const doc = el.ownerDocument, win = doc.defaultView;
  if (relay === undefined) relay = relayUrl(doc);
  try { if (relay) new URL(relay); } catch { relay = null; }
  let frame = null, mode = relay ? "relay" : "local";
  let nonce = "", rev = 0, path = null, detach = () => {}, running = autorun, gate = null;
  let url = null, expect = 0, navs = 0, html = null;
  let hello = false, beat = 0, dog = 0, helloTimer = 0, hung = false;
  // the in-frame capture script rate-limits itself, but the app's code can post directly
  let win0 = 0, count = 0;
  const flood = () => { const now = Date.now(); if (now - win0 > 1000) { win0 = now; count = 0; } return ++count > 300; };
  const status = (state) => { try { onStatus({ state, rev, path: path || source.snapshot(port)?.entry }); } catch (e) { console.error(e); } };
  const log = (level, text) => {
    const entry = { level, text, src: "(preview)", line: 0, col: 0, ms: 0, rev };
    source.pushLog?.(port, entry); onLog(entry);
  };

  const makeFrame = () => {
    frame = doc.createElement("iframe");
    frame.setAttribute("sandbox", "allow-scripts");
    frame.setAttribute("allow", "");
    frame.setAttribute("referrerpolicy", "no-referrer");
    frame.setAttribute("title", `preview :${port}`);
    frame.className = "pv-frame";
    frame.style.cssText = "border:0;width:100%;height:100%;display:block;background:#fff";
    hello = false; expect = 0; navs = 0;
    if (mode === "relay") {
      expect = 1;
      frame.src = relay;
      clearTimeout(helloTimer);
      // the relay never answered (not deployed, blocked): run here instead
      helloTimer = setTimeout(() => {
        if (hello || !frame) return;
        log("warn", "the isolated preview host did not answer; running the preview in this tab");
        mode = "local"; frame.remove(); makeFrame(); if (running) load();
      }, HELLO_MS);
    }
    el.appendChild(frame);
    frame.addEventListener("load", onFrameLoad);   // after inserting: the empty frame's about:blank load is not a navigation
  };
  const blank = () => {
    html = null;
    if (!frame) return;
    if (mode === "relay") { if (hello) frame.contentWindow?.postMessage({ pvr: "doc", html: "" }, "*"); return; }
    expect = 1; frame.src = "about:blank";
    if (url) { URL.revokeObjectURL(url); url = null; }
  };
  const show = () => {
    if (!frame || html == null) return;
    if (mode === "relay") {
      if (!hello) return;   // sent on hello
      frame.contentWindow?.postMessage({ pvr: "doc", html }, "*");   // the relay is sandboxed too: an opaque origin
      onShow?.();
      return;
    }
    const old = url;
    url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    expect = 1; frame.src = url;
    onShow?.();
    if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
  };
  // local mode: every load we did not start is the page navigating itself somewhere
  function onFrameLoad() {
    if (mode !== "local") return;
    if (expect) { expect = 0; return; }
    navBlocked();
  }
  const navBlocked = (stopped = false) => {
    if (stopped || ++navs > MAX_NAV) {
      log("error", "preview stopped: the page keeps navigating away from itself");
      if (mode === "local") blank();
      status("stopped");
      return;
    }
    log("error", "navigation blocked: the preview tried to leave its page (location.href / meta refresh); reloaded rev " + rev);
    if (mode === "local") show();
  };

  const load = () => {
    const snap = source.snapshot(port);
    if (!snap) { blank(); rev = 0; status("stopped"); return; }
    if (!running) return;
    if (hung) { hung = false; gate?.remove(); gate = null; makeFrame(); }
    if (path && !snap.files.has(path)) path = null;   // the page went away: back to the entry
    nonce = crypto.getRandomValues(new Uint32Array(2)).join("-");
    rev = snap.rev; navs = 0;
    html = buildPreviewDoc(snap, { path: path || snap.entry, nonce }).html;
    show();
    status("loading");
  };

  // relay mode: no heartbeat for HANG_MS while the tab is visible means the agent's code hung the
  // preview's process. Background tabs throttle timers, so a hidden tab never counts as hung.
  const watchdog = () => {
    clearInterval(dog);
    beat = Date.now();
    dog = setInterval(() => {
      if (doc.visibilityState !== "visible") { beat = Date.now(); return; }
      if (Date.now() - beat > HANG_MS) onHang();
    }, 500);
  };
  const onHang = () => {
    clearInterval(dog); dog = 0;
    hung = true; hello = false;
    frame?.remove(); frame = null;
    detach(); detach = () => {};
    log("error", `preview hung (infinite loop?): no answer for ${HANG_MS / 1000} s, so it was stopped; it runs again on the next edit`);
    status("hung");
    gate = doc.createElement("button");
    gate.type = "button"; gate.className = "pv-run"; gate.textContent = `preview hung · run :${port} again`;
    gate.addEventListener("click", () => { detach = source.attach?.(port) || (() => {}); load(); });
    el.appendChild(gate);
  };

  const onMessage = (e) => {
    if (!frame || e.source !== frame.contentWindow) return;
    const d = e.data;
    if (!d || typeof d !== "object") return;
    if (mode === "relay" && typeof d.pvr === "string") {
      if (d.pvr === "hello" && !hello) { hello = true; clearTimeout(helloTimer); watchdog(); show(); }
      else if (d.pvr === "beat") beat = Date.now();
      else if (d.pvr === "nav") navBlocked(!!d.stopped);
      return;
    }
    if (d.pv !== nonce || flood()) return;
    const ms = Math.max(0, Number(d.ms) || 0);
    if (d.t === "log") {
      const entry = { level: LEVELS.has(d.level) ? d.level : "log", text: str(d.text, 1200), src: str(d.src, 200), line: d.line >>> 0, col: d.col >>> 0, ms, rev };
      source.pushLog?.(port, entry);
      onLog(entry);
    } else if (d.t === "ready" || d.t === "idle") {
      source.frameEvent?.(port, { t: d.t, rev, ms });
      if (d.t === "ready") status("ready");
    } else if (d.t === "done") onDone?.(d);   // run_js's snippet finished
    else if (d.t === "nav") {
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
    if (u.stopped) { detach(); detach = () => {}; blank(); rev = 0; status("stopped"); return; }
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
  makeFrame();
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
    get frame() { return frame; },
    get mode() { return mode; },
    get rev() { return rev; },
    get path() { return path; },
    get loaded() { return html != null; },
    run: start,
    reload() { if (running) load(); },
    navigate(p) { path = p || null; load(); },
    destroy() {
      off(); detach(); clearInterval(dog); clearTimeout(helloTimer);
      win.removeEventListener("message", onMessage);
      frame?.remove(); gate?.remove();
      if (url) URL.revokeObjectURL(url);
    },
  };
}

// "open ↗": the page in a tab of its own. The tab is a blob: page of this origin holding only a
// sandboxed frame (allow-scripts, as here), so the agent's code still runs in an opaque origin;
// its srcdoc's base is that blob: URL, which says nothing about the room. noopener: no way back.
export function openPreviewTab(source, port, path = null) {
  const snap = source.snapshot(port);
  if (!snap) return false;
  const { html } = buildPreviewDoc(snap, { path: path && snap.files.has(path) ? path : snap.entry, nonce: "" });
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
  const page = `<!doctype html><meta charset="utf-8"><title>localhost:${port}</title>`
    + `<style>html,body{margin:0;height:100%;background:#fff}iframe{border:0;width:100%;height:100%;display:block}</style>`
    + `<iframe sandbox="allow-scripts" allow="" referrerpolicy="no-referrer" srcdoc="${esc(html)}"></iframe>`;
  const url = URL.createObjectURL(new Blob([page], { type: "text/html" }));
  globalThis.open?.(url, "_blank", "noopener");
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return true;
}
