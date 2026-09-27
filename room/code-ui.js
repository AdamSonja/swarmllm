// Code mode's DOM (docs/design/harness-app.md E): the agent timeline, tool and diff cards, the
// file tree and the preview pane. No agent or room logic here: room/code.js drives it.
//
// The timeline is rendered only from ai-code-* messages, on the host as on the peers (the host
// renders its own messages before broadcasting them), so every screen shows the same thing.
// Everything that came from the model or the preview is untrusted text: it goes in with
// textContent, and model prose through mdChat (room/markdown.js), which escapes first.
import { mdChat } from "./markdown.js";

const $ = (id) => document.getElementById(id);
const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const CON_MAX = 300;   // console rows kept per port

// rows: lineDiff output as [[op, text, skip?]] (the wire form). Too long to diff: head / tail
// (the first and last lines of the proposed file) and, on the host, full for "view full file".
function diffBlock(d, onFull) {
  const box = h("div", "cm-diff" + (d.isNew ? " new" : ""));
  const head = h("div", "dh");
  head.append(h("b", null, d.path));
  if (d.isNew) head.append(h("span", null, `new file · ${plural(d.lines, "line")}`));
  else { head.append(h("span", "add", `+${d.add}`), h("span", "del", `-${d.del}`)); }
  if (d.full != null && onFull) {
    const b = h("button", "full", "view full file"); b.type = "button";
    b.onclick = () => onFull(d.path, d.full);
    head.append(b);
  }
  const rows = h("div", "rows");   // as wide as the longest line, so every row's tint spans it
  if (d.risky) box.append(h("div", "risk", "this file can run commands on your machine (a script, a hook, an npm script): read it before approving"));
  box.append(head, rows);
  if (!d.rows) {
    const all = [...(d.head || [])], tail = d.tail || [];
    for (const t of all) rows.append(h("div", "r r-add", t || " "));
    const hidden = d.lines - all.length - tail.length;
    if (hidden > 0 || !all.length) rows.append(h("div", "r r-skip", all.length ? `… ${plural(hidden, "more line")} (view full file to read them)` : d.isNew ? `(${plural(d.lines, "line")}, too long to show)` : "(too many changes to show)"));
    for (const t of tail) rows.append(h("div", "r r-add", t || " "));
    return box;
  }
  if (!d.isNew && !d.add && !d.del) { rows.append(h("div", "r r-skip", "(no changes)")); return box; }
  for (const [op, text, skip] of d.rows) {
    if (skip) { rows.append(h("div", "r r-skip", `… ${plural(skip, "unchanged line")}`)); continue; }
    rows.append(h("div", "r" + (op === "+" ? " r-add" : op === "-" ? " r-del" : ""), text || " "));
  }
  if (d.more) rows.append(h("div", "r r-skip", `… ${plural(d.more, "more line")}`));
  return box;
}

export function codeUI({ onMode = () => {} } = {}) {
  const pane = $("chatpane"), log = $("code-log");
  let host = false, mode = "chat", empty = null, waiting = 0;
  const title0 = document.title;
  // short announcements for screen readers (the streamed log itself is not live)
  const say = (text) => { const s = $("code-status"); if (s) s.textContent = text; };

  // ---------------- mode switch
  function show(m) {
    mode = m;
    pane.classList.toggle("code-mode", m === "code");
    $("mode-chat").setAttribute("aria-selected", String(m === "chat"));
    $("mode-code").setAttribute("aria-selected", String(m === "code"));
    if (m === "code" && !waiting) $("mode-code").classList.remove("fresh");
    onMode(m);
  }
  $("mode-chat").addEventListener("click", () => show("chat"));
  // Left / Right move between the tabs of a tablist
  const arrows = (list) => list.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const tabs = [...list.querySelectorAll('[role="tab"]')].filter((t) => !t.hidden), i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const t = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    e.preventDefault(); t.focus(); t.click();
  });
  arrows($("mode-bar")); arrows($("code-out-tabs"));
  const poke = () => { $("mode-bar").hidden = false; if (mode !== "code") $("mode-code").classList.add("fresh"); };

  // ---------------- timeline
  const near = () => log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  const add = (el) => { const stick = near(); empty?.remove(); empty = null; log.append(el); if (stick) log.scrollTop = log.scrollHeight; return el; };
  const key = (...a) => a.join(":");
  const find = (k) => log.querySelector(`[data-k="${CSS.escape(k)}"]`);
  function placeholder(text) {
    log.replaceChildren();
    empty = h("div", "cm-empty");
    empty.innerHTML = text;
    log.append(empty);
  }
  function clear() { log.replaceChildren(); empty = null; viewFile(null); }

  function apply(d) {
    switch (d.t) {
      case "ai-code-start": {
        say("the agent started");
        const u = h("div", "cm-user");
        u.append(h("div", "who", d.name || "host"), h("div", "bubble", d.text));
        add(u);
        break;
      }
      case "ai-code-tok": {
        const k = key("t", d.mid, d.step);
        let el = find(k);
        if (!el) { el = add(h("div", "cm-text")); el.dataset.k = k; el.dataset.raw = ""; el.setAttribute("aria-busy", "true"); }
        el.dataset.raw += d.text;
        const stick = near();
        el.innerHTML = mdChat(el.dataset.raw.replace(/^\s+/, ""));
        if (!el.dataset.raw.trim()) el.remove();
        if (stick) log.scrollTop = log.scrollHeight;
        break;
      }
      case "ai-code-tool": toolCard(d); break;
      case "ai-code-note": add(h("div", "cm-note" + (d.err ? " err" : ""), d.text)); break;
      case "ai-code-done": {
        for (const t of log.querySelectorAll('.cm-text[aria-busy="true"]')) t.removeAttribute("aria-busy");
        const text = d.stats || `${plural(d.steps || 0, "step")} · ${d.reason || "done"}`;
        add(h("div", "cm-stats", text));
        say("the agent finished: " + text);
        break;
      }
    }
  }

  function toolCard(d) {
    const k = key("c", d.mid, d.i);
    let el = find(k);
    if (!el) {
      // the model's text before a tool call is complete once the call starts
      for (const t of log.querySelectorAll('.cm-text[aria-busy="true"]')) t.removeAttribute("aria-busy");
      el = add(h("div", "cm-tool"));
      el.dataset.k = k;
      el.dataset.name = d.name || "?";
      const det = h("details"), sum = h("summary");
      sum.append(h("span", "nm", el.dataset.name), h("span", "br", ""), h("span", "ms", ""), h("span", "chip", ""));
      det.append(sum);
      el.append(det);
    }
    const stick = near(), det = el.firstChild, sum = det.firstChild;
    if (d.brief != null) {
      sum.querySelector(".br").textContent = d.brief.startsWith(el.dataset.name) ? d.brief.slice(el.dataset.name.length).trim() : d.brief;
      sum.querySelector(".br").title = d.brief;
    }
    const chip = sum.querySelector(".chip");
    if (d.state) {
      chip.className = "chip " + d.state.replace(/[^a-z]/g, "");
      chip.textContent = d.state === "pending" ? "needs approval" : d.state;
      if (d.state === "running") say(`running ${el.dataset.name}`);
      else if (d.state === "pending") say(`${el.dataset.name} needs approval`);
      else if (d.state === "done" || d.state === "error") say(`${el.dataset.name} ${d.state}`);
    }
    if (d.ms != null) sum.querySelector(".ms").textContent = d.ms < 1000 ? `${d.ms} ms` : `${(d.ms / 1000).toFixed(1)} s`;
    if (d.result != null) {
      let pre = det.querySelector("pre.res");
      if (!pre) { pre = h("pre", "res"); det.append(pre); }
      pre.textContent = d.result;
    }
    if (d.diff && !el.querySelector(".cm-diff")) el.append(diffBlock(d.diff, host ? viewFull : null));
    // peers (and the host's own record) see the pending state; the host adds buttons with ask()
    let ap = el.querySelector(".cm-approve");
    if (d.state === "pending" && !host && !ap) { ap = h("div", "cm-approve"); ap.append(h("span", "wait", "waiting for the host's approval")); el.append(ap); }
    if (d.state !== "pending") ap?.remove();
    if (d.state === "error") det.open = true;
    if (stick) log.scrollTop = log.scrollHeight;
    return el;
  }

  // host only: Approve / Reject… / Allow edits for this task, on the card of call i. While it
  // waits, the Code tab carries a dot and the page title says so (the host may be in Chat).
  function waitMark(on) {
    waiting = Math.max(0, waiting + (on ? 1 : -1));
    if (waiting) { $("mode-code").classList.add("fresh"); document.title = "(needs approval) " + title0; }
    else { if (mode === "code") $("mode-code").classList.remove("fresh"); document.title = title0; }
  }
  function ask(mid, i, { risky = false } = {}) {
    const el = find(key("c", mid, i));
    if (!el) return Promise.resolve({ ok: false, reason: "approval card missing" });
    el.querySelector(".cm-approve")?.remove();
    const ap = h("div", "cm-approve");
    const yes = h("button", "ok", "Approve"), no = h("button", null, "Reject…"), all = h("button", null, "Allow edits for this task");
    yes.type = no.type = all.type = "button";
    ap.append(yes, no);
    if (!risky) ap.append(all);   // a risky file asks every time anyway
    el.append(ap);
    el.scrollIntoView({ block: "nearest" });
    if (mode === "code") yes.focus({ preventScroll: true });
    waitMark(true);
    let settled = false;
    return new Promise((res) => {
      const done = (v) => { ap.remove(); if (!settled) { settled = true; waitMark(false); } res(v); };
      ap.cancel = () => { if (!settled) { settled = true; waitMark(false); } ap.remove(); };
      yes.onclick = () => done(true);
      all.onclick = () => done("all");
      no.onclick = () => {
        ap.replaceChildren();
        const why = h("input"); why.type = "text"; why.placeholder = "why? (optional, the agent reads it)"; why.maxLength = 300;
        const send = h("button", null, "Reject"); send.type = "button";
        const back = h("button", null, "Cancel"); back.type = "button";
        ap.append(why, send, back);
        why.focus();
        const go = () => done({ ok: false, reason: why.value.trim() });
        send.onclick = go;
        const cancel = () => { if (!settled) { settled = true; waitMark(false); } ap.remove(); ask(mid, i, { risky }).then(res); };
        why.onkeydown = (e) => { if (e.key === "Enter") go(); else if (e.key === "Escape") { e.preventDefault(); cancel(); } };
        back.onclick = cancel;
      };
    });
  }

  // the run was stopped while the card waited: take the buttons away
  function cancelAsk(mid, i) { find(key("c", mid, i))?.querySelector(".cm-approve")?.cancel?.(); }

  // ---------------- files
  let fileClick = () => {};
  function tree(paths) {
    const t = $("code-tree");
    t.replaceChildren();
    if (!paths?.length) { t.append(h("div", "none", "no files yet")); return; }
    let prev = [];
    for (const p of paths.slice(0, 500)) {
      const parts = p.split("/");
      for (let i = 0; i < parts.length - 1; i++) {
        if (prev[i] === parts[i]) continue;
        const d = h("div", "d", parts[i] + "/"); d.style.paddingLeft = 12 + i * 14 + "px"; t.append(d);
        prev = parts.slice(0, i + 1);
      }
      prev = parts.slice(0, -1);
      const f = h("button", "f", parts[parts.length - 1]);
      f.type = "button";
      f.style.paddingLeft = 12 + (parts.length - 1) * 14 + "px";
      f.dataset.path = p;
      f.onclick = () => { t.querySelectorAll(".f.on").forEach((x) => x.classList.remove("on")); f.classList.add("on"); fileClick(p); };
      t.append(f);
    }
    if (paths.length > 500) t.append(h("div", "none", `(+${paths.length - 500} more)`));
  }
  function viewFile(text, label = null) {
    const v = $("code-view");
    v.replaceChildren();
    if (text == null) { v.append(h("span", "hint", "select a file to view it")); return; }
    if (label) v.append(h("span", "hint", label + "\n"));
    const lines = text.split("\n");
    if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
    lines.slice(0, 5000).forEach((l, i) => { v.append(h("span", "ln", String(i + 1)), l + "\n"); });
  }
  // a long proposed file, from its approval card (host only): shown in the Files tab
  function viewFull(path, text) {
    outTab("files");
    $("code-tree").querySelectorAll(".f.on").forEach((x) => x.classList.remove("on"));
    viewFile(text, `proposed ${path} (not written yet)`);
    $("code-view").scrollIntoView({ block: "nearest" });
  }
  function outTab(name) {
    for (const b of $("code-out-tabs").children) {
      b.classList.toggle("on", b.dataset.tab === name);
      b.setAttribute("aria-selected", String(b.dataset.tab === name));
      b.tabIndex = b.dataset.tab === name ? 0 : -1;
    }
    $("pv-panel").hidden = name !== "preview";
    $("files-panel").hidden = name !== "files";
  }
  $("code-out-tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-tab]"); if (b) outTab(b.dataset.tab); });

  // ---------------- preview: tabs per port, one mounted view each, the console strip
  const ports = new Map();   // port -> { tab, view (div), mount, rows: [], rev, path, state }
  let active = null, onClose = null, onReload = () => {}, onOpen = () => {};
  function portTab(port, { mount, closable }) {
    let P = ports.get(port);
    if (P) return P;
    const wrap = h("span", "pv-tabw");
    const tab = h("button", "pv-tab", ":" + port); tab.type = "button";
    tab.onclick = () => activate(port);
    wrap.append(tab);
    // its own button, so the keyboard reaches it and a click on the tab never stops the port
    if (closable) { const x = h("button", "pv-x", "×"); x.type = "button"; x.title = `stop serving :${port}`; x.setAttribute("aria-label", x.title); x.onclick = () => onClose?.(port); wrap.append(x); }
    $("pv-tabs").append(wrap);
    const view = h("div", "pv-view"); view.hidden = true;
    $("pv-frame-wrap").append(view);
    P = { tab, wrap, view, rows: [], rev: 0, path: "", state: "", mount: null };
    ports.set(port, P);
    P.mount = mount(view, P);
    $("pv-empty").hidden = true;
    $("pv-console").hidden = false;
    return P;
  }
  function dropPort(port) {
    const P = ports.get(port);
    if (!P) return;
    P.mount?.destroy(); P.wrap.remove(); P.view.remove();
    ports.delete(port);
    if (active === port) { active = null; const next = ports.keys().next().value; if (next != null) activate(next); else { $("pv-empty").hidden = false; $("pv-console").hidden = true; bar(); renderConsole(); } }
  }
  function activate(port) {
    active = port;
    for (const [p, P] of ports) { P.view.hidden = p !== port; P.tab.classList.toggle("on", p === port); }
    bar(); renderConsole();
  }
  function bar() {
    const P = ports.get(active), a = $("pv-addr");
    a.replaceChildren();
    if (P) a.append(h("span", "host", "localhost"), `:${active}/${P.path || "index.html"}`);
    else a.textContent = "no port served";
    $("pv-open").hidden = !P || !P.rev;
    $("pv-state").textContent = P ? (P.state === "ready" ? `rev ${P.rev}` : P.state === "loading" ? "loading…" : P.state === "waiting" ? "click to run" : P.state === "stopped" ? "stopped" : P.state === "hung" ? "hung" : "") : "";
  }
  function status(port, s) {
    const P = ports.get(port);
    if (!P) return;
    if (s.rev && s.rev !== P.rev && s.state === "loading") {
      P.rows.forEach((r) => (r.old = true));
      P.rows.push({ sep: `rev ${s.rev}` + (P.rev ? " · reloaded" : "") });
    }
    if (s.rev) P.rev = s.rev;
    P.path = s.path || P.path; P.state = s.state;
    if (port === active) { bar(); renderConsole(); }
  }
  function logRow(port, e) {
    const P = ports.get(port);
    if (!P) return;
    P.rows.push(e);
    if (P.rows.length > CON_MAX) P.rows.splice(0, P.rows.length - CON_MAX);
    if (e.level === "error" && port === active) conOpen(true);
    if (port === active) renderConsole();
  }
  let conTimer = 0;
  function renderConsole() {
    if (conTimer) return;
    conTimer = requestAnimationFrame(() => {
      conTimer = 0;
      const P = ports.get(active), rows = $("pv-con-rows"), stick = rows.scrollHeight - rows.scrollTop - rows.clientHeight < 30;
      rows.replaceChildren();
      const cur = (P?.rows || []).filter((r) => !r.old && !r.sep);
      const errs = cur.filter((r) => r.level === "error").length, warns = cur.filter((r) => r.level === "warn").length, logs = cur.length - errs - warns;
      const c = $("pv-counts");
      c.replaceChildren();
      if (errs) c.append(h("b", "e", plural(errs, "error")), " · ");
      if (warns) c.append(h("b", "w", plural(warns, "warning")), " · ");
      c.append(plural(logs, "log"));
      c.dataset.errors = String(errs);
      $("pv-to-agent").hidden = !host || !errs;   // only when there is something to fix
      for (const r of P?.rows || []) {
        if (r.sep) { rows.append(h("div", "pv-row rev", r.sep)); continue; }
        const row = h("div", `pv-row ${r.level}${r.old ? " old" : ""}`);
        const at = r.src ? `${r.src}${r.line ? ":" + r.line : ""}` : "";
        if (at) row.append(h("span", "src", at));
        row.append(h("span", null, r.text));
        rows.append(row);
      }
      if (stick) rows.scrollTop = rows.scrollHeight;
    });
  }
  function conOpen(open) {
    $("pv-console").classList.toggle("closed", !open);
    $("pv-con-toggle").textContent = open ? "console ▾" : "console ▸";
    $("pv-con-toggle").setAttribute("aria-expanded", String(open));
  }
  $("pv-con-toggle").onclick = () => conOpen($("pv-console").classList.contains("closed"));
  $("pv-clear").onclick = () => { const P = ports.get(active); if (P) { P.rows = []; renderConsole(); } };
  $("pv-reload").onclick = () => { if (active != null) onReload(active); };
  $("pv-open").onclick = () => { if (active != null) onOpen(active, ports.get(active)?.path || null); };

  // ---------------- host vs peer chrome
  function setHost(v) {
    host = !!v;
    $("code-project").hidden = !host;
    $("code-row").hidden = !host;
    $("code-bar").hidden = !host;
    $("code-driver").hidden = host;
    $("pv-to-agent").hidden = !host || !(+$("pv-counts").dataset.errors > 0);
  }
  setHost(false);
  viewFile(null);

  return {
    show, poke, apply, ask, cancelAsk, clear, placeholder, setHost, tree, viewFile, outTab,
    get mode() { return mode; },
    get activePort() { return active; },
    onFile(fn) { fileClick = fn; },
    onClosePort(fn) { onClose = fn; },
    onReload(fn) { onReload = fn; },
    onOpen(fn) { onOpen = fn; },
    portTab, dropPort, activate, status, logRow, ports,
    ctx(text, warn) { $("code-ctx").textContent = text || ""; $("code-ctx").classList.toggle("warn", !!warn); },
    running(on) { $("code-send").hidden = !!on; $("code-stop").hidden = !on; },
  };
}
