// Code mode (docs/design/harness-app.md E): the host's agent controller and the peers' read-only
// view. room.js imports this lazily (the Code tab, or a host's code message arriving at a peer)
// and calls initCode(roomApi, { mock }) once; it returns { show(mode) }.
//
// Host: a project (OPFS scratch folder or a picked folder, harness/projects.js), a PreviewServer
// over it, the 8 tools, and an Agent over the room's model (harness/room-model.js; with
// ?mock=code, window.__pooledMock.model instead). A run holds the room's lock for all its steps.
// Everything the timeline shows is an ai-code-* message: the host renders it, keeps it for late
// joiners and broadcasts it (visibility rules apply, room.js sendCode).
// Peer: renders those messages, and mirrors the served ports through PreviewSubscriber; each
// peer runs the preview itself, click-to-run the first time.
import { codeUI } from "./code-ui.js";
import { Agent, briefCall } from "../harness/agent.js";
import { codingTools } from "../harness/codetools.js";
import { PreviewServer } from "../harness/preview.js";
import { previewTools } from "../harness/preview-tools.js";
import { runJsTool, runJsAvailable } from "../harness/run-js.js";
import { mountPreview, openPreviewTab } from "../harness/preview-frame.js";
import { PreviewPublisher, PreviewSubscriber } from "../harness/preview-sync.js";
import { lineDiff } from "../harness/diff.js";
import { listProjects, createProject, openProject, openFolder, canOpenFolder, saveSession, loadSession } from "../harness/projects.js";
import { roomModel } from "../harness/room-model.js";
import { detectStyle } from "../harness/tools.js";
import { normPath, riskyPath } from "../harness/workspace.js";
import { CODE_SYSTEM } from "../harness/code-prompt.js";

const $ = (id) => document.getElementById(id);
const str = (v, n) => String(v ?? "").slice(0, n);
const cap = (s, n) => (s.length > n ? s.slice(0, n) + `\n…(${s.length - n} chars cut)` : s);
const HIST = 50, TOK_MS = 50, EDGE = 50;
// tools whose results are the project's own content: for a folder on disk they stay on the host
const READS = new Set(["read_file", "search", "list_dir"]);
const WORDS = new Set(("a an the me my us our please build make create write code develop implement simple small little basic new "
  + "game app application website site page web in with using for of to and that js javascript html css plain canvas").split(" "));
// "build a tetris game" -> "tetris"
export function projectName(text) {
  const keep = (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => !WORDS.has(w)).slice(0, 2);
  return keep.join(" ") || "project";
}
// the approval card's diff: lineDiff rows as [op, text] / [" ", "", skip]. The path is the one
// written (normalised). Too long to diff: the first and last EDGE lines, and (host only) the
// whole proposed file for "view full file".
export function makeDiff({ path, before, after, error }) {
  try { path = normPath(path); } catch { /* the tool reports it */ }
  const lines = after ? after.split("\n").length - (after.endsWith("\n") ? 1 : 0) : 0;
  const rows = lineDiff(before ?? "", after ?? "", { max: 400 });
  const d = { path, isNew: before == null, lines, add: 0, del: 0, rows: null };
  if (rows) {
    d.rows = rows.map((r) => (r.skip ? [" ", "", r.skip] : [r.op, r.text]));
    for (const r of rows) { if (r.op === "+") d.add++; else if (r.op === "-") d.del++; }
  } else {
    d.add = lines;
    const all = (after ?? "").split("\n");
    if (all.length && all[all.length - 1] === "") all.pop();
    d.head = all.slice(0, EDGE);
    d.tail = all.length > 2 * EDGE ? all.slice(-EDGE) : [];
    d.full = after ?? "";
  }
  if (error) d.error = error;
  return d;
}
// ai-code-tool's diff is at most ~4 KB on the wire (a long file's edges ~8 KB); never the full file
function wireDiff(d) {
  if (!d) return d;
  const { full, ...w } = d;
  if (w.head) { w.head = w.head.map((t) => t.slice(0, 160)); w.tail = w.tail.map((t) => t.slice(0, 160)); }
  if (!w.rows) return w;
  const rows = w.rows.map(([o, t, s]) => (s ? [o, "", s] : [o, t.slice(0, 160)]));
  let k = rows.length;
  while (k > 0 && JSON.stringify(rows.slice(0, k)).length > 3800) k = Math.floor(k * 0.8);
  return { ...w, rows: rows.slice(0, k), more: rows.length - k };
}

// the eval suite (tests/eval/, not deployed) runs only on a development host
const DEV_HOST = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(globalThis.location?.hostname || "");
const EVAL = DEV_HOST ? new URLSearchParams(globalThis.location?.search || "").get("eval") : null;

export async function initCode(api, { mock = null } = {}) {
  const ui = codeUI({ onMode: (m) => { if (m === "code") entered(); } });
  const isHost = () => api.role() === "host" || (!api.role() && !!api.myId() && api.myId() === api.hostId());
  let sid = crypto.getRandomValues(new Uint32Array(1))[0].toString(36);

  // ================================================================ host
  let project = null, server = null, publisher = null, tools = [], agent = null, agentSrc = null, agentStyle = null, model = null;
  let sessionJson = null, hist = [], tree = [], running = false, ctrl = null, allowTask = false;
  let userAuto = null;   // the user's own tick of "auto-approve edits", kept across projects
  let mid = "", toolN = 0;
  const callIdx = new Map();

  // render, remember for late joiners, broadcast
  function emit(msg, wire = msg) {
    ui.apply(msg);
    record(wire);
    api.broadcast(wire);
  }
  function record(m) {
    if (m.t === "ai-code-live") return;   // live typing is not history: the finished call's card is
    if (m.t === "ai-code-tok") {
      const last = hist[hist.length - 1];
      if (last?.t === m.t && last.mid === m.mid && last.step === m.step) { last.text += m.text; return; }
    } else if (m.t === "ai-code-tool") {
      const j = hist.findIndex((x) => x.t === m.t && x.mid === m.mid && x.i === m.i);
      if (j >= 0) { hist[j] = { ...hist[j], ...m }; return; }
    }
    hist.push({ ...m });
    if (hist.length > 4 * HIST) hist.splice(0, hist.length - 4 * HIST);
  }
  const note = (text, err = false) => emit({ t: "ai-code-note", mid, text, ...(err ? { err: true } : {}) });
  // model text, coalesced so a peer gets ~20 messages a second, not one per token
  let tokBuf = "", tokStep = 0, tokTimer = 0;
  function flushTok() {
    clearTimeout(tokTimer); tokTimer = 0;
    if (tokBuf) { const text = tokBuf; tokBuf = ""; emit({ t: "ai-code-tok", mid, step: tokStep, text }); }
  }
  function tok(step, text) {
    if (step !== tokStep) flushTok();
    tokStep = step; tokBuf += text;
    tokTimer ||= setTimeout(flushTok, TOK_MS);
  }
  // the tool call the model is typing, sent as deltas (a new call resets), at most every TOK_MS
  let liveRaw = null, liveSent = 0, liveN = 0, liveStep = 0, liveTimer = 0;
  function flushLive() {
    clearTimeout(liveTimer); liveTimer = 0;
    if (liveRaw == null || liveRaw.length <= liveSent) return;
    emit({ t: "ai-code-live", mid, step: liveStep, n: liveN, reset: liveSent === 0, text: liveRaw.slice(liveSent) });
    liveSent = liveRaw.length;
  }
  function live(step, raw) {
    if (raw == null) {
      if (liveRaw != null) { flushLive(); emit({ t: "ai-code-live", mid, step: liveStep, n: liveN, end: true }); }
      liveRaw = null; liveSent = 0; return;
    }
    if (liveRaw == null || raw.length < liveSent || step !== liveStep) { liveN++; liveSent = 0; }
    liveRaw = raw; liveStep = step;
    liveTimer ||= setTimeout(flushLive, TOK_MS);
  }
  const tool = (i, fields, wire) => {
    const base = { t: "ai-code-tool", mid, i, ...fields };
    emit(base, wire ? { ...base, ...wire } : base);
  };

  async function sendFiles() {
    if (!project) return;
    try { tree = (await project.ws.walk(500)).slice(0, 500); } catch { tree = []; }
    ui.tree(tree);
    api.broadcast({ t: "ai-code-files", tree });
  }

  // ---- projects
  async function refreshProjects() {
    const sel = $("code-proj-select"), list = await listProjects().catch(() => []);
    sel.replaceChildren(new Option(list.length ? "open a project…" : "no projects yet", ""));
    for (const p of list) sel.add(new Option(p.name + (p.kind === "folder" ? " (folder)" : ""), p.id));
    sel.value = project?.id || "";
    $("code-proj-kind").textContent = project ? (project.kind === "folder" ? "folder on disk · edits ask first" : "saved in this browser") : "";
    $("code-newtask").disabled = !project;
  }
  function closeProject() {
    if (running) ctrl?.abort();
    publisher?.close(); server?.close();
    for (const port of [...ui.ports.keys()]) ui.dropPort(port);
    project = server = publisher = agent = model = null; agentSrc = null; tools = [];
  }
  // keepAuto: the project run() made for the first request keeps the box as the user left it
  async function useProject(p, { keepAuto = false } = {}) {
    if (!p) return;
    closeProject();
    project = p;
    server = new PreviewServer(p.ws);
    // run_js only when the snippet runs on the isolated preview host (a loop there cannot freeze the room)
    tools = [...codingTools(p.ws, { server }), ...previewTools(server), ...(runJsAvailable() ? [runJsTool(server)] : [])];
    publisher = new PreviewPublisher(server, { send: api.send, broadcast: api.broadcast, channel: api.channel });
    server.onUpdate(portUpdate);
    const saved = await loadSession(p.id).catch(() => null);
    sessionJson = saved?.agent || null;
    hist = Array.isArray(saved?.hist) ? saved.hist : [];
    sid = crypto.getRandomValues(new Uint32Array(1))[0].toString(36);
    // scratch files only live here (on unless the user turned it off); a real folder asks per edit
    if (!keepAuto || p.kind === "folder") $("code-auto").checked = p.kind === "opfs" && userAuto !== false;
    ui.clear();
    for (const m of hist) ui.apply(m);
    if (!hist.length) ui.placeholder(`project <b>${escapeHTML(p.name)}</b> is empty<br>ask for something to build`);
    await sendFiles();
    api.broadcast({ t: "ai-code-history", sid, items: hist.slice(-HIST), tree });
    refreshProjects();
    ctxMeter();
  }
  const escapeHTML = (s) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  function portUpdate(u) {
    if (u.stopped) { ui.dropPort(u.port); return; }
    if (ui.ports.has(u.port)) return;   // the mounted frame follows its own updates
    const port = u.port, src = server;
    ui.portTab(port, { closable: true, mount: (el) => mountPreview(el, src, port, { onLog: (e) => ui.logRow(port, e), onStatus: (s) => ui.status(port, s) }) });
    ui.activate(port);
    ui.outTab("preview");
  }
  ui.onClosePort((port) => { if (isHost()) server?.stop(port); else ui.dropPort(port); });
  ui.onReload(async (port) => {
    const P = ui.ports.get(port);
    if (isHost() && server) { const s = await server.refresh(port); if (!s) P?.mount.reload(); }
    else P?.mount.reload();
  });
  ui.onOpen((port, path) => { openPreviewTab(isHost() ? server : sub, port, path); });
  ui.onFile(async (path) => {
    if (isHost() && project) {
      try { ui.viewFile(cap(await project.ws.read(path), 200000)); } catch (e) { ui.viewFile(`(${e.message})`); }
      return;
    }
    // a peer has the content of served files only
    for (const s of (sub?.ports() || []).map((p) => sub.snapshot(p.port))) {
      const rel = s.dir ? (path.startsWith(s.dir + "/") ? path.slice(s.dir.length + 1) : null) : path;
      const f = rel != null && s.files.get(rel);
      if (f) { ui.viewFile(cap(new TextDecoder().decode(f.bytes), 200000)); return; }
    }
    ui.viewFile("(the file stays on the host; files of a served preview show here)");
  });

  $("code-proj-select").addEventListener("change", async (e) => {
    const id = e.target.value;
    if (!id || id === project?.id) return;
    if (running) { e.target.value = project?.id || ""; return; }
    try { await useProject(await openProject(id)); } catch (err) { localNote(err.message, true); refreshProjects(); }
  });
  const newName = $("code-new-name");
  $("code-new").addEventListener("click", () => {
    if (running) return;
    const on = newName.hidden;
    newName.hidden = !on; $("code-proj-select").hidden = on;
    if (on) { newName.value = ""; newName.focus(); }
  });
  newName.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") { newName.hidden = true; $("code-proj-select").hidden = false; return; }
    if (e.key !== "Enter" || !newName.value.trim()) return;
    const name = newName.value.trim();
    newName.hidden = true; $("code-proj-select").hidden = false;
    try { await useProject(await createProject(name)); } catch (err) { localNote("could not create the project: " + err.message, true); }
  });
  $("code-open").hidden = !canOpenFolder();
  $("code-open").addEventListener("click", async () => {
    if (running) return;
    try { const p = await openFolder(); if (p) await useProject(p); } catch (err) { localNote("could not open the folder: " + err.message, true); }
  });
  $("code-newtask").addEventListener("click", () => {
    if (running || !project) return;
    agent?.reset(); sessionJson = null;
    mid = "";
    note("new task: the agent starts fresh · files and previews stay");
    save();
    ctxMeter();
  });
  $("code-auto").addEventListener("change", (e) => { if (project?.kind !== "folder") userAuto = e.target.checked; });
  $("pv-to-agent").addEventListener("click", () => { $("code-prompt").value = "Fix the errors in the preview console"; grow(); $("code-prompt").focus(); });

  // a line only this screen sees (not part of the session)
  function localNote(text, err = false) { ui.apply({ t: "ai-code-note", text, err }); }
  function save() {
    if (!project) return;
    saveSession(project.id, { v: 1, agent: agent ? agent.toJSON() : sessionJson, hist: hist.slice(-4 * HIST) }).catch((e) => console.warn("code session not saved", e));
  }
  function ctxMeter() {
    if (!agent || !model?.count) { ui.ctx(""); return; }   // a scripted model has no token count to show
    const max = api.maxSeq(), last = model?.stats?.last;
    let used;
    try { used = last ? last.prompt + last.generated : agent._size(); } catch { used = 0; }
    ui.ctx(used ? `context ${used} / ${max}` : "", used > max * 0.8);
  }

  // ---- the agent
  function ensureAgent() {
    // rebuilt when the model's tool format changes too (a re-deal to another model)
    const style = mock?.model ? "xml" : detectStyle(api.chatTemplate());
    const src = mock?.model || "room";
    if (agent && agentSrc === src && agentStyle === style) return;
    model = mock?.model ? (typeof mock.model === "function" ? { generate: mock.model } : mock.model) : roomModel(api, { tools, style, maxNew: 8192 });
    const json = agent ? agent.toJSON() : sessionJson;
    agent = Agent.from(json, {
      generate: model.generate, tools, style, system: CODE_SYSTEM, maxSteps: 30, approve, onEvent,
      budget: model.budget || Infinity, count: model.count || null,
      usage: model.stats ? () => model.stats.last : null, idsFor: model.idsFor || null, adopt: model.adopt || null, idsTag: model.idsTag || null,
    });
    agentSrc = src; agentStyle = style;
  }
  async function approve(call, info) {
    const i = callIdx.get(call);
    const diff = info ? makeDiff(info) : null;
    // a file of a folder on disk that can run commands (package.json, a script, a dotfile) always
    // asks, whatever auto-approve and "Allow edits for this task" say
    if (diff && project?.kind === "folder" && riskyPath(diff.path)) diff.risky = true;
    // a failing edit is not worth a question (run() returns the error to the model), nor is a
    // write that changes nothing
    const same = diff && !diff.error && diff.rows && !diff.isNew && !diff.add && !diff.del;
    const auto = !diff?.risky && ($("code-auto").checked || allowTask || info?.error || same);
    tool(i, { state: auto ? "approved" : "pending", diff }, { diff: wireDiff(diff) });
    if (auto) return true;
    let off = null;
    const stopped = new Promise((r) => { const f = () => r({ ok: false, reason: "stopped" }); ctrl.signal.addEventListener("abort", f, { once: true }); off = () => ctrl?.signal.removeEventListener("abort", f); });
    const v = await Promise.race([ui.ask(mid, i, { risky: !!diff?.risky }), stopped]);
    off?.(); ui.cancelAsk(mid, i);
    if (v === "all") allowTask = true;
    const ok = v === true || v === "all";
    tool(i, { state: ok ? "approved" : v?.reason === "stopped" ? "stopped" : "declined" });
    return ok || v;
  }
  function onEvent(e) {
    switch (e.type) {
      case "text": tok(e.step, e.text); break;
      case "call-live": live(e.step, e.raw); break;
      case "tool-start": {
        flushTok(); live(e.step, null);
        const i = ++toolN;
        callIdx.set(e.call, i);
        tool(i, { step: e.step, name: str(e.call.name || /<function=([^>\s]+)>/.exec(e.call.raw || "")?.[1] || "tool call", 60), brief: str(briefCall(e.call), 200), state: "running" });
        break;
      }
      case "tool": {
        const i = callIdx.get(e.call), r = String(e.result);
        const state = r === "declined by the user: stopped" ? "stopped" : /^declined by the user/.test(r) ? "declined" : /^error/.test(r) ? "error" : "done";
        // a folder on disk: what the agent read stays on the host (peers see its size only)
        const wire = project?.kind === "folder" && READS.has(e.call.name) && state === "done"
          ? `(${r.split("\n").length} lines · a folder on disk: the output stays on the host)` : cap(r, 600);
        tool(i, { state, result: cap(r, 4000), ms: e.ms }, { result: wire });
        if (state === "done" && ["write_file", "edit_file"].includes(e.call.name)) sendFiles();
        ctxMeter();
        break;
      }
      case "compacted":
        if (e.tier < 4) note(`older steps shortened to fit the context (${e.before} → ${e.after} tokens)`);
        break;
      case "usage": ctxMeter(); break;
      case "limit": note(`stopped after ${e.steps} steps`); break;
      case "stuck": note("stopped: the same tool call failed three times in a row. If other devices are in the room, the split model may be producing bad output: try the same request on one device, or re-deal", true); break;
    }
  }
  function stats(r, t0, gen0) {
    const s = model?.stats, gen = s ? s.generated - gen0 : 0, secs = ((Date.now() - t0) / 1000).toFixed(1);
    const parts = [`${r.steps} step${r.steps === 1 ? "" : "s"}`, `${r.calls} tool call${r.calls === 1 ? "" : "s"}`];
    if (gen) parts.push(`${gen} tok`, `${(s.tps || 0).toFixed(1)} tok/s`);
    parts.push(`${api.peers().length + 1} device${api.peers().length ? "s" : ""}`, `${secs} s`);
    if (r.reason !== "done") parts.push(r.reason);
    return parts.join(" · ");
  }
  async function run() {
    const box = $("code-prompt"), text = box.value.trim();
    if (!text || running || !isHost()) return;
    if (!api.ready()) { localNote("the model is not loaded yet: pick a model in Chat and press Start", true); return; }
    if (EVAL != null && /^\/eval\b/.test(text)) { box.value = ""; grow(); return runEval(text.slice(5).trim() || EVAL); }
    running = true;
    ui.running(true);
    try {
      if (!project) await useProject(await createProject(projectName(text)), { keepAuto: true });
      if (!api.lock("code")) {
        localNote(api.busy() ? "the room is answering a chat question: send again when it is done" : "the room cannot run the agent right now (a device left? re-deal first)", true);
        return;
      }
      box.value = ""; grow();
      ensureAgent();
      mid = "m" + Date.now().toString(36);
      toolN = 0; callIdx.clear(); allowTask = false;
      ctrl = new AbortController();
      emit({ t: "ai-code-start", sid, mid, name: api.name?.() || "host", text: str(text, 4000) });
      const t0 = Date.now(), gen0 = model?.stats?.generated || 0;
      let r;
      try { r = await agent.run(text, { signal: ctrl.signal }); }
      catch (err) { console.error(err); r = { steps: 0, calls: 0, reason: "error" }; note("error: " + err.message, true); }
      finally { flushTok(); api.unlock(); }
      if (r.reason === "stopped") note("stopped");
      if (r.reason === "context") note(r.text, true);
      emit({ t: "ai-code-done", mid, steps: r.steps, reason: r.reason, stats: stats(r, t0, gen0) });
      save();
      ctxMeter();
    } catch (err) {
      localNote(err.message, true);
    } finally {
      running = false; ctrl = null;
      ui.running(false);
    }
  }
  // ?eval=all (or ?eval=tetris,todo): "/eval [ids]" in the prompt runs the eval suite
  // (tests/eval/) on the room's model, each task in a fresh in-memory project; the records and
  // trajectories download as .jsonl at the end
  async function runEval(spec) {
    if (!api.lock("code")) { localNote("the room is busy: try again when it is done", true); return; }
    running = true; ui.running(true); ctrl = new AbortController();
    const lines = [];
    try {
      let TASKS, byId, S;
      try { ({ TASKS, byId } = await import("../tests/eval/tasks/index.js")); S = await import("../tests/eval/suite.js"); }
      catch { throw new Error("the eval suite is not deployed here (run it from a local checkout)"); }
      const tasks = !spec || spec === "all" ? TASKS : spec.split(",").map((id) => byId(id.trim())).filter(Boolean);
      const style = detectStyle(api.chatTemplate());
      localNote(`eval: ${tasks.length} task${tasks.length === 1 ? "" : "s"} on ${api.peers().length + 1} device(s)`);
      const recs = await S.runSuite(tasks, {
        model: "room", signal: ctrl.signal, root: document.body,
        makeModel: ({ tools }) => ({ ...roomModel(api, { tools, style, maxNew: 8192 }), style }),
        onResult: ({ rec, trajectory }) => {
          lines.push(JSON.stringify(rec), JSON.stringify({ trajectory }));
          localNote(`${rec.ok ? "PASS" : "FAIL"} ${rec.id} · ${rec.reason} · ${rec.steps} steps · ${rec.generated} tok · ${(rec.ms / 1000).toFixed(0)} s`, !rec.ok);
        },
      });
      localNote(S.summary(recs));
    } catch (err) { localNote("eval failed: " + err.message, true); }
    finally {
      api.unlock(); running = false; ctrl = null; ui.running(false);
      if (lines.length) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" }));
        a.download = `eval-room-${new Date().toISOString().slice(0, 16).replace(/:/g, "")}.jsonl`;
        a.click();
      }
    }
  }
  api.onStop(() => ctrl?.abort());
  $("code-send").addEventListener("click", run);
  $("code-stop").addEventListener("click", () => api.stop());
  const grow = () => { const p = $("code-prompt"); p.style.height = "auto"; p.style.height = Math.min(p.scrollHeight, 160) + "px"; };
  $("code-prompt").addEventListener("input", grow);
  $("code-prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !matchMedia("(pointer: coarse)").matches) { e.preventDefault(); run(); }
  });
  // Esc stops the run, but not from a field or the approval buttons (Esc there backs out of them)
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !running || ui.mode !== "code" || e.defaultPrevented) return;
    if (e.target?.closest?.("input, textarea, select, .cm-approve")) return;
    api.stop();
  });
  api.on("ai-pv-want", (from, d) => { if (isHost()) publisher?.onWant(from, d); });
  api.onPeerJoin((id) => {
    if (!isHost() || (!hist.length && !server?.ports().length)) return;
    api.send(id, { t: "ai-code-history", sid, items: hist.slice(-HIST), tree });
    publisher?.helloTo(id);
  });

  // ================================================================ peer
  let sub = null;
  function peerView() {
    if (sub) return sub;
    sub = new PreviewSubscriber({ send: (m) => api.send(api.hostId(), m), hostId: () => api.hostId() });
    sub.onUpdate((u) => {
      if (u.stopped) { ui.dropPort(u.port); return; }
      if (ui.ports.has(u.port)) return;
      const port = u.port;
      ui.portTab(port, { closable: false, mount: (el) => mountPreview(el, sub, port, { autorun: false, onLog: (e) => ui.logRow(port, e), onStatus: (s) => ui.status(port, s) }) });
      if (ui.activePort == null) ui.activate(port);
    });
    return sub;
  }
  // peers take the host's messages as untrusted data: typed and capped here, textContent in the UI
  function clean(d) {
    const o = { t: d.t, mid: str(d.mid, 40) };
    if ("step" in d) o.step = d.step >>> 0;
    if ("i" in d) o.i = d.i >>> 0;
    for (const k of ["sid", "name", "text", "brief", "state", "result", "stats", "reason"]) if (k in d) o[k] = str(d[k], k === "text" ? 8000 : k === "result" ? 4000 : 400);
    if ("ms" in d) o.ms = d.ms >>> 0;
    if ("steps" in d) o.steps = d.steps >>> 0;
    if (d.err) o.err = true;
    if ("n" in d) o.n = d.n >>> 0;
    if (d.reset) o.reset = true;
    if (d.end) o.end = true;
    if (d.diff && typeof d.diff === "object") {
      const x = d.diff;
      const lines = (a) => (Array.isArray(a) ? a.slice(0, 50).map((t) => str(t, 200)) : null);
      o.diff = { path: str(x.path, 300), isNew: !!x.isNew, lines: x.lines >>> 0, add: x.add >>> 0, del: x.del >>> 0, more: x.more >>> 0,
        rows: Array.isArray(x.rows) ? x.rows.slice(0, 400).map((r) => [str(r?.[0], 1), str(r?.[1], 200), r?.[2] >>> 0]) : null,
        head: lines(x.head), tail: lines(x.tail), risky: !!x.risky };
    }
    return o;
  }
  const peerMsg = (d) => {
    if (isHost()) return;
    peerView();
    ui.setHost(false);
    ui.poke();
    const m = clean(d);
    if (m.t === "ai-code-start" && m.sid && m.sid !== sid) { sid = m.sid; }
    ui.apply(m);
  };
  for (const t of ["ai-code-start", "ai-code-tok", "ai-code-live", "ai-code-tool", "ai-code-note", "ai-code-done"]) api.on(t, (from, d) => peerMsg(d));
  api.on("ai-code-files", (from, d) => { if (!isHost() && Array.isArray(d.tree)) ui.tree(d.tree.slice(0, 500).map((p) => str(p, 300))); });
  api.on("ai-code-history", (from, d) => {
    if (isHost()) return;
    peerView(); ui.setHost(false); ui.poke();
    sid = str(d.sid, 40);
    ui.clear();
    for (const it of (Array.isArray(d.items) ? d.items : []).slice(-HIST)) if (it && typeof it === "object") ui.apply(clean(it));
    if (Array.isArray(d.tree)) ui.tree(d.tree.slice(0, 500).map((p) => str(p, 300)));
  });
  api.on("ai-pv", (from, d) => { if (!isHost()) { peerView().onManifest(from, d); ui.poke(); } });
  api.on("ai-pv-blob", (from, d) => { if (!isHost()) peerView().onBlob(from, d); });
  api.on("ai-pv-stop", (from, d) => { if (!isHost()) peerView().onStop(from, d); });

  // ================================================================ both
  function entered() {
    const host = isHost();
    ui.setHost(host);
    if (host) {
      refreshProjects();
      if (!project && !hist.length) {
        ui.placeholder(api.ready()
          ? "<b>Code mode</b>: the room's model writes a web app, serves it on a port and fixes its own errors.<br>Try “build a tetris game”."
          : "<b>Code mode</b> runs on the room's model.<br>Pick a model in Chat and press Start, then ask for something to build.");
      }
      setTimeout(() => $("code-prompt").focus(), 0);
    } else if (!$("code-log").children.length) ui.placeholder("the host hasn't started the agent yet<br>what it does shows up here, live");
  }
  api.onRole(() => {
    const host = isHost();
    ui.setHost(host);
    if (!host && running) ctrl?.abort();
    // a device left mid-run: the next step would wait out the lap timeouts, so stop here
    else if (running && !api.ready() && !ctrl?.signal.aborted) { note("a device left: stopped · re-deal the layers, then send again", true); ctrl?.abort(); }
  });
  ui.setHost(isHost());
  return { show: (m) => ui.show(m) };
}
