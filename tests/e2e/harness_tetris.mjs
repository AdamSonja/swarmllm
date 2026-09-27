// Code mode end to end (docs/design/harness-app.md G.2, the acceptance test), with a scripted
// model instead of a GPU: two headless Chromium tabs in a real room (local PeerServer, real
// WebRTC, no WebGPU), the host in Code mode, the agent driven by tests/scripted-model.js.
//
//   "build a tetris game" -> index.html, style.css, game.js written (game.js with a TDZ bug)
//   -> serve :5173 -> the preview reports "ReferenceError ... game.js" back to the agent
//   -> edit_file fixes it -> preview_logs shows the new rev clean -> the answer
//   -> the peer tab sees the timeline, runs its own sandboxed copy of the preview at the same rev.
//
//   NODE_PATH=<dir with peer + peerjs + playwright> node tests/e2e/harness_tetris.mjs [--shots] [--headed]
//   --shots: saves docs/design/shots/code-desktop.png, code-400.png, code-preview.png, code-peer.png
//   --port 18990 --signal-port 9011
// Prerequisites: playwright, the `peer` server package and the PeerJS client bundle (`peerjs`)
// importable from NODE_PATH or ./node_modules. Without the PeerJS bundle it prints SKIP and
// exits 0 (tests/e2e/preview_browser.mjs still covers the preview without a room).
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const PORT = +arg("port", 18990), SIGNAL_PORT = +arg("signal-port", 9011);
const PEERJS_URL = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js";
const SHOTS = path.join(ROOT, "docs/design/shots");
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1).padStart(6) + "s", ...a);

function resolvePkg(name) {
  const dirs = [...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean), path.join(ROOT, "node_modules")];
  for (const d of dirs) for (const base of [d, path.join(d, "node_modules")]) if (fs.existsSync(path.join(base, name, "package.json"))) return path.join(base, name);
  return null;
}
const peerjsDir = resolvePkg("peerjs"), peerDir = resolvePkg("peer");
if (!peerjsDir || !fs.existsSync(path.join(peerjsDir, "dist/peerjs.min.js"))) { console.log("SKIP: peerjs client missing (npm i --no-save peerjs@1.5.4, or put it on NODE_PATH)"); process.exit(0); }
if (!peerDir) { console.log("SKIP: the `peer` server package is missing"); process.exit(0); }
const peerjsJs = fs.readFileSync(path.join(peerjsDir, "dist/peerjs.min.js"));

// ---------------------------------------------------------------- the scripted model (runs in the host page)
async function installModel() {
  const { scripted, xmlCall } = await import("/tests/scripted-model.js");
  const checks = window.__checks = [];
  const check = (name, ok, detail = "") => checks.push({ name, ok: !!ok, detail: String(detail).slice(0, 600) });
  const lastResult = (req) => req.turns[req.turns.length - 1].text;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tetris</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <h1>TETRIS</h1>
  <canvas id="board" width="240" height="480"></canvas>
  <p class="help">← → move · ↑ rotate · space drops</p>
  <script type="module" src="game.js"></script>
</body>
</html>
`;
  const css = `body { margin: 0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; background: #14161f; color: #e8e6df; font: 14px/1.4 ui-monospace, monospace; }
h1 { margin: 0; font-size: 18px; letter-spacing: 0.4em; }
canvas { border: 2px solid #3a3f55; border-radius: 6px; max-height: 70vh; }
.help { margin: 0; color: #8b877a; font-size: 12px; }
`;
  const bug = `draw();   // first frame
const ctx = canvas.getContext("2d");`;
  const fix = `const ctx = canvas.getContext("2d");
draw();   // first frame`;
  const game = `// Tetris on a canvas: arrow keys move, up rotates, space drops.
const COLS = 10, ROWS = 20, S = 24;
const canvas = document.getElementById("board");
const COLORS = ["#000", "#2b4eff", "#f2b134", "#1a9e5c", "#d1242f", "#8a5cf6", "#18a0b8", "#f06b2b"];
const SHAPES = [
  [[1, 1, 1, 1]],
  [[2, 2], [2, 2]],
  [[0, 3, 0], [3, 3, 3]],
  [[4, 4, 0], [0, 4, 4]],
  [[0, 5, 5], [5, 5, 0]],
  [[6, 0, 0], [6, 6, 6]],
  [[0, 0, 7], [7, 7, 7]],
];
const board = Array.from({ length: ROWS }, (_, y) => Array.from({ length: COLS }, (_, x) => (y > 16 && (x * 7 + y * 3) % 5 ? 1 + ((x + y) % 7) : 0)));
let piece = spawn(), score = 0, over = false;

function spawn() {
  const m = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  return { m, x: Math.floor((COLS - m[0].length) / 2), y: 0 };
}
function collide(m, x, y) {
  return m.some((row, j) => row.some((v, i) => v && (y + j >= ROWS || x + i < 0 || x + i >= COLS || board[y + j]?.[x + i])));
}
function rotate(m) { return m[0].map((_, i) => m.map((row) => row[i]).reverse()); }
function merge() {
  piece.m.forEach((row, j) => row.forEach((v, i) => { if (v) board[piece.y + j][piece.x + i] = v; }));
  for (let y = ROWS - 1; y >= 0; y--) {
    if (board[y].every(Boolean)) { board.splice(y, 1); board.unshift(Array(COLS).fill(0)); score += 100; y++; }
  }
  piece = spawn();
  if (collide(piece.m, piece.x, piece.y)) over = true;
}
function cell(x, y, v) {
  ctx.fillStyle = COLORS[v];
  ctx.fillRect(x * S + 1, y * S + 1, S - 2, S - 2);
}
function draw() {
  ctx.fillStyle = "#0b0c12";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  board.forEach((row, y) => row.forEach((v, x) => v && cell(x, y, v)));
  piece.m.forEach((row, j) => row.forEach((v, i) => v && cell(piece.x + i, piece.y + j, v)));
  ctx.fillStyle = "#e8e6df";
  ctx.font = "14px monospace";
  ctx.fillText(over ? "GAME OVER" : "score " + score, 8, 18);
}
${bug}

function tick() {
  if (over) return;
  if (collide(piece.m, piece.x, piece.y + 1)) merge(); else piece.y++;
  draw();
}
document.addEventListener("keydown", (e) => {
  const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowDown: [0, 1] };
  if (moves[e.key] && !collide(piece.m, piece.x + moves[e.key][0], piece.y + moves[e.key][1])) { piece.x += moves[e.key][0]; piece.y += moves[e.key][1]; }
  if (e.key === "ArrowUp") { const r = rotate(piece.m); if (!collide(r, piece.x, piece.y)) piece.m = r; }
  if (e.key === " ") { while (!collide(piece.m, piece.x, piece.y + 1)) piece.y++; merge(); }
  draw();
});
setInterval(tick, 500);
console.log("tetris ready");
`;
  window.__pooledMock.model = scripted([
    "I'll build it as three files: the page, its style and the game.\n" + xmlCall("write_file", { path: "index.html", content: html }),
    xmlCall("write_file", { path: "style.css", content: css }),
    xmlCall("write_file", { path: "game.js", content: game }),
    (req) => { check("write results reach the model", /wrote game\.js \(\d+ lines/.test(lastResult(req)), lastResult(req)); return "Now serve it.\n" + xmlCall("serve", { port: 5173 }); },
    (req) => {
      const r = lastResult(req);
      window.__serveResult = r;
      check("serve result names the ReferenceError in game.js", /ReferenceError/.test(r) && /game\.js:\d+/.test(r), r);
      return "The preview shows `ctx` used before it is declared. Moving the line.\n" + xmlCall("edit_file", { path: "game.js", old: bug, new: fix });
    },
    async (req) => {
      const r = lastResult(req);
      check("edit result says the preview reloaded", /^<tool_response>\nedited game\.js .*preview :5173 reloaded/.test(r), r);
      await sleep(1500);   // a real model takes longer than the reload
      return xmlCall("preview_logs", { port: 5173, since: 1 });
    },
    (req) => {
      const r = lastResult(req);
      window.__logsResult = r;
      const cur = r.split(/\(rev \d+, current\)/)[1] ?? r;
      check("preview_logs: the current rev ran clean", /tetris ready/.test(cur) && !/\] error /.test(cur), r);
      return "Tetris is running on :5173.";
    },
  ], { piece: 12 });
}

// ---------------------------------------------------------------- main
const srv = serveRepo(PORT, {});
const peerServer = spawn(process.execPath, [path.join(peerDir, "dist/bin/peerjs.js"), "--port", String(SIGNAL_PORT), "--host", "127.0.0.1", "--path", "/"], { stdio: ["ignore", "ignore", "pipe"] });
let peerErr = "";
peerServer.stderr.on("data", (d) => { peerErr += d; });
for (let i = 0; ; i++) {
  if (await fetch(`http://127.0.0.1:${SIGNAL_PORT}/peerjs/id`).then((r) => r.ok, () => false)) break;
  if (i > 50 || peerServer.exitCode !== null) { console.error("PeerServer did not start:", peerErr.slice(0, 400)); process.exit(2); }
  await new Promise((r) => setTimeout(r, 200));
}
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), headless: !flag("headed"),
  args: ["--no-sandbox", "--allow-loopback-in-peer-connection", "--disable-features=WebRtcHideLocalIpsWithMdns"] });
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok: !!ok }); console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail && !ok ? "  " + String(detail).slice(0, 600) : ""}`); };
let code = 1;
try {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.route("**/*", (route) => {
    const url = route.request().url();
    if (url.startsWith(`http://127.0.0.1:${PORT}/`) || url.startsWith(`http://127.0.0.1:${SIGNAL_PORT}/`)) return route.continue();
    if (url.split("?")[0] === PEERJS_URL) return route.fulfill({ status: 200, contentType: "text/javascript", body: peerjsJs });
    if (url.startsWith("https://fonts.googleapis.com/")) return route.fulfill({ status: 200, contentType: "text/css", body: "" });
    return route.abort();
  });
  const host = await ctx.newPage(), peer = await ctx.newPage();
  const errs = { host: [], peer: [] };
  // the app's own bug (rev 1 of game.js) is expected, and is reported by the preview, not the room
  const expected = /Cannot access 'ctx' before initialization|net::ERR_FAILED|Failed to load resource/;
  for (const [n, p] of Object.entries({ host, peer })) {
    p.on("console", (m) => { if (m.type() === "error" && !expected.test(m.text()) && !/Could not connect to peer/.test(m.text())) errs[n].push(m.text().slice(0, 300)); });
    p.on("pageerror", (e) => { if (!expected.test(String(e))) errs[n].push("pageerror: " + String(e).slice(0, 300)); });
  }
  const base = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}`;
  await host.goto(base + "&mock=code");
  await peer.goto(base);
  for (const [p, n] of [[host, "host"], [peer, "peer"]]) {
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 30000 });
    await p.fill("#name-input", n + "-e2e");
  }
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
  const room = (await host.textContent("#room-badge")).trim();
  await peer.fill("#code-input", room);
  await peer.click("#join-btn");
  for (const p of [host, peer]) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 30000 });
  log(`room ${room}: host + peer`);

  // ---- host: Code mode, a new project, the scripted model
  check("host sees the Code tab", await host.isVisible("#mode-code"));
  check("peer has no Code tab before the host starts", !(await peer.isVisible("#mode-code")));
  await host.click("#mode-code");
  await host.waitForSelector("#code-project:not([hidden])", { timeout: 15000 });
  check("chat is hidden in Code mode", !(await host.isVisible("#ai-row")) && await host.isVisible("#code-log"));
  await host.evaluate(installModel);
  await host.click("#code-new");
  await host.fill("#code-new-name", "tetris");
  await host.press("#code-new-name", "Enter");
  await host.waitForFunction(() => document.getElementById("code-proj-select").value === "opfs:tetris", null, { timeout: 10000 });
  check("New project 'tetris' is an OPFS project, auto-approve on", await host.isChecked("#code-auto"));
  await host.uncheck("#code-auto");   // exercise the approval card once
  await host.fill("#code-prompt", "build a tetris game");
  await host.click("#code-send");
  // the first edit asks; "Allow edits for this task" approves the rest
  await host.waitForSelector(".cm-approve button", { timeout: 15000 }).catch(async (e) => { console.error("timeline:", await host.evaluate(() => document.getElementById("code-log").innerText), errs); throw e; });
  await peer.waitForFunction(() => /waiting for the host's approval/.test(document.getElementById("code-log")?.textContent || ""), null, { timeout: 10000 })
    .then(() => check("peer sees the pending approval", true), () => check("peer sees the pending approval", false));
  check("the approval card shows the new file's diff", await host.evaluate(() => { const d = document.querySelector(".cm-tool .cm-diff"); return !!d && /index\.html/.test(d.textContent) && d.querySelectorAll(".r-add").length > 5; }));
  await host.click("text=Allow edits for this task");
  await host.waitForSelector(".cm-stats", { timeout: 60000 });
  log("agent run finished");

  // ---- host asserts
  const H = await host.evaluate(() => ({
    checks: window.__checks,
    tabs: [...document.querySelectorAll(".pv-tab")].map((t) => t.textContent),
    tools: [...document.querySelectorAll("#code-log .cm-tool")].map((t) => t.querySelector(".nm").textContent + ":" + t.querySelector(".chip").textContent),
    stats: document.querySelector(".cm-stats")?.textContent,
    answer: [...document.querySelectorAll("#code-log .cm-text")].pop()?.textContent,
    oldErrors: document.querySelectorAll("#pv-con-rows .pv-row.error.old").length,
    curErrors: document.getElementById("pv-counts").dataset.errors,
    state: document.getElementById("pv-state").textContent,
    tree: [...document.querySelectorAll("#code-tree .f")].map((f) => f.dataset.path),
  }));
  for (const c of H.checks) check("model saw: " + c.name, c.ok, c.detail);
  check("7 scripted replies used", H.checks.length === 4, JSON.stringify(H.checks.map((c) => c.name)));
  check(".pv-tab :5173", H.tabs.some((t) => t.startsWith(":5173")), H.tabs);
  check("timeline: 6 tool cards, all done", H.tools.join(" ") === "write_file:done write_file:done write_file:done serve:done edit_file:done preview_logs:done", H.tools.join(" "));
  check("final answer shown", /Tetris is running on :5173/.test(H.answer || ""), H.answer);
  check(".cm-stats present", /7 steps · 6 tool calls/.test(H.stats || ""), H.stats);
  check("console showed the error, then a new rev with none", H.oldErrors >= 1 && H.curErrors === "0", JSON.stringify(H));
  check("file tree lists the 3 files", H.tree.join(",") === "game.js,index.html,style.css", H.tree);
  const hostFrame = host.frames().find((f) => f !== host.mainFrame());
  const hostPx = await hostFrame.evaluate(() => {
    const c = document.getElementById("board");
    const g = c.getContext("2d"), d = g.getImageData(0, 0, c.width, c.height).data;
    let drawn = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i]) drawn++;
    return { w: c.width, h: c.height, drawn, origin: window.origin };
  });
  check("host preview: canvas has size and drawn pixels", hostPx.w > 0 && hostPx.drawn > 1000, JSON.stringify(hostPx));
  check("host preview runs in an opaque origin", hostPx.origin === "null", hostPx.origin);
  const hostRev = +(/rev (\d+)/.exec(H.state)?.[1] || 0);

  // ---- peer asserts
  await peer.waitForSelector("#mode-code", { state: "visible", timeout: 10000 });
  check("peer: the Code tab appeared with a dot", await peer.evaluate(() => document.getElementById("mode-code").classList.contains("fresh")));
  await peer.click("#mode-code");
  await peer.waitForFunction(() => document.querySelectorAll("#code-log .cm-tool").length >= 6 && document.querySelector("#code-log .cm-stats"), null, { timeout: 15000 });
  const P = await peer.evaluate(() => ({
    tools: [...document.querySelectorAll("#code-log .cm-tool")].map((t) => t.querySelector(".nm").textContent + ":" + t.querySelector(".chip").textContent),
    readOnly: document.getElementById("code-row").hidden && !document.getElementById("code-driver").hidden,
    tree: [...document.querySelectorAll("#code-tree .f")].map((f) => f.dataset.path),
    run: document.querySelector(".pv-run")?.textContent,
  }));
  check("peer: timeline shows the 6 tool cards", P.tools.join(" ") === H.tools.join(" "), P.tools.join(" "));
  check("peer: read-only (no prompt row, 'the host is driving')", P.readOnly);
  check("peer: file tree", P.tree.join(",") === H.tree.join(","), P.tree);
  check("peer: click-to-run button", P.run === "Run preview :5173", P.run);
  await peer.click(".pv-run");
  await peer.waitForFunction(() => /^rev \d+$/.test(document.getElementById("pv-state").textContent), null, { timeout: 10000 });
  await peer.waitForTimeout(700);
  const peerRev = +(/rev (\d+)/.exec(await peer.textContent("#pv-state"))?.[1] || 0);
  const peerFrame = peer.frames().find((f) => f !== peer.mainFrame());
  const peerPx = await peerFrame.evaluate(() => {
    const c = document.getElementById("board"), d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let drawn = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i]) drawn++;
    return { w: c.width, drawn, origin: window.origin };
  });
  check("peer preview: canvas drawn, same rev as the host", peerPx.w > 0 && peerPx.drawn > 1000 && peerRev === hostRev && hostRev > 0, JSON.stringify({ peerPx, peerRev, hostRev }));
  check("peer preview runs in an opaque origin", peerPx.origin === "null", peerPx.origin);
  check("peer console: its own copy ran clean", await peer.evaluate(() => document.getElementById("pv-counts").dataset.errors === "0"));

  if (flag("shots")) {
    fs.mkdirSync(SHOTS, { recursive: true });
    await host.evaluate(() => { document.getElementById("code-log").scrollTop = 1e6; });
    await host.waitForTimeout(300);
    await host.screenshot({ path: path.join(SHOTS, "code-desktop.png") });
    await host.locator("#pv-frame-wrap").screenshot({ path: path.join(SHOTS, "code-preview.png") });
    await peer.evaluate(() => { document.getElementById("code-log").scrollTop = 1e6; });
    await peer.waitForTimeout(300);
    await peer.screenshot({ path: path.join(SHOTS, "code-peer.png") });
    await host.setViewportSize({ width: 400, height: 860 });
    await host.waitForTimeout(400);
    await host.evaluate(() => { document.getElementById("code-log").scrollTop = 1e6; });
    await host.screenshot({ path: path.join(SHOTS, "code-400.png") });
    await host.evaluate(() => document.getElementById("code-out").scrollIntoView());
    await host.waitForTimeout(200);
    await host.screenshot({ path: path.join(SHOTS, "code-400-preview.png") });
    check("400px: no horizontal page scroll", await host.evaluate(() => document.documentElement.scrollWidth <= 400 && document.getElementById("chatpane").scrollWidth <= 400));
    await host.setViewportSize({ width: 1440, height: 900 });
    log(`screenshots in ${SHOTS}`);
  }

  // ---- Stop mid-answer releases the room; the next request runs
  await host.evaluate(async () => {
    const { scripted } = await import("/tests/scripted-model.js");
    window.__pooledMock.model = scripted(["Adding a score box. " + "Let me think about the layout first. ".repeat(60), "Nothing to add."], { piece: 4, delay: 20 });
  });
  await host.fill("#code-prompt", "add a score box");
  await host.click("#code-send");
  await host.waitForFunction(() => /Let me think/.test([...document.querySelectorAll("#code-log .cm-text")].pop()?.textContent || ""), null, { timeout: 10000 });
  await host.click("#code-stop");
  await host.waitForFunction(() => document.querySelectorAll("#code-log .cm-stats").length >= 2, null, { timeout: 10000 });
  const S = await host.evaluate(() => ({ note: [...document.querySelectorAll("#code-log .cm-note")].map((n) => n.textContent), stats: [...document.querySelectorAll("#code-log .cm-stats")].pop().textContent,
    busy: window.__pooledMock.api.busy(), send: !document.getElementById("code-send").hidden }));
  check("Stop ends the run and releases the room lock", S.note.includes("stopped") && /stopped/.test(S.stats) && !S.busy && S.send, JSON.stringify(S));
  await host.fill("#code-prompt", "anything else?");
  await host.click("#code-send");
  await host.waitForFunction(() => document.querySelectorAll("#code-log .cm-stats").length >= 3, null, { timeout: 10000 });
  const after = await host.evaluate(() => [...document.querySelectorAll("#code-log .cm-text")].pop()?.textContent || "");
  check("the next request runs after a stop", /Nothing to add/.test(after), after);
  await peer.waitForFunction(() => document.querySelectorAll("#code-log .cm-stats").length >= 3, null, { timeout: 10000 })
    .then(() => check("peer follows the later requests", true), () => check("peer follows the later requests", false));

  // ---- chat mode is still there
  await host.click("#mode-chat");
  check("Chat tab brings the chat back", await host.evaluate(() => !document.getElementById("chatpane").classList.contains("code-mode") && getComputedStyle(document.getElementById("code-pane")).display === "none"));

  for (const [n, e] of Object.entries(errs)) check(`${n}: no page or console errors`, !e.length, e.join("\n"));
  code = results.every((r) => r.ok) ? 0 : 1;
  console.log(`${results.filter((r) => r.ok).length}/${results.length} passed`);
} catch (e) {
  console.error(e);
  code = 1;
} finally {
  await browser.close();
  srv.close();
  peerServer.kill();
}
process.exit(code);
