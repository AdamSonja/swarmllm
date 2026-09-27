// The Code-mode preview sandbox alone (docs/design/harness-app.md G.2): a PreviewServer over an
// in-memory project, mounted with mountPreview in a sandboxed srcdoc iframe, driven through the
// serve / preview_logs tools. No WebGPU, no room, no PeerJS; takes a few seconds.
//   node tests/e2e/preview_browser.mjs
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";
const PORT = 18986;

async function setup() {
  const { MemoryWorkspace, watch } = await import("/harness/workspace.js");
  const { PreviewServer } = await import("/harness/preview.js");
  const { previewTools } = await import("/harness/preview-tools.js");
  const { codingTools } = await import("/harness/codetools.js");
  const { mountPreview } = await import("/harness/preview-frame.js");
  const c = new OffscreenCanvas(4, 4), g = c.getContext("2d");
  g.fillStyle = "#0f0"; g.fillRect(0, 0, 4, 4);
  const png = new Uint8Array(await (await c.convertToBlob({ type: "image/png" })).arrayBuffer());
  localStorage.setItem("room-secret", "s3cret");   // the frame must not see this
  const ws = watch(new MemoryWorkspace({
    "index.html": `<!doctype html>
<html><head><title>t</title><link rel="stylesheet" href="style.css"></head>
<body><canvas id="board" width="120" height="60"></canvas><img id="spr" src="img/sprite.png">
<a id="next" href="page2.html">next</a>
<script type="module" src="game.js"></script></body></html>`,
    "style.css": "body { background: rgb(1, 2, 3); margin: 0 }",
    "game.js": `import { draw } from "./lib/draw.js";
const ctx = document.getElementById("board").getContext("2d");
draw(ctx);
window.__level = await (await fetch("data/level.json")).json();
setTimeout(() => {
  boom();
}, 50);
`,
    "lib/draw.js": `import { COLOR } from "./color.js";\nexport function draw(ctx) {\n  console.log("drawing", COLOR);\n  ctx.fillStyle = COLOR; ctx.fillRect(0, 0, 120, 60);\n}\n`,
    "lib/color.js": `export const COLOR = "rgb(255, 0, 0)";\n`,
    "data/level.json": `{"rows": 20}`,
    "page2.html": `<!doctype html><p id="p2">page two</p><script>console.log("on page 2")</script>`,
  }));
  await ws.writeBytes("img/sprite.png", png);
  const server = new PreviewServer(ws);
  const T = Object.fromEntries([...codingTools(ws, { server }), ...previewTools(server)].map((t) => [t.name, t]));
  const el = document.createElement("div");
  el.style.cssText = "width:400px;height:300px";
  document.body.append(el);
  const logs = [];
  const view = mountPreview(el, server, 5173, { onLog: (e) => logs.push(e) });
  Object.assign(window, { __ws: ws, __server: server, __T: T, __logs: logs, __view: view, __mountPreview: mountPreview });
  return await T.serve.run({});
}

const srv = serveRepo(PORT, {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: ["--no-sandbox"] });
const results = [];
const check = (n, ok, d = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"} ${n}${d && !ok ? "  " + String(d).slice(0, 400) : ""}`); };
let code = 1;
try {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => { if (!/boom is not defined|Cannot access 'ctx'/.test(String(e))) pageErrors.push(String(e)); });   // the app's own bug is expected
  await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
  const serveOut = await page.evaluate(setup);
  console.log("--- serve result\n" + serveOut + "\n---");
  check("serve reports the port and files", /^serving \. on :5173 \(index\.html, 8 files, [\d.]+ (KB|B)\)/.test(serveOut), serveOut);
  check("serve reports the load and the error with its file:line", /loaded in \d+ ms · 1 error:/.test(serveOut) && /error game\.js:6:\d+ ReferenceError: boom is not defined/.test(serveOut), serveOut);

  const frame = () => page.frames().find((f) => f !== page.mainFrame());
  const F = await frame().evaluate(async (port) => {
    const out = {};
    const px = document.getElementById("board").getContext("2d").getImageData(5, 5, 1, 1).data;
    out.pixel = [...px];
    out.bg = getComputedStyle(document.body).backgroundColor;
    out.img = document.getElementById("spr").naturalWidth;
    out.origin = window.origin;
    out.level = window.__level;
    const threw = (f) => { try { f(); return false; } catch { return true; } };
    out.parentDoc = threw(() => parent.document.title);
    out.parentStorage = threw(() => parent.localStorage.getItem("room-secret"));
    out.cookie = threw(() => document.cookie);
    localStorage.setItem("hi", "5");
    out.shim = localStorage.getItem("hi") === "5" && localStorage.getItem("room-secret") === null;
    out.opfs = await navigator.storage.getDirectory().then(() => "opened", (e) => "blocked: " + e.name);
    out.idb = await new Promise((res) => { try { const q = indexedDB.open("pooled-projects"); q.onsuccess = () => res("opened"); q.onerror = () => res("blocked"); } catch (e) { res("blocked: " + e.name); } });
    out.fetchRoom = await fetch(`http://127.0.0.1:${port}/p2p.html`).then(() => "fetched", (e) => "blocked: " + e.name);
    out.fetchRel = await fetch("/p2p.html").then((r) => r.status, () => "threw");
    out.imgRoom = await new Promise((res) => { const i = new Image(); i.onload = () => res("loaded"); i.onerror = () => res("blocked"); i.src = `http://127.0.0.1:${port}/favicon.svg`; });
    out.top = threw(() => { top.location.href = "https://example.com/"; });
    out.alert = (alert("hello"), "returned");
    return out;
  }, PORT);
  check("module chain of 3 drew the canvas", F.pixel.join() === "255,0,0,255", F.pixel);
  check("CSS applied", F.bg === "rgb(1, 2, 3)", F.bg);
  check("image from the project loaded", F.img === 4, F.img);
  check("fetch of a project file through the shim", F.level?.rows === 20, JSON.stringify(F.level));
  check("opaque origin", F.origin === "null", F.origin);
  check("parent.document blocked", F.parentDoc);
  check("parent localStorage blocked", F.parentStorage);
  check("cookies blocked", F.cookie);
  check("localStorage shim works and holds nothing of the room's", F.shim);
  check("OPFS blocked", F.opfs.startsWith("blocked"), F.opfs);
  check("IndexedDB blocked", F.idb.startsWith("blocked"), F.idb);
  check("fetch to the room's origin refused by the CSP", F.fetchRoom.startsWith("blocked"), F.fetchRoom);
  check("rooted fetch is a 404 from the shim, not a request", F.fetchRel === 404, F.fetchRel);
  check("image from the room's origin refused", F.imgRoom === "blocked", F.imgRoom);
  check("top navigation blocked", F.top);
  check("alert does not block", F.alert === "returned");

  const L = await page.evaluate(() => window.__logs.map((e) => ({ level: e.level, text: e.text.split("\n")[0], src: e.src, line: e.line })));
  check("console.log carries its file and line", L.some((e) => e.level === "log" && e.text === 'drawing rgb(255, 0, 0)' && e.src === "lib/draw.js" && e.line === 3), JSON.stringify(L));
  check("the uncaught error carries game.js:6", L.some((e) => e.level === "error" && /boom is not defined/.test(e.text) && e.src === "game.js" && e.line === 6), JSON.stringify(L));
  check("sandbox escapes logged as errors, alert as info", L.some((e) => /404 p2p\.html \(fetch\)/.test(e.text)) && L.some((e) => e.level === "info" && e.text === "alert: hello"), JSON.stringify(L));

  // live reload: an edit through the tools bumps the rev and the frame redraws
  const edit = await page.evaluate(() => window.__T.edit_file.run({ path: "lib/color.js", old: "rgb(255, 0, 0)", new: "rgb(0, 0, 255)" }));
  check("edit result says the preview reloads", edit === "edited lib/color.js line 1 (1 -> 1 lines) · preview :5173 reloaded", edit);
  const rev2 = await page.evaluate(() => new Promise((res) => { const off = window.__server.onUpdate((u) => { off(); res(u); }); }));
  check("update after the debounce with the changed file", rev2.rev === 2 && rev2.changed.join() === "lib/color.js", JSON.stringify(rev2));
  await page.evaluate(() => window.__server.whenIdle(5173, 3000));
  const px2 = await frame().evaluate(() => [...document.getElementById("board").getContext("2d").getImageData(5, 5, 1, 1).data].join());
  check("frame reloaded with the edit", px2 === "0,0,255,255", px2);
  const logsOut = await page.evaluate(() => window.__T.preview_logs.run({ since: 0 }));
  console.log("--- preview_logs\n" + logsOut + "\n---");
  check("preview_logs lists both revs and a cursor", /\(rev 1\)/.test(logsOut) && /\(rev 2, current\)/.test(logsOut) && /next: since=\d+$/.test(logsOut), logsOut);

  // a relative link loads the other page in the same sandbox
  await frame().evaluate(() => document.getElementById("next").click());
  await page.waitForFunction(() => window.__logs.some((e) => e.text === "on page 2"), null, { timeout: 5000 }).catch(() => {});
  const p2 = await frame().evaluate(() => document.getElementById("p2")?.textContent).catch(() => null);
  check("link navigation rebuilds the document for page2.html", p2 === "page two", p2);

  // a second port from a subfolder: a top-level TDZ bug in a module and a missing image
  const broken = await page.evaluate(async () => {
    await window.__ws.write("broken/index.html", `<canvas id=b></canvas><img src="gone.png"><script type=module src="main.js"></script>`);
    await window.__ws.write("broken/main.js", `// the classic first-draft bug\nctx.fillRect(0, 0, 1, 1);\nconst ctx = document.getElementById("b").getContext("2d");\n`);
    const el = document.createElement("div"); document.body.append(el);
    window.__view2 = window.__mountPreview(el, window.__server, 5174);
    return window.__T.serve.run({ dir: "broken", port: 5174 });
  });
  console.log("--- serve :5174\n" + broken + "\n---");
  check("second port: TDZ error at main.js:2 and the missing image", /^serving broken on :5174 \(index\.html, 2 files/.test(broken)
    && /· 2 errors:/.test(broken) && /error main\.js:2:\d+ ReferenceError: Cannot access 'ctx' before initialization/.test(broken) && /\nmissing: gone\.png$/.test(broken), broken);
  const ports = await page.evaluate(() => window.__server.ports().map((p) => `${p.port}:${p.dir}`).join(" "));
  check("both ports listed", ports === "5173: 5174:broken", ports);

  // a peer-style mount waits for a click before running anything
  const gated = await page.evaluate(async () => {
    const el = document.createElement("div"); document.body.append(el);
    const v = window.__mountPreview(el, window.__server, 5173, { autorun: false });
    const before = !v.frame.getAttribute("srcdoc") && el.querySelector(".pv-run")?.textContent;
    el.querySelector(".pv-run").click();
    const after = (v.frame.getAttribute("srcdoc") || "").length > 0;
    v.destroy();
    return { before, after };
  });
  check("click-to-run gate", gated.before === "Run preview :5173" && gated.after, JSON.stringify(gated));

  const stop = await page.evaluate(() => window.__T.stop_serve.run({ port: 5173 }));
  check("stop_serve", stop === "stopped :5173" && (await page.evaluate(() => !window.__view.frame.getAttribute("srcdoc"))), stop);
  check("no errors in the room page", !pageErrors.length, pageErrors.join("\n"));
  code = results.every(Boolean) ? 0 : 1;
  console.log(`\n${results.filter(Boolean).length}/${results.length} checks passed`);
  console.log(code ? "PREVIEW FAIL" : "PREVIEW PASS");
} catch (e) {
  console.error("FAILED:", e.stack || e);
  code = 2;
} finally {
  await browser.close(); srv.close();
}
process.exit(code);
