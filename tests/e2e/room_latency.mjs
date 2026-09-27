// Room latency sweep: one room of N headless Chromium tabs on this machine (real PeerJS signaling,
// real WebRTC on loopback, the machine's one GPU), then for each one-way latency and for plain and
// speculative decoding: new chat, ask the `japan` prompt, record time to first token, prefill and
// decode tok/s. Manual trigger only (benchmark branch bench/latency).
//
//   node tests/e2e/room_latency.mjs --model qwen3.6-35b-moe --devices 2 --lat 0,5,20 --maxnew 128
//
// Latency is emulated in the page, not by the kernel (netem needs root): the room already has
// ?netlag=ms, which holds every activation frame a device sends for that long before it goes on
// the wire. This harness serves room.js with two dev-only edits so one loaded room can sweep:
// the lag is read from window.__netlag at send time, and window.__nospec switches speculative
// decoding off (plain one-token-per-hop decode). room.js on disk is not changed.
// What it does not emulate: bandwidth, loss, the congestion-window round trips a real link pays on
// big frames (docs/bench-log.md, "Hidden-state transport"), and control messages (chat tokens,
// telemetry) are not delayed. Every tab shares one GPU, so per-hop compute is serialized.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const MODEL = arg("model", "qwen3.6-35b-moe");
const DEVICES = Math.max(1, +arg("devices", 2));
const LATS = arg("lat", "0,5,20").split(",").map(Number);
const MODES = arg("modes", "plain,spec").split(",");
const ROUNDS = +arg("rounds", 2);          // per (lat, mode); the first is a warm-up, the rest are kept
const MAXNEW = +arg("maxnew", 128);
const PORT = +arg("port", 8123), SIGNAL_PORT = +arg("signal-port", 9000), TLS_PORT = PORT + 1;
const PROMPT = "I am planning a two week trip through Japan in late October with my partner. We land in Tokyo, want three days there, then a day trip to Nikko, then the bullet train to Kyoto for four days with a side trip to Nara, then two nights in Osaka, and we fly home from Osaka. We like food markets, old temples, hiking, and small neighborhood bars, and we want to avoid the most crowded tourist spots where we can. Our budget is moderate, around two hundred dollars a day for the two of us not counting hotels. Please give me a day by day itinerary with one main activity each morning and afternoon, a neighborhood to eat dinner in each night, and tell me which days I should buy a rail pass for and whether it is worth it at all.";
const NEED = { "qwen3.8-27b": 16.5, "qwen3.6-35b-moe": 22.5, "qwen3-1.7b": 2.0, "qwen3-0.6b": 0.8 }[MODEL] || 2;
// roughly even split: every device pledges its share plus a little (the host also holds embed/head)
const GB = (name) => String(Math.ceil(NEED / DEVICES + (name === "host" ? 2 : 0.5)));
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

// dev-only view of room.js: live netlag and a plain-decode switch (see the header)
function patchRoom(src) {
  const a = "if (NETLAG) { setTimeout(() => sendHiddenNow(id, msg), NETLAG); return; }";
  const b = "else if (ai.engine.mtp && ai.engine.specStep) {";
  if (!src.includes(a) || !src.includes(b)) throw new Error("room.js changed: update patchRoom in room_latency.mjs");
  return src.replace(a, "const lag = window.__netlag ?? NETLAG; if (lag) { setTimeout(() => sendHiddenNow(id, msg), lag); return; }")
    .replace(b, "else if (ai.engine.mtp && ai.engine.specStep && !window.__nospec) {");
}
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream");
  if (p === path.join(ROOT, "room.js")) { r.end(patchRoom(fs.readFileSync(p, "utf8"))); return; }
  fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-lat-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const BASE = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}&maxnew=${MAXNEW}&peerweights=0`;

const browser = await chromium.launch({ headless: false, args: ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=65536"] });
const ctx = await browser.newContext({ userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36", ignoreHTTPSErrors: true });
await ctx.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
});
const tabs = { host: await ctx.newPage() };
for (let i = 1; i < DEVICES; i++) tabs["worker" + i] = await ctx.newPage();
const errs = [];
for (const [n, p] of Object.entries(tabs)) { p.on("pageerror", (e) => errs.push(n + ": " + String(e).slice(0, 200))); p.on("crash", () => errs.push(n + ": crashed")); }
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s", ...a);
const host = tabs.host;
const out = { model: MODEL, devices: DEVICES, maxnew: MAXNEW, split: "", rows: [] };
try {
  for (const p of Object.values(tabs)) await p.goto(BASE);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  for (const [n, p] of Object.entries(tabs)) { await p.fill("#name-input", n); await p.fill("#join-gb", GB(n)); }
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  for (const n of Object.keys(tabs).filter((k) => k !== "host")) { await tabs[n].fill("#code-input", code); await tabs[n].click("#join-btn"); await tabs[n].waitForTimeout(150); }
  for (const p of Object.values(tabs)) await p.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, DEVICES, { timeout: 120000 });
  await host.waitForTimeout(3000);
  await host.selectOption("#ai-model", MODEL);
  const tLoad = Date.now();
  await host.click("#ai-start");
  for (const [n, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 1800000, polling: 2000 });
  const st0 = await host.textContent("#ai-status");
  if (/^failed:/.test(st0)) throw new Error("load " + st0);
  out.loadS = Math.round((Date.now() - tLoad) / 1000);
  out.split = await host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  log("online in", out.loadS, "s;", out.split);
  await host.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });   // greedy: same answer every round
  for (const lat of LATS) for (const mode of MODES) {
    for (const p of Object.values(tabs)) await p.evaluate(([l, ns]) => { window.__netlag = l; window.__nospec = ns; }, [lat, mode === "plain"]);
    for (let r = 0; r < ROUNDS; r++) {
      await host.evaluate(() => document.getElementById("new-chat").click());
      await host.waitForTimeout(500);
      await host.evaluate((text) => {
        const box = document.getElementById("ai-prompt"); box.value = text; box.dispatchEvent(new Event("input"));
        window.__ttft = null; const t = performance.now();
        const ob = new MutationObserver(() => { const b = [...document.querySelectorAll(".m.bot .bubble")].pop(); if (b && b.textContent.trim()) { window.__ttft = performance.now() - t; ob.disconnect(); } });
        ob.observe(document.body, { subtree: true, childList: true, characterData: true });
        document.getElementById("ai-send").click();
      }, PROMPT);
      await host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 600000, polling: 250 });
      const st = await host.textContent("#ai-status");
      const ttft = await host.evaluate(() => window.__ttft);
      const pre = /prefill (\d+) tok in ([\d.]+)s/.exec(st), dec = /(\d+) tok · ([\d.]+) tok\/s/.exec(st);
      // the host's last speculative-step note ("spec: a/d drafts accepted · lap ms · K tok/s ...")
      const spec = await host.evaluate(() => { try { return JSON.parse(localStorage.getItem("pooled-crumb") || "{}").s || ""; } catch { return ""; } });
      const row = { lat, mode, round: r, ttftMs: ttft && Math.round(ttft), prefillTok: pre && +pre[1], prefillS: pre && +pre[2], tokens: dec && +dec[1], tps: dec && +dec[2], spec: mode === "spec" && spec.startsWith("spec:") ? spec.slice(0, 200) : "", status: st.slice(0, 160) };
      log(JSON.stringify(row));
      out.rows.push(row);
    }
  }
} catch (e) {
  out.error = String(e).slice(0, 400);
  log("FAILED", out.error);
} finally {
  out.errors = errs;
  console.log(JSON.stringify(out));
  await browser.close(); srv.close(); wsrv.close(); peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
}
