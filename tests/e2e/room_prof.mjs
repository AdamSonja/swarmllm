// Room hop profiler: one room of N devices on this machine (one headless Chromium each) (real PeerJS signaling,
// real WebRTC on loopback, the machine's one GPU), greedy answers in plain and speculative mode, and
// a timestamp trace of every activation frame through every tab, so each lap splits into
// host compute, pack, send, wire, deliver, queue, unpack, GPU, readback, pack, send ... head.
// Manual trigger only (research branch exp/base).
//
//   node tests/e2e/room_prof.mjs --model qwen3.6-35b-moe --devices 2 [--maxnew 48] [--modes plain,spec] [--out f.json] [--query gpusample=1]
//
// room.js and room/transport.js on disk are not changed: this harness serves them with trace marks
// added (patchRoom / patchTransport below; they fail loudly if the anchors move), plus the dev-only
// plain-decode switch room_latency uses (window.__nospec). An init script in every tab adds a GPU
// timestamp pair around every command buffer while tracing (empty timestamped passes at the start
// and end of each encoder, so the engine's passes are untouched) and times every mapAsync.
// Clock: marks are performance.timeOrigin + performance.now() in each browser, shifted onto the
// harness's clock by a measured offset per browser (clockOffset).
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const MODEL = arg("model", "qwen3.6-35b-moe");
const DEVICES = Math.max(2, +arg("devices", 2));
const MODES = arg("modes", "plain,spec").split(",");
const MAXNEW = +arg("maxnew", 48);
const OUT = arg("out", "");
const WIRE = arg("wire", "stripe4");
const PORT = +arg("port", 8131), SIGNAL_PORT = +arg("signal-port", 9011), TLS_PORT = PORT + 1;
const PROMPT = arg("prompt", "Write the Python code for two sum. Code only.");
const NEED = { "qwen3.8-27b": 16.5, "qwen3.6-35b-moe": 22.5, "qwen3-1.7b": 2.0, "qwen3-0.6b": 0.8 }[MODEL] || 2;
const GB = (name) => String(Math.ceil(NEED / DEVICES + (name === "host" ? 2 : 0.5)));
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

// ---- serve-time trace marks ----
const HP = "const __HP = (...a) => globalThis.__hpMark?.(...a);\n";
function rep(src, a, b, file) { if (!src.includes(a)) throw new Error(`${file} changed: anchor not found: ${a.slice(0, 80)}`); return src.replace(a, b); }
function patchRoom(s) {
  const f = "room.js";
  s = HP + s;
  // plain-decode switch (room_latency's)
  s = rep(s, "else if (ai.engine.mtp && ai.engine.specStep) {", "else if (ai.engine.mtp && ai.engine.specStep && !window.__nospec) {", f);
  // host, one token
  s = rep(s, "  let h = await ai.engine.embedRun(id, pos);", "  __HP('h.lap0', 'ai-hidden', pos); let h = await ai.engine.embedRun(id, pos); __HP('h.emb1', 'ai-hidden', pos);", f);
  s = rep(s, "    sendChain({ t: \"ai-hidden\", pos, ...packWire(h) });\n    h = await returned;",
    "    { const __w = packWire(h); __HP('h.pack1', 'ai-hidden', pos); sendChain({ t: \"ai-hidden\", pos, ...__w }); }\n    h = await returned; __HP('h.ret', 'ai-hidden', pos);", f);
  s = rep(s, "  const logits = await ai.engine.headFromHidden(h);", "  __HP('h.head0', 'ai-hidden', pos); const logits = await ai.engine.headFromHidden(h); __HP('h.head1', 'ai-hidden', pos);", f);
  // GPU sampling (exp/gpu-sample, --query gpusample=1): the same marks around the candidates head
  if (s.includes("    const c = await ai.engine.headFromHiddenIds(h, desc);"))
    s = rep(s, "    const c = await ai.engine.headFromHiddenIds(h, desc);", "    __HP('h.head0', 'ai-hidden', pos); const c = await ai.engine.headFromHiddenIds(h, desc); __HP('h.head1', 'ai-hidden', pos);", f);
  // host, speculative verify lap
  s = rep(s, "          const tLap = performance.now();", "          const tLap = performance.now(); __HP('h.lap0', 'ai-hidden-b', pos, tokens.length);", f);
  s = rep(s, "          const hostMs = performance.now() - tLap;", "          const hostMs = performance.now() - tLap; __HP('h.emb1', 'ai-hidden-b', pos, tokens.length);", f);
  s = rep(s, "          sendChain({ t: \"ai-hidden-b\", basePos: pos, n: tokens.length, spec: 1, ...packWire(hb) });\n          const h = await returned;",
    "          { const __w = packWire(hb); __HP('h.pack1', 'ai-hidden-b', pos, tokens.length); sendChain({ t: \"ai-hidden-b\", basePos: pos, n: tokens.length, spec: 1, ...__w }); }\n          const h = await returned; __HP('h.ret', 'ai-hidden-b', pos, tokens.length);", f);
  s = rep(s, "        const toks = viaLookup ? await ai.engine.specStepDrafts(next, sample, lk, spec) : await ai.engine.specStep(next, sample, K, spec);",
    "        __HP('h.step0', viaLookup ? 'lookup' : 'draft', ai.engine.pos, viaLookup ? lk.length : K); const toks = viaLookup ? await ai.engine.specStepDrafts(next, sample, lk, spec) : await ai.engine.specStep(next, sample, K, spec); __HP('h.step1', viaLookup ? 'lookup' : 'draft', ai.engine.pos, toks.length);", f);
  // host, returned frames
  s = rep(s, "    case \"ai-hiddenret-b\": lapDone(\"b\" + d.basePos, unpackWire(d)); break;",
    "    case \"ai-hiddenret-b\": { __HP('h.unp0', 'ai-hiddenret-b', d.basePos); const __u = unpackWire(d); __HP('h.unp1', 'ai-hiddenret-b', d.basePos); lapDone(\"b\" + d.basePos, __u); break; }", f);
  s = rep(s, "    case \"ai-hiddenret\": lapDone(d.pos, unpackWire(d)); break;",
    "    case \"ai-hiddenret\": { __HP('h.unp0', 'ai-hiddenret', d.pos); const __u = unpackWire(d); __HP('h.unp1', 'ai-hiddenret', d.pos); lapDone(d.pos, __u); break; }", f);
  // worker
  s = rep(s, "      ai.q = ai.q.then(() => workerFrame(d))", "      __HP('w.enq', d.t, d.t === 'ai-hidden' ? d.pos : d.basePos); ai.q = ai.q.then(() => workerFrame(d))", f);
  s = rep(s, "  const t0 = performance.now();\n  if (d.t === \"ai-hidden-b\") {", "  const t0 = performance.now(); __HP('w.start', d.t, d.t === 'ai-hidden' ? d.pos : d.basePos, d.n || 1);\n  if (d.t === \"ai-hidden-b\") {", f);
  s = rep(s, "    const xs = unpackWire(d);", "    const xs = unpackWire(d); __HP('w.unp1', d.t, d.basePos);", f);
  s = rep(s, "    teleNote(d.spec ? \"spec\" : \"pre\", performance.now() - t0);", "    __HP('w.gpu1', d.t, d.basePos); teleNote(d.spec ? \"spec\" : \"pre\", performance.now() - t0);", f);
  s = rep(s, "    const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) };",
    "    const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) }; __HP('w.pack1', d.t, d.basePos);", f);
  s = rep(s, "    const hin = unpackWire(d);", "    const hin = unpackWire(d); __HP('w.unp1', d.t, d.pos);", f);
  s = rep(s, "    teleNote(\"one\", performance.now() - t0);", "    __HP('w.gpu1', d.t, d.pos); teleNote(\"one\", performance.now() - t0);", f);
  s = rep(s, "    const msg = { pos: d.pos, ...packWire(h) };", "    const msg = { pos: d.pos, ...packWire(h) }; __HP('w.pack1', d.t, d.pos);", f);
  return s;
}
function patchTransport(s) {
  const f = "room/transport.js";
  s = HP + s;
  s = rep(s, "  const flags = packFlags(msg);", "  const flags = packFlags(msg); __HP('send0', msg.t, pos, bytes.length, nSlices);", f);
  s = rep(s, "    ch.send(buf);\n  }\n  return true;", "    ch.send(buf);\n  }\n  __HP('send1', msg.t, pos);\n  return true;", f);
  s = rep(s, "  if (!r) { r = {", "  if (!r) { __HP('rx0', KINDS[kind], pos, nSlices); r = {", f);
  s = rep(s, "  link.recv++;", "  link.recv++; __HP('rx1', KINDS[kind], pos);", f);
  s = rep(s, "    link.expect++;\n    onFrame(m);", "    link.expect++;\n    __HP('dlv', m.t, m.pos ?? m.basePos);\n    onFrame(m);", f);
  return s;
}
const PATCHED = { [path.join(ROOT, "room.js")]: patchRoom, [path.join(ROOT, "room/transport.js")]: patchTransport };

// ---- per-tab trace + GPU hooks (runs before any page script) ----
const INIT = `(() => {
  const T = () => performance.timeOrigin + performance.now();
  window.__hpOn = false; window.__hp = []; window.__gp = { subs: [], maps: [] };
  window.__hpMark = (ev, kind, pos, a, b) => { if (window.__hpOn) window.__hp.push([ev, kind, pos, T(), a, b]); };
  if (!self.GPUAdapter) return;
  const oReq = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc = {}) {
    const ts = this.features.has("timestamp-query");
    const d = await oReq.call(this, ts ? { ...desc, requiredFeatures: [...(desc.requiredFeatures || []), "timestamp-query"] } : desc);
    if (!ts) return d;
    const MAXQ = 4096, qs = d.createQuerySet({ type: "timestamp", count: MAXQ });
    let nq = 0; const cbRec = new WeakMap(); const did = (window.__gpDevs ||= []).length;
    const oEnc = d.createCommandEncoder.bind(d);
    d.createCommandEncoder = (dd) => {
      const e = oEnc(dd);
      if (!window.__hpOn || nq + 2 > MAXQ) return e;
      const rec = { q: nq, d: did, tEnc: T(), t: 0, gpu: 0 }; nq += 2;
      e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: rec.q } }).end();
      const fin = e.finish.bind(e);
      e.finish = (x) => { e.beginComputePass({ timestampWrites: { querySet: qs, endOfPassWriteIndex: rec.q + 1 } }).end(); const cb = fin(x); cbRec.set(cb, rec); return cb; };
      return e;
    };
    const q = d.queue, oSub = q.submit.bind(q);
    q.submit = (cbs) => { if (window.__hpOn) { const t = T(); for (const cb of cbs) { const r = cbRec.get(cb) || { q: -1, tEnc: t }; r.t = t; window.__gp.subs.push(r); } } return oSub(cbs); };
    // the room makes several devices (a probe, the engine's, a throwaway): each resolves its own queries
    window.__gpDevs.push(async () => {
      if (!nq) return;
      const res = d.createBuffer({ size: nq * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const rd = d.createBuffer({ size: nq * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e = oEnc(); e.resolveQuerySet(qs, 0, nq, res, 0); e.copyBufferToBuffer(res, 0, rd, 0, nq * 8); oSub([e.finish()]);
      await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
      for (const s of window.__gp.subs) if (s.d === did && s.q >= 0 && t[s.q + 1] > t[s.q]) s.gpu = Number(t[s.q + 1] - t[s.q]) / 1e6;
      nq = 0;
    });
    window.__gpResolve = async () => { for (const f of window.__gpDevs) await f(); };
    return d;
  };
  const oMap = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) {
    if (!window.__hpOn) return oMap.apply(this, a);
    const r = { t0: T(), t1: 0, bytes: a[2] ?? this.size }; window.__gp.maps.push(r);
    return oMap.apply(this, a).then((v) => { r.t1 = T(); return v; });
  };
})();`;

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream");
  if (PATCHED[p]) { r.end(PATCHED[p](fs.readFileSync(p, "utf8"))); return; }
  fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
// check the patches apply before starting a browser
for (const [p, fn] of Object.entries(PATCHED)) fn(fs.readFileSync(p, "utf8"));
if (process.argv.includes("--check")) { for (const [p, fn] of Object.entries(PATCHED)) fs.writeFileSync(path.join(os.tmpdir(), "prof_" + path.basename(p)), fn(fs.readFileSync(p, "utf8"))); console.log("patches apply"); process.exit(0); }
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-prof-"));
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
// --query "a=1&b=2": extra room URL parameters on every tab (e.g. gpusample=1)
const BASE = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}&maxnew=${MAXNEW}&peerweights=0&wire=${WIRE}` + (arg("query") ? "&" + arg("query") : "");

// One Chromium per device, each with its own on-disk profile (as tests/e2e/room_latency.mjs on
// bench/latency does: with every tab in one off-the-record context the Cache API weight store lives
// in RAM and the 35B MoE crashes the browser). Separate browser and GPU processes, like separate machines.
const ARGS = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=65536"];
const UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const profiles = [], contexts = [], tabs = {};
for (let i = 0; i < DEVICES; i++) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-prof-profile-")); profiles.push(dir);
  const c = await chromium.launchPersistentContext(dir, { headless: false, args: ARGS, userAgent: UA, ignoreHTTPSErrors: true });
  await c.addInitScript(INIT);
  await c.route("**/*.gguf", (route) => {
    const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
    if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
    return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
  });
  contexts.push(c);
  tabs[i === 0 ? "host" : "worker" + i] = c.pages()[0] || await c.newPage();
}
// each browser has its own clock origin: offset of (timeOrigin + now) against this process's clock,
// from the lowest-round-trip of 15 probes (error <= half that round trip)
async function clockOffset(p) {
  let best = null;
  for (let i = 0; i < 15; i++) {
    const a = performance.timeOrigin + performance.now();
    const t = await p.evaluate(() => performance.timeOrigin + performance.now());
    const b = performance.timeOrigin + performance.now();
    if (!best || b - a < best.rtt) best = { rtt: b - a, off: t - (a + b) / 2 };
  }
  return best;
}
const errs = [];
for (const [n, p] of Object.entries(tabs)) {
  p.on("pageerror", (e) => errs.push(n + ": " + String(e).slice(0, 200)));
  p.on("crash", () => { errs.push(n + ": crashed"); console.error(n, "CRASHED"); });
  p.on("console", (m) => { if (m.type() === "error") console.error(n, "console:", m.text().slice(0, 200)); });
}
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s", ...a);
const host = tabs.host;
const out = { model: MODEL, devices: DEVICES, wire: WIRE, maxnew: MAXNEW, prompt: PROMPT, split: "", runs: [] };
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
  const poll = setInterval(async () => { for (const [n, p] of Object.entries(tabs)) { try { log(n, (await p.textContent("#ai-status")).slice(0, 100)); } catch {} } }, 30000);
  for (const [n, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 1800000, polling: 2000 });
  clearInterval(poll);
  const st0 = await host.textContent("#ai-status");
  if (/^failed:/.test(st0)) throw new Error("load " + st0);
  out.loadS = Math.round((Date.now() - tLoad) / 1000);
  out.split = await host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  log("online in", out.loadS, "s;", out.split);
  await host.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
  const ask = async () => {
    await host.evaluate(() => document.getElementById("new-chat").click());
    await host.waitForTimeout(500);
    await host.evaluate((text) => { const box = document.getElementById("ai-prompt"); box.value = text; box.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, PROMPT);
    await host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 600000, polling: 250 });
    return host.textContent("#ai-status");
  };
  for (const mode of MODES) {
    for (const p of Object.values(tabs)) await p.evaluate((ns) => { window.__nospec = ns; }, mode === "plain");
    log(mode, "warm-up:", (await ask()).slice(0, 120));
    for (const p of Object.values(tabs)) await p.evaluate(() => { window.__hp = []; window.__gp = { subs: [], maps: [] }; window.__hpOn = true; });
    const st = await ask();
    for (const p of Object.values(tabs)) await p.evaluate(() => { window.__hpOn = false; });
    for (const p of Object.values(tabs)) await p.evaluate(() => window.__gpResolve?.());
    const traces = {};
    for (const [n, p] of Object.entries(tabs)) {
      const c = await clockOffset(p);
      const tr = await p.evaluate(() => ({ hp: window.__hp, gp: window.__gp }));
      // shift every mark onto this process's clock
      for (const e of tr.hp) e[3] -= c.off;
      for (const x of tr.gp.subs) { x.t -= c.off; x.tEnc -= c.off; }
      for (const m of tr.gp.maps) { m.t0 -= c.off; if (m.t1) m.t1 -= c.off; }
      traces[n] = { ...tr, clock: c };
    }
    const reply = await host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return (b[b.length - 1]?.textContent || "").slice(0, 160); });
    const crumb = await host.evaluate(() => { try { return JSON.parse(localStorage.getItem("pooled-crumb") || "{}").s || ""; } catch { return ""; } });
    log(mode, "measured:", st.slice(0, 160));
    out.runs.push({ mode, status: st, reply, crumb, traces });
  }
} catch (e) {
  out.error = String(e).slice(0, 400);
  log("FAILED", out.error);
} finally {
  out.errors = errs;
  if (OUT) fs.writeFileSync(OUT, JSON.stringify(out));
  else console.log(JSON.stringify(out));
  peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
  await Promise.race([Promise.all(contexts.map((c) => c.close().catch(() => {}))), new Promise((r) => setTimeout(r, 15000))]);   // a crashed browser can hang close()
  for (const d of profiles) fs.rmSync(d, { recursive: true, force: true });
  srv.close(); wsrv.close();
  process.exit(out.error ? 1 : 0);
}
