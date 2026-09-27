// Tables from a tests/e2e/room_prof.mjs trace: node tests/e2e/room_prof_report.mjs trace.json [--json]
// Per lap (one frame's trip host -> workers -> host), medians over the measured answer:
//   host:   compute (embed + its layers: CPU encode, GPU span, readback wait), pack (NaN scan + f16),
//           send (slicing + channel sends)
//   per worker hop: wire (sender's last send -> this tab's last slice), deliver (reassembly -> app),
//           queue (waiting behind the previous frame), unpack (f16 -> f32), encode (unpack -> first
//           submit: writeBuffer + encoding), GPU (timestamped spans of its submits), readback (the
//           rest of the compute window: submit -> mapAsync resolved minus GPU), pack, send
//   return: wire, deliver + unpack, promise resolution back into the decode loop
// and per token / step what the host does outside the lap (head, drafts, sampling, UI).
import fs from "fs";
const J = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const asJson = process.argv.includes("--json");
const med = (a) => { const s = a.filter((x) => x != null && isFinite(x)).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const mean = (a) => { const s = a.filter((x) => x != null && isFinite(x)); return s.length ? s.reduce((x, y) => x + y, 0) / s.length : null; };
const r2 = (x) => (x == null ? "" : Math.round(x * 100) / 100);
function table(rows, cols) {
  const w = cols.map((c) => Math.max(c.length, ...rows.map((r) => String(r[c] ?? "").length)));
  const line = (v) => "| " + v.map((x, i) => String(x ?? "").padEnd(w[i])).join(" | ") + " |";
  return [line(cols), "|" + w.map((x) => "-".repeat(x + 2)).join("|") + "|", ...rows.map((r) => line(cols.map((c) => r[c])))].join("\n");
}
const report = { model: J.model, devices: J.devices, wire: J.wire, split: J.split, runs: [] };
for (const run of J.runs || []) {
  const T = run.traces, names = Object.keys(T), workers = names.filter((n) => n !== "host");
  const ev = {};   // tab -> ev -> "kind|pos" -> [t, a, b]
  for (const n of names) { ev[n] = {}; for (const [e, k, p, t, a, b] of T[n].hp) { const m = (ev[n][e] ||= {}); const key = k + "|" + p; if (!m[key]) m[key] = [t, a, b]; } }
  const get = (n, e, k, p) => ev[n][e]?.[k + "|" + p]?.[0];
  const gpuIn = (n, a, b) => { const s = T[n].gp.subs.filter((x) => x.t >= a && x.t <= b); return { n: s.length, gpu: s.reduce((x, y) => x + (y.gpu || 0), 0), first: s[0]?.t, last: s.at(-1)?.t }; };
  const mapsIn = (n, a, b) => T[n].gp.maps.filter((m) => m.t0 >= a && m.t0 <= b);
  const laps = [];
  for (const [key, [t0, cols]] of Object.entries(ev.host["h.lap0"] || {})) {
    const [kind, ps] = key.split("|"), pos = +ps, rk = kind === "ai-hidden" ? "ai-hiddenret" : "ai-hiddenret-b";
    const h = (e, k = kind) => get("host", e, k, pos);
    const ret = h("h.ret"); if (!ret) continue;
    const L = { kind, pos, cols: cols || 1, lap: ret - t0 };
    const hc = gpuIn("host", t0, h("h.emb1"));
    L.hostCompute = h("h.emb1") - t0; L.hostGpu = hc.gpu; L.hostSubmits = hc.n; L.hostEncode = hc.first ? hc.first - t0 : null;
    L.hostPack = h("h.pack1") - h("h.emb1"); L.hostSend = h("send1") - h("send0");
    const order = workers.filter((w) => get(w, "rx1", kind, pos)).sort((a, b) => get(a, "rx1", kind, pos) - get(b, "rx1", kind, pos));
    let prevSend = h("send1");
    L.hops = order.map((w, i) => {
      const g = (e, k = kind) => get(w, e, k, pos);
      const outK = i === order.length - 1 ? rk : kind;
      const c = gpuIn(w, g("w.unp1"), g("w.gpu1"));
      const compute = g("w.gpu1") - g("w.unp1");
      const hop = { tab: w, wire: g("rx1") - prevSend, firstSlice: g("rx0") - prevSend, deliver: g("dlv") - g("rx1"), queue: g("w.start") - g("dlv"), unpack: g("w.unp1") - g("w.start"),
        encode: c.first ? c.first - g("w.unp1") : null, gpu: c.gpu, submits: c.n, readback: c.first ? compute - (c.first - g("w.unp1")) - c.gpu : null, compute,
        maps: mapsIn(w, g("w.unp1"), g("w.gpu1")).length, pack: g("w.pack1") - g("w.gpu1"), send: get(w, "send1", outK, pos) - g("w.pack1") };
      prevSend = get(w, "send1", outK, pos);
      return hop;
    });
    L.retWire = get("host", "rx1", rk, pos) - prevSend; L.retDeliver = get("host", "dlv", rk, pos) - get("host", "rx1", rk, pos);
    L.retUnpack = get("host", "h.unp1", rk, pos) - get("host", "h.unp0", rk, pos); L.retResolve = ret - get("host", "h.unp1", rk, pos);
    if (h("h.head0")) L.head = h("h.head1") - h("h.head0");
    laps.push(L);
  }
  laps.sort((a, b) => a.pos - b.pos);
  const dec = run.mode === "plain" ? laps.filter((l) => l.head != null) : laps.filter((l) => l.kind === "ai-hidden-b");
  const R = { mode: run.mode, status: run.status, crumb: run.crumb, laps: dec.length };
  // lap segments
  const seg = (f) => ({ median: r2(med(dec.map(f))), mean: r2(mean(dec.map(f))) });
  const rows = [
    ["host: embed + layers (total)", (l) => l.hostCompute], ["  host: CPU encode", (l) => l.hostEncode], ["  host: GPU span", (l) => l.hostGpu], ["  host: readback wait (rest)", (l) => l.hostCompute - l.hostGpu - (l.hostEncode || 0)],
    ["host: NaN scan + f16 pack", (l) => l.hostPack], ["host: send (slice + channel)", (l) => l.hostSend],
  ];
  const nh = Math.max(0, ...dec.map((l) => l.hops.length));
  for (let i = 0; i < nh; i++) {
    const H = (f) => (l) => (l.hops[i] ? f(l.hops[i]) : null);
    rows.push([`hop ${i + 1} wire (last slice)`, H((x) => x.wire)], [`hop ${i + 1}   first slice`, H((x) => x.firstSlice)], [`hop ${i + 1} deliver`, H((x) => x.deliver)], [`hop ${i + 1} queue`, H((x) => x.queue)],
      [`hop ${i + 1} unpack f16`, H((x) => x.unpack)], [`hop ${i + 1} CPU encode`, H((x) => x.encode)], [`hop ${i + 1} GPU span`, H((x) => x.gpu)], [`hop ${i + 1} readback wait`, H((x) => x.readback)],
      [`hop ${i + 1} NaN scan + pack`, H((x) => x.pack)], [`hop ${i + 1} send`, H((x) => x.send)],
      [`hop ${i + 1} TOTAL (arrival -> next arrival)`, H((x) => x.deliver + x.queue + x.compute + x.unpack + x.pack + x.send)],
      [`hop ${i + 1} overhead (TOTAL - GPU)`, H((x) => x.deliver + x.queue + x.compute + x.unpack + x.pack + x.send - x.gpu)]);
  }
  rows.push(["return wire", (l) => l.retWire], ["return deliver", (l) => l.retDeliver], ["return unpack f16", (l) => l.retUnpack], ["return promise -> loop", (l) => l.retResolve], ["LAP (host send start -> hidden back)", (l) => l.lap]);
  if (run.mode === "plain") rows.push(["head (upload + norm + LM head + 1 MB readback)", (l) => l.head]);
  R.segments = rows.map(([name, f]) => ({ segment: name, ...seg(f) }));
  R.hopSummary = [];
  for (let i = 0; i < nh; i++) {
    const hs = dec.map((l) => l.hops[i]).filter(Boolean);
    R.hopSummary.push({ hop: i + 1, total: r2(med(hs.map((x) => x.deliver + x.queue + x.compute + x.unpack + x.pack + x.send + x.wire))), gpu: r2(med(hs.map((x) => x.gpu))),
      wire: r2(med(hs.map((x) => x.wire))), readback: r2(med(hs.map((x) => x.readback))), encode: r2(med(hs.map((x) => x.encode))), jsPackUnpack: r2(med(hs.map((x) => x.unpack + x.pack))),
      deliverQueueSend: r2(med(hs.map((x) => x.deliver + x.queue + x.send))), submits: r2(med(hs.map((x) => x.submits))), maps: r2(med(hs.map((x) => x.maps))) });
  }
  // host time outside the lap, per token (plain) or per step (spec)
  if (run.mode === "plain") {
    const between = []; for (let i = 1; i < dec.length; i++) between.push(dec[i].pos === dec[i - 1].pos + 1 ? (ev.host["h.lap0"]["ai-hidden|" + dec[i].pos][0] - get("host", "h.ret", "ai-hidden", dec[i - 1].pos)) : null);
    R.perToken = { tokenMs: r2(med(dec.slice(1).map((l, i) => ev.host["h.lap0"]["ai-hidden|" + l.pos][0] - ev.host["h.lap0"]["ai-hidden|" + dec[i].pos][0]))), lapMs: r2(med(dec.map((l) => l.lap))),
      outsideLapMs: r2(med(between)), headMs: r2(med(dec.map((l) => l.head))) };
  } else {
    const steps = Object.entries(ev.host["h.step0"] || {}).map(([k, [t, K]]) => { const [src, p] = k.split("|"); const e1 = Object.entries(ev.host["h.step1"] || {}).find(([k2, v]) => v[0] > t); return { src, pos: +p, K, t0: t, t1: e1 ? e1[1][0] : null, toks: e1 ? e1[1][1] : null }; })
      .filter((s) => s.t1).sort((a, b) => a.t0 - b.t0);
    for (const s of steps) {
      const lap = dec.find((l) => l.pos === s.pos);
      s.lap = lap ? lap.lap : null; s.cols = lap?.cols;
      const g = gpuIn("host", s.t0, s.t1); s.hostGpuInStep = g.gpu; s.hostSubmits = g.n; s.hostMaps = mapsIn("host", s.t0, s.t1).length;
      s.ms = s.t1 - s.t0;
    }
    const byCols = {};
    for (const s of steps) (byCols[s.cols] ||= []).push(s);
    R.perStep = Object.entries(byCols).map(([c, ss]) => ({ verifyCols: c, steps: ss.length, stepMs: r2(med(ss.map((s) => s.ms))), lapMs: r2(med(ss.map((s) => s.lap))), outsideLapMs: r2(med(ss.map((s) => s.ms - s.lap))),
      tokPerStep: r2(mean(ss.map((s) => s.toks))), hostSubmits: r2(med(ss.map((s) => s.hostSubmits))), hostMaps: r2(med(ss.map((s) => s.hostMaps))), hostGpuMs: r2(med(ss.map((s) => s.hostGpuInStep))),
      bytesPerHop: 2 * (J.model === "qwen3.8-27b" ? 5120 : 2048) * c }));
    R.byColsHops = Object.entries(byCols).map(([c]) => {
      const ls = dec.filter((l) => String(l.cols) === c);
      return { verifyCols: c, lap: r2(med(ls.map((l) => l.lap))), hostCompute: r2(med(ls.map((l) => l.hostCompute))), hop1Gpu: r2(med(ls.map((l) => l.hops[0]?.gpu))), hop1Wire: r2(med(ls.map((l) => l.hops[0]?.wire))),
        hop1Readback: r2(med(ls.map((l) => l.hops[0]?.readback))), retWire: r2(med(ls.map((l) => l.retWire))) };
    });
  }
  report.runs.push(R);
}
if (asJson) console.log(JSON.stringify(report, null, 1));
else {
  console.log(`${report.model}, ${report.devices} devices, wire ${report.wire}\n${report.split}`);
  for (const R of report.runs) {
    console.log(`\n## ${R.mode}: ${R.laps} laps · ${R.status}\n${R.crumb || ""}`);
    console.log(table(R.segments, ["segment", "median", "mean"]));
    console.log(table(R.hopSummary, Object.keys(R.hopSummary[0] || { hop: 0 })));
    if (R.perToken) console.log(table([R.perToken], Object.keys(R.perToken)));
    if (R.perStep) { console.log(table(R.perStep, Object.keys(R.perStep[0] || { verifyCols: 0 }))); console.log(table(R.byColsHops, Object.keys(R.byColsHops[0] || { verifyCols: 0 }))); }
  }
}
