// Mixture-of-experts FFN kernels (Qwen3.5 / 3.6 MoE: top-k routed experts + one shared expert).
//
// Experts are stored as the GGUF stacks them, [nExp][dOut][dIn], and uploaded exactly like any
// Q4_0 / Q8_0 matrix with nExp * dOut rows, so expert e's row r is row e * dOut + r: the kernels
// only add an offset, read from the routing result on the GPU (no readback to pick experts).
//
// One launch covers every (column, slot) pair: workgroup y = col * K + slot. Each pair's math is
// the same whether a pass has 1 column (decode) or many (verify / prefill), so the batched path
// gives the same bits as the one-token path, which keeps speculative decoding exact.
//
//   moe_router      logits [col][nExp] -> sel [col][K] (expert ids), selw [col][K] (weights)
//   moe_gu_{q4,q8}  h[col,slot] = silu(Wgate[e] x[col]) * (Wup[e] x[col])
//   moe_dn_{q4,q8}  y[col,slot] = Wdown[e] h[col,slot]
//   moe_combine     x[col] += sum_k selw[col,k] * y[col,k] + sigmoid(sg[col]) * shared[col]

const ROWS = 4;

// Expert GEMV kernels, laid out like the dense cooperative GEMV (coop.js): four threads share a 32-weight
// block (one Q4 word or two Q8 words each), x is read once per block as two vec4s and reused for every row
// in the workgroup, and dequant is unpack4xU8 / unpack4xI8 into vec4 dots. Workgroup y = (column, slot):
// every (column, slot) pair does the same math whether a pass has 1 column or many, so the batched path
// stays bit-identical to the one-token path.
const q4lo = (w) => `vec4<f32>(unpack4xU8(${w} & 0x0F0F0F0Fu)) - vec4<f32>(8.0)`;
const q4hi = (w) => `vec4<f32>(unpack4xU8((${w} >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0)`;
const i8x4 = (w) => `vec4<f32>(unpack4xI8(${w}))`;
// one thread's share of block b of expert row er: x from vec4 array X at vec4 offset xc
function term(fmt, Q, SC, X, er, xc) {
  const sc = `unpack2x16float(${SC}[(${er} * nb + b) >> 1u])[(${er} * nb + b) & 1u]`;
  if (fmt === "q4") return `${sc} * (dot(${q4lo(`${Q}[(${er} * nb + b) * 4u + qt]`)}, ${X}[${xc} + b * 8u + qt]) + dot(${q4hi(`${Q}[(${er} * nb + b) * 4u + qt]`)}, ${X}[${xc} + b * 8u + qt + 4u]))`;
  return `${sc} * (dot(${i8x4(`${Q}[(${er} * nb + b) * 8u + qt * 2u]`)}, ${X}[${xc} + b * 8u + qt * 2u]) + dot(${i8x4(`${Q}[(${er} * nb + b) * 8u + qt * 2u + 1u]`)}, ${X}[${xc} + b * 8u + qt * 2u + 1u]))`;
}
const tree = (WG, n, red) => `
  workgroupBarrier();
  for (var st: u32 = ${WG / 2}u; st > 0u; st >>= 1u) {
    if (t < st) {
${Array.from({ length: n }, (_, r) => `      ${red}[${r * WG}u + t] += ${red}[${r * WG}u + t + st];`).join("\n")}
    }
    workgroupBarrier();
  }`;

function guKernel(fmt, WG) {
  const P = `mg${fmt}`, LANES = WG / 4;
  const rows = (f) => Array.from({ length: ROWS }, (_, r) => f(r)).join("\n");
  return `
@group(1) @binding(0) var<storage, read> ${P}_gq: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_gs: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uq: array<u32>;
@group(1) @binding(3) var<storage, read> ${P}_us: array<u32>;
@group(1) @binding(4) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> ${P}_h: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(7) var<uniform> ${P}_s: MOE;
var<workgroup> ${P}_red: array<f32, ${2 * ROWS * WG}>;
@compute @workgroup_size(${WG})
fn moe_gu_${fmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y; let qt = t & 3u; let bl = t >> 2u;
  let e = ${P}_sel[cs]; let nb = S.dIn / 32u; let row0 = wg.x * ${ROWS}u; let xc = (cs / S.K) * (S.xs / 4u);
${rows((r) => `  var g${r}: f32 = 0.0; var u${r}: f32 = 0.0; let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`)}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${rows((r) => `    g${r} += ${term(fmt, `${P}_gq`, `${P}_gs`, `${P}_x`, `er${r}`, "xc")};\n    u${r} += ${term(fmt, `${P}_uq`, `${P}_us`, `${P}_x`, `er${r}`, "xc")};`)}
  }
${rows((r) => `  ${P}_red[${r * WG}u + t] = g${r}; ${P}_red[${(ROWS + r) * WG}u + t] = u${r};`)}
${tree(WG, 2 * ROWS, `${P}_red`)}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < S.dOut) { let gg = ${P}_red[t * ${WG}u]; ${P}_h[cs * S.ys + row] = gg / (1.0 + exp(-gg)) * ${P}_red[(${ROWS}u + t) * ${WG}u]; }
  }
}`;
}

function dnKernel(fmt, WG) {
  const P = `md${fmt}`, LANES = WG / 4;
  const rows = (f) => Array.from({ length: ROWS }, (_, r) => f(r)).join("\n");
  return `
@group(1) @binding(0) var<storage, read> ${P}_q: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_sc: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(4) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(5) var<uniform> ${P}_s: MOE;
var<workgroup> ${P}_red: array<f32, ${ROWS * WG}>;
@compute @workgroup_size(${WG})
fn moe_dn_${fmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y; let qt = t & 3u; let bl = t >> 2u;
  let e = ${P}_sel[cs]; let nb = S.dIn / 32u; let row0 = wg.x * ${ROWS}u; let xc = cs * (S.xs / 4u);   // each (column, slot) has its own input h
${rows((r) => `  var y${r}: f32 = 0.0; let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`)}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${rows((r) => `    y${r} += ${term(fmt, `${P}_q`, `${P}_sc`, `${P}_x`, `er${r}`, "xc")};`)}
  }
${rows((r) => `  ${P}_red[${r * WG}u + t] = y${r};`)}
${tree(WG, ROWS, `${P}_red`)}
  if (t < ${ROWS}u) { let row = row0 + t; if (row < S.dOut) { ${P}_y[cs * S.ys + row] = ${P}_red[t * ${WG}u]; } }
}`;
}

// ---------------- fused MoE FFN (moeFuse, the default when every MoE layer qualifies) ----------------
// Per layer and column: router GEMV (the shared-expert gate row appended as row nExp, one launch),
// moe_route (softmax, top-K, sigmoid of the shared gate), moe_gus (K routed slots + the shared expert as
// slot K, gate/up + SiLU), moe_dnc (down for all K + 1 slots, combine and residual add in one epilogue).
// Slot K reads the shared expert's own weights, which may be in another format than the routed experts
// (Q8_0 shared, Q4_0 routed in the bartowski Q4_0 file): the branch on the slot index is uniform per
// workgroup. K is baked into the kernels (KS = K + 1 slots). As before, workgroup y = (column, slot) or
// column, so every column does the same math in a 1-column pass as in an N-column pass.
//
// Uniform MOEF: dOut, dIn (routed matrix), sDim (shared expert FFN width: rows for gate/up, dIn for down),
// nExp, xs (input / residual column stride, floats), ys (h stride per (column, slot), floats), norm,
// shared (unused, 1), oUq / oGs / oUs (word offsets of the shared up qs, gate scales, up scales in the
// packed shared gate/up buffer, whose gate qs start at 0), pad.

// termOff: like term(), with word offsets into Q / SC and explicit block count / index names
function termOff(fmt, Q, qo, SC, so, X, er, xc, nb = "nb", b = "b") {
  const bi = `(${er} * ${nb} + ${b})`;
  const sc = `unpack2x16float(${SC}[${so} + (${bi} >> 1u)])[${bi} & 1u]`;
  if (fmt === "q4") return `${sc} * (dot(${q4lo(`${Q}[${qo} + ${bi} * 4u + qt]`)}, ${X}[${xc} + ${b} * 8u + qt]) + dot(${q4hi(`${Q}[${qo} + ${bi} * 4u + qt]`)}, ${X}[${xc} + ${b} * 8u + qt + 4u]))`;
  return `${sc} * (dot(${i8x4(`${Q}[${qo} + ${bi} * 8u + qt * 2u]`)}, ${X}[${xc} + ${b} * 8u + qt * 2u]) + dot(${i8x4(`${Q}[${qo} + ${bi} * 8u + qt * 2u + 1u]`)}, ${X}[${xc} + ${b} * 8u + qt * 2u + 1u]))`;
}

// Router: same softmax as moe_router (max and sum trees, same order), then top-K by rank: thread i counts
// the experts ordered before it, (p_j > p_i) or (p_j == p_i and j < i), and writes itself to that slot if
// it is below K. That is exactly the order of moe_router's K argmax rounds (ties to the lower index), so
// the ids and weights are the same bits, without the 8 x 8 barrier rounds. Slot K gets the shared gate
// sigmoid(logit[nExp]) with moe_combine's expression.
function routeKernel(K) {
  const KS = K + 1;
  return `
@group(1) @binding(0) var<storage, read> rt_l: array<f32>;
@group(1) @binding(1) var<storage, read_write> rt_sel: array<u32>;
@group(1) @binding(2) var<storage, read_write> rt_w: array<f32>;
@group(1) @binding(3) var<uniform> rt_s: MOEF;
var<workgroup> rt_p: array<f32, 1024>;
var<workgroup> rt_v: array<f32, 256>;
var<workgroup> rt_ki: array<u32, ${K}>;
var<workgroup> rt_kv: array<f32, ${K}>;
@compute @workgroup_size(256)
fn moe_route(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wg.x; let t = lid.x; let n = rt_s.nExp;
  let lb = col * rt_s.xs;
  if (t < ${K}u) { rt_ki[t] = 0u; rt_kv[t] = 0.0; }
  var m: f32 = -3.0e38;
  for (var i: u32 = t; i < n; i += 256u) { m = max(m, rt_l[lb + i]); }
  rt_v[t] = m;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) { if (t < st) { rt_v[t] = max(rt_v[t], rt_v[t + st]); } workgroupBarrier(); }
  let mx = rt_v[0];
  workgroupBarrier();
  var s: f32 = 0.0;
  for (var i: u32 = t; i < n; i += 256u) { let p = exp(rt_l[lb + i] - mx); rt_p[i] = p; s += p; }
  rt_v[t] = s;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) { if (t < st) { rt_v[t] += rt_v[t + st]; } workgroupBarrier(); }
  let inv = 1.0 / rt_v[0];
  for (var i: u32 = t; i < n; i += 256u) { rt_p[i] = rt_p[i] * inv; }
  workgroupBarrier();
  for (var i: u32 = t; i < n; i += 256u) {
    let p = rt_p[i];
    var r: u32 = 0u;
    for (var j: u32 = 0u; j < n; j++) { let q = rt_p[j]; r += select(0u, 1u, q > p || (q == p && j < i)); }
    if (r < ${K}u && p == p) { rt_ki[r] = i; rt_kv[r] = p; }
  }
  workgroupBarrier();
  if (t == 0u) {
    var tot: f32 = 0.0;
    for (var k: u32 = 0u; k < ${K}u; k++) { tot += rt_kv[k]; }
    for (var k: u32 = 0u; k < ${K}u; k++) {
      rt_sel[col * ${KS}u + k] = rt_ki[k];
      rt_w[col * ${KS}u + k] = select(rt_kv[k], rt_kv[k] / tot, rt_s.norm == 1u);
    }
    let g = rt_l[lb + n];
    rt_sel[col * ${KS}u + ${K}u] = 0u;
    rt_w[col * ${KS}u + ${K}u] = 1.0 / (1.0 + exp(-g));
  }
}`;
}

// gate/up for K routed slots + the shared expert (slot K). Grid: x = ceil(max(dOut, sDim) / ROWS),
// y = column * KS + slot. Same per-thread layout and reduction as moe_gu.
function gusKernel(fmt, sfmt, K, WG = 256) {
  const KS = K + 1, P = `gs${fmt}${sfmt}`, LANES = WG / 4;
  const rows = (f) => Array.from({ length: ROWS }, (_, r) => f(r)).join("\n");
  return `
@group(1) @binding(0) var<storage, read> ${P}_gq: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_gs: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uq: array<u32>;
@group(1) @binding(3) var<storage, read> ${P}_us: array<u32>;
@group(1) @binding(4) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> ${P}_h: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(7) var<storage, read> ${P}_sh: array<u32>;
@group(1) @binding(8) var<uniform> ${P}_s: MOEF;
var<workgroup> ${P}_red: array<f32, ${2 * ROWS * WG}>;
@compute @workgroup_size(${WG})
fn moe_gus_${fmt}_${sfmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y; let col = cs / ${KS}u; let slot = cs - col * ${KS}u;
  let qt = t & 3u; let bl = t >> 2u; let nb = S.dIn / 32u; let row0 = wg.x * ${ROWS}u; let xc = col * (S.xs / 4u);
${rows((r) => `  var g${r}: f32 = 0.0; var u${r}: f32 = 0.0;`)}
  var dOut = S.dOut;
  if (slot < ${K}u) {
    let e = ${P}_sel[cs];
${rows((r) => `    let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`)}
    for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${rows((r) => `      g${r} += ${termOff(fmt, `${P}_gq`, "0u", `${P}_gs`, "0u", `${P}_x`, `er${r}`, "xc")};\n      u${r} += ${termOff(fmt, `${P}_uq`, "0u", `${P}_us`, "0u", `${P}_x`, `er${r}`, "xc")};`)}
    }
  } else {
    dOut = S.sDim;
${rows((r) => `    let sr${r} = min(row0 + ${r}u, S.sDim - 1u);`)}
    for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${rows((r) => `      g${r} += ${termOff(sfmt, `${P}_sh`, "0u", `${P}_sh`, "S.oGs", `${P}_x`, `sr${r}`, "xc")};\n      u${r} += ${termOff(sfmt, `${P}_sh`, "S.oUq", `${P}_sh`, "S.oUs", `${P}_x`, `sr${r}`, "xc")};`)}
    }
  }
${rows((r) => `  ${P}_red[${r * WG}u + t] = g${r}; ${P}_red[${(ROWS + r) * WG}u + t] = u${r};`)}
${tree(WG, 2 * ROWS, `${P}_red`)}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < dOut) { let gg = ${P}_red[t * ${WG}u]; ${P}_h[cs * S.ys + row] = gg / (1.0 + exp(-gg)) * ${P}_red[(${ROWS}u + t) * ${WG}u]; }
  }
}`;
}

// down for all KS slots of one column + combine + residual: workgroup (row block, column) runs every
// slot's GEMV rows (the same per-thread terms and one tree for all slots), then thread r writes
// x[row] += sum_k w_k y_k (k = 0 .. K - 1 in order) + w_K y_shared, moe_combine's expression and order.
function dncKernel(fmt, sfmt, K, R, WG = 64) {
  const KS = K + 1, P = `dc${fmt}${sfmt}`, LANES = WG / 4;
  const ks = Array.from({ length: K }, (_, k) => k), rs = Array.from({ length: R }, (_, r) => r);
  const acc = (k, r) => `y${k}_${r}`;
  return `
@group(1) @binding(0) var<storage, read> ${P}_q: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_sc: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_h: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> ${P}_x: array<f32>;
@group(1) @binding(4) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(5) var<storage, read> ${P}_w: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_sq: array<u32>;
@group(1) @binding(7) var<storage, read> ${P}_ss: array<u32>;
@group(1) @binding(8) var<uniform> ${P}_s: MOEF;
var<workgroup> ${P}_red: array<f32, ${KS * R * WG}>;
@compute @workgroup_size(${WG})
fn moe_dnc_${fmt}_${sfmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let col = wg.y; let qt = t & 3u; let bl = t >> 2u;
  let nb = S.dIn / 32u; let nbs = S.sDim / 32u; let row0 = wg.x * ${R}u; let hs4 = S.ys / 4u;
${rs.map((r) => `  let rr${r} = min(row0 + ${r}u, S.dOut - 1u);`).join("\n")}
${ks.map((k) => `  let e${k} = ${P}_sel[col * ${KS}u + ${k}u]; let xc${k} = (col * ${KS}u + ${k}u) * hs4;
  ${rs.map((r) => `var ${acc(k, r)}: f32 = 0.0; let er${k}_${r} = e${k} * S.dOut + rr${r};`).join(" ")}`).join("\n")}
  ${rs.map((r) => `var ${acc(K, r)}: f32 = 0.0;`).join(" ")}
  let xcs = (col * ${KS}u + ${K}u) * hs4;
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${ks.map((k) => rs.map((r) => `    ${acc(k, r)} += ${termOff(fmt, `${P}_q`, "0u", `${P}_sc`, "0u", `${P}_h`, `er${k}_${r}`, `xc${k}`)};`).join("\n")).join("\n")}
  }
  for (var b: u32 = bl; b < nbs; b += ${LANES}u) {
${rs.map((r) => `    ${acc(K, r)} += ${termOff(sfmt, `${P}_sq`, "0u", `${P}_ss`, "0u", `${P}_h`, `rr${r}`, "xcs", "nbs")};`).join("\n")}
  }
${Array.from({ length: KS }, (_, k) => rs.map((r) => `  ${P}_red[${(k * R + r) * WG}u + t] = ${acc(k, r)};`).join("\n")).join("\n")}
${tree(WG, KS * R, `${P}_red`)}
  if (t < ${R}u) {
    let row = row0 + t;
    if (row < S.dOut) {
      let wb = col * ${KS}u;
      var o: f32 = 0.0;
      for (var k: u32 = 0u; k < ${K}u; k++) { o += ${P}_w[wb + k] * ${P}_red[(k * ${R}u + t) * ${WG}u]; }
      o += ${P}_w[wb + ${K}u] * ${P}_red[(${K * R}u + t) * ${WG}u];
      ${P}_x[col * S.xs + row] += o;
    }
  }
}`;
}

// The fused kernels for one engine: K (top-k), R (output rows per moe_dnc workgroup), and the
// (routed, shared) format pairs the model's layers need for gate/up and for down.
export function moeFusedWGSL({ K, R = 2, gu = [], dn = [] }) {
  if (!(K >= 1 && K <= 16) || ![1, 2, 4].includes(R)) throw new Error(`moeFusedWGSL: K ${K}, R ${R}`);
  return /* wgsl */ `
// ---------------- fused mixture of experts (engine/wgsl/moe.js moeFusedWGSL) ----------------
struct MOEF { dOut: u32, dIn: u32, sDim: u32, nExp: u32, xs: u32, ys: u32, norm: u32, shared: u32, oUq: u32, oGs: u32, oUs: u32, pad: u32 };
${routeKernel(K)}
${gu.map(([f, s]) => gusKernel(f, s, K)).join("\n")}
${dn.map(([f, s]) => dncKernel(f, s, K, R)).join("\n")}
`;
}

export function moeWGSL() {
  return /* wgsl */ `
// ---------------- mixture of experts (engine/wgsl/moe.js) ----------------
struct MOE { dOut: u32, dIn: u32, K: u32, nExp: u32, xs: u32, ys: u32, norm: u32, pad: u32 };

// Router: softmax over all experts, the K largest probabilities (ties: lower index), their probabilities as
// weights, renormalised to sum 1 when norm = 1 (norm_topk_prob). One workgroup of 256 per column; max and sum
// are tree reductions and each of the K picks is an argmax tree over (probability, index), so every step is
// parallel and the order of every sum is fixed.
@group(1) @binding(0) var<storage, read> mr_l: array<f32>;
@group(1) @binding(1) var<storage, read_write> mr_sel: array<u32>;
@group(1) @binding(2) var<storage, read_write> mr_w: array<f32>;
@group(1) @binding(3) var<uniform> mr_s: MOE;
var<workgroup> mr_p: array<f32, 1024>;
var<workgroup> mr_v: array<f32, 256>;
var<workgroup> mr_i: array<u32, 256>;
var<workgroup> mr_k: array<f32, 16>;
@compute @workgroup_size(256)
fn moe_router(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wg.x; let t = lid.x; let n = mr_s.nExp; let K = mr_s.K;
  let lb = col * mr_s.xs;
  var m: f32 = -3.0e38;
  for (var i: u32 = t; i < n; i += 256u) { m = max(m, mr_l[lb + i]); }
  mr_v[t] = m;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) { if (t < st) { mr_v[t] = max(mr_v[t], mr_v[t + st]); } workgroupBarrier(); }
  let mx = mr_v[0];
  workgroupBarrier();
  var s: f32 = 0.0;
  for (var i: u32 = t; i < n; i += 256u) { let p = exp(mr_l[lb + i] - mx); mr_p[i] = p; s += p; }
  mr_v[t] = s;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) { if (t < st) { mr_v[t] += mr_v[t + st]; } workgroupBarrier(); }
  let inv = 1.0 / mr_v[0];
  for (var i: u32 = t; i < n; i += 256u) { mr_p[i] = mr_p[i] * inv; }
  workgroupBarrier();
  for (var k: u32 = 0u; k < K; k++) {
    var bv: f32 = -1.0; var bi: u32 = 0u;
    for (var i: u32 = t; i < n; i += 256u) { if (mr_p[i] > bv) { bv = mr_p[i]; bi = i; } }
    mr_v[t] = bv; mr_i[t] = bi;
    workgroupBarrier();
    for (var st: u32 = 128u; st > 0u; st >>= 1u) {
      if (t < st) { let ov = mr_v[t + st]; let oi = mr_i[t + st]; if (ov > mr_v[t] || (ov == mr_v[t] && oi < mr_i[t])) { mr_v[t] = ov; mr_i[t] = oi; } }
      workgroupBarrier();
    }
    if (t == 0u) { mr_sel[col * K + k] = mr_i[0]; mr_k[k] = mr_v[0]; mr_p[mr_i[0]] = -2.0; }   // taken
    workgroupBarrier();
  }
  if (t == 0u) {
    var tot: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) { tot += mr_k[k]; }
    for (var k: u32 = 0u; k < K; k++) { mr_w[col * K + k] = select(mr_k[k], mr_k[k] / tot, mr_s.norm == 1u); }
  }
}
${guKernel("q4", 256)}
${guKernel("q8", 256)}
${dnKernel("q4", 64)}
${dnKernel("q8", 64)}

// Combine: x[col] += sum_k w_k y[col,k] (k in order) + sigmoid(sg[col]) * shared[col].
// dOut = dim, xs = x column stride, ys = y (col,slot) stride, nExp (reused) = shared column stride,
// norm (reused) = 1 when there is a shared expert, pad (reused) = shared-gate logit column stride.
@group(1) @binding(0) var<storage, read_write> mc_x: array<f32>;
@group(1) @binding(1) var<storage, read> mc_y: array<f32>;
@group(1) @binding(2) var<storage, read> mc_w: array<f32>;
@group(1) @binding(3) var<storage, read> mc_sh: array<f32>;
@group(1) @binding(4) var<storage, read> mc_sg: array<f32>;
@group(1) @binding(5) var<uniform> mc_s: MOE;
@compute @workgroup_size(64)
fn moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let col = gid.y; let S = mc_s;
  if (i >= S.dOut) { return; }
  var o: f32 = 0.0;
  for (var k: u32 = 0u; k < S.K; k++) { o += mc_w[col * S.K + k] * mc_y[(col * S.K + k) * S.ys + i]; }
  if (S.norm == 1u) {
    let g = mc_sg[col * S.pad];
    o += (1.0 / (1.0 + exp(-g))) * mc_sh[col * S.nExp + i];
  }
  mc_x[col * S.xs + i] += o;
}
`;
}
