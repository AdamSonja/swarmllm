// Expert-grouped MoE FFN for wide prefill ubatches (engine option moeGroupPrefill; engine/qwen35.js
// _prefillGrouped). Candidate (D) of docs/research/prefill-profile-2026-09.md.
//
// The per-(column, slot) fused kernels (engine/wgsl/moe.js moe_gus / moe_dnc) stream an expert's rows once
// for every token that picked it. A 16-token pass picks ~102 distinct experts for 128 assignments, so
// there is little to share inside one pass. The grouped prefill therefore runs layer-major over a ubatch
// of U tokens (U a multiple of the pass width): each layer's attention / DeltaNet part runs as the usual
// 16-column passes (same kernels, same bits), the router output of all U tokens is kept, and then:
//
//   moe_gsort                one workgroup: counting sort of the U * (K + 1) (column, slot) pairs by expert
//                            (the shared expert is expert nExp), stable by pair index, cut into chunks of up
//                            to UC pairs; writes the chunk list and the indirect launch sizes.
//   moe_gusg_{f}_{s}         one workgroup per (row block, chunk): gate/up + SiLU for every pair of the chunk
//   moe_dng_{f}_{s}          one workgroup per (row block, chunk): down projection per pair -> y[pair]
//   moe_combw                x[col] += sum_k w_k y[col, k] (k = 0 .. K - 1 in order) + w_K y[col, K]
//
// Exactness (the whole point): every pair keeps the per-pair kernels' arithmetic. The grouped kernels have the
// same workgroup sizes, the same thread -> (block, quarter) map, build every term from the same template
// (moe.js termW), add the blocks into the pair's own accumulator in the same order, reduce with the same tree
// and apply the same epilogue expressions (moe_combw is moe_dnc's combine, in the same order). Only the weight
// loads are shared between the pairs of a chunk; a load does not change a value. So a token's h, y and x bits
// do not depend on grouping, on the chunk size or on which other tokens are in the ubatch. The layer-major
// order does not change any value either: layer l of token t still sees exactly the inputs it saw before
// (DeltaNet state and KV cache of layer l after the earlier passes of that layer).
//
// Chunk list layout (one u32 buffer "grp"): chunk c at words [4c .. 4c + 3] = (first pair index into the pair
// list, pair count n, expert, 0), then the pair list itself at word CO (= 4 * maxChunks): pair indices
// cs = col * KS + slot, grouped by expert (experts ascending), ascending within an expert.
import { FOPS, termW, scOff, wOff, tree } from "./moe.js";

const ROWS = 4;   // the fused gate/up kernel's rows per workgroup (moe.js ROWS)
const WG_MEM = 16384;

// Sizes for a ubatch of U tokens: pairs, the chunk bound (sum over experts of ceil(count / UC) <=
// pairs / UC + experts), the pair list offset and the buffer size in words.
export function moeGroupSizes({ U, K, nExp, UC }) {
  const pairs = U * (K + 1), maxChunks = Math.ceil(pairs / UC) + nExp + 1, CO = 4 * maxChunks;
  return { pairs, maxChunks, CO, words: CO + pairs };
}

// ---- moe_gsort ----
// Thread t owns the experts e with e % 256 == t (QM = ceil((nExp + 1) / 256) of them). Pass 1 counts each
// owned expert's pairs over the pair list (read in tiles of 1024 through workgroup memory), a scan in expert
// order gives each expert its first chunk and first pair-list slot, pass 2 appends each pair to its expert's
// list in pair order (stable), and the owner writes its experts' chunk records. No atomics: the result is a
// pure function of sel.
function sortKernel(K, UC, QM, CO, O) {
  const KS = K + 1, D = O.div;
  return `
@group(1) @binding(0) var<storage, read> gso_sel: array<u32>;
@group(1) @binding(1) var<storage, read_write> gso_grp: array<u32>;
@group(1) @binding(2) var<storage, read_write> gso_ind: array<u32>;
@group(1) @binding(3) var<uniform> gso_s: MOEG;
var<workgroup> gso_tile: array<u32, 1024>;
var<workgroup> gso_sc: array<u32, 256>;
var<workgroup> gso_sp: array<u32, 256>;
var<workgroup> gso_su: array<u32, 256>;
@compute @workgroup_size(256)
fn moe_gsort(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let N = gso_s.n; let nE = gso_s.nExp;
  var cnt: array<u32, ${QM}>;   // zero-initialized
  var placed: array<u32, ${QM}>;   // zero-initialized
  var cbase: array<u32, ${QM}>;   // zero-initialized
  var pbase: array<u32, ${QM}>;   // zero-initialized
  for (var i0: u32 = 0u; i0 < N; i0 += 1024u) {
    for (var j: u32 = t; j < 1024u; j += 256u) {
      let i = i0 + j;
      if (i < N) { let slot = i % ${KS}u; gso_tile[j] = select(min(gso_sel[i], nE - 1u), nE, slot == ${K}u); }
    }
    workgroupBarrier();
    let m = min(1024u, N - i0);
    for (var j: u32 = 0u; j < m; j++) { let e = gso_tile[j]; if ((e & 255u) == t) { cnt[e >> 8u] += 1u; } }
    workgroupBarrier();
  }
  var carryC: u32 = 0u; var carryP: u32 = 0u; var carryU: u32 = 0u;
  for (var q: u32 = 0u; q < ${QM}u; q++) {
    let c = cnt[q]; let nch = ${D(`c + ${UC - 1}u`, `${UC}u`)};
    gso_sc[t] = nch; gso_sp[t] = c; gso_su[t] = select(0u, 1u, c > 0u && q * 256u + t < nE);
    workgroupBarrier();
    for (var off: u32 = 1u; off < 256u; off <<= 1u) {   // inclusive scan in expert order (Hillis-Steele)
      var a = gso_sc[t]; var b = gso_sp[t]; var u = gso_su[t];
      if (t >= off) { a += gso_sc[t - off]; b += gso_sp[t - off]; u += gso_su[t - off]; }
      workgroupBarrier();
      gso_sc[t] = a; gso_sp[t] = b; gso_su[t] = u;
      workgroupBarrier();
    }
    cbase[q] = carryC + gso_sc[t] - nch; pbase[q] = carryP + gso_sp[t] - c;
    carryC += gso_sc[255]; carryP += gso_sp[255]; carryU += gso_su[255];
    workgroupBarrier();
  }
  for (var i0: u32 = 0u; i0 < N; i0 += 1024u) {
    for (var j: u32 = t; j < 1024u; j += 256u) {
      let i = i0 + j;
      if (i < N) { let slot = i % ${KS}u; gso_tile[j] = select(min(gso_sel[i], nE - 1u), nE, slot == ${K}u); }
    }
    workgroupBarrier();
    let m = min(1024u, N - i0);
    for (var j: u32 = 0u; j < m; j++) {
      let e = gso_tile[j];
      if ((e & 255u) == t) { let q = e >> 8u; gso_grp[${CO}u + pbase[q] + placed[q]] = i0 + j; placed[q] += 1u; }
    }
    workgroupBarrier();
  }
  for (var q: u32 = 0u; q < ${QM}u; q++) {
    let e = q * 256u + t; let c = cnt[q];
    for (var k: u32 = 0u; k * ${UC}u < c; k++) {
      let ci = (cbase[q] + k) * 4u;
      gso_grp[ci] = pbase[q] + k * ${UC}u; gso_grp[ci + 1u] = min(${UC}u, c - k * ${UC}u); gso_grp[ci + 2u] = e; gso_grp[ci + 3u] = 0u;
    }
  }
  if (t == 0u) {
    gso_ind[0] = gso_s.gx; gso_ind[1] = carryC; gso_ind[2] = 1u;
    gso_ind[3] = gso_s.dx; gso_ind[4] = carryC; gso_ind[5] = 1u;
    gso_ind[6] = carryU; gso_ind[7] = N;
  }
}`;
}

// the chunk header shared by the grouped kernels: n (workgroup-uniform), first pair, expert, and the chunk's
// pair indices cs0 .. cs{UC-1} (absent pairs alias the chunk's last pair and are never written)
function chunkHead(P, UC, CO) {
  return `  let cb = wg.y * 4u;
  if (t == 0u) { ${P}_n = ${P}_grp[cb + 1u]; }
  let n = workgroupUniformLoad(&${P}_n);
  let p0 = ${P}_grp[cb]; let e = ${P}_grp[cb + 2u];
${Array.from({ length: UC }, (_, u) => `  let cs${u} = ${P}_grp[${CO}u + p0${u ? ` + min(${u}u, n - 1u)` : ""}];`).join("\n")}`;
}
const users = (UC, f) => Array.from({ length: UC }, (_, u) => f(u)).join("\n");
const guard = (u, body) => u ? `if (n > ${u}u) { ${body} }` : `{ ${body} }`;

// ---- moe_gusg: gate/up + SiLU for a chunk of pairs of one expert (routed, or the shared expert e == nExp) ----
// Per pair: moe_gus's accumulators g_r / u_r, its block loop, its tree over 2 * ROWS arrays and its epilogue.
// The reduction scratch holds PP pairs at a time (8 KB per pair at WG 256), so the pairs reduce in turn.
export function gusGroupKernel(fmt, sfmt, K, UC, CO, WG = 256, O = FOPS) {
  const KS = K + 1, P = `gg${fmt}${sfmt}`, LANES = WG / 4, D = O.div, NACC = 2 * ROWS;
  const PP = Math.max(1, Math.min(UC, Math.floor((WG_MEM - 16) / (NACC * WG * 4))));
  const rows = (f) => Array.from({ length: ROWS }, (_, r) => f(r)).join("\n");
  const ld = (f, pre, Q, qo, SC, so, er) => rows((r) => { const [w0, w1] = wOff(f, Q, qo, `${er}${r}`);
    return `      let ${pre}w${r} = ${w0};${w1 ? ` let ${pre}v${r} = ${w1};` : ""} let ${pre}s${r} = ${scOff(SC, so, `${er}${r}`)};`; });
  const acc = (f) => users(UC, (u) => `      ${guard(u, rows((r) => `g${u}_${r} += ${termW(f, `gs${r}`, `gw${r}`, `gv${r}`, `${P}_x`, `xc${u}`, "b", O)}; u${u}_${r} += ${termW(f, `us${r}`, `uw${r}`, `uv${r}`, `${P}_x`, `xc${u}`, "b", O)};`).replace(/\n/g, " "))}`);
  let red = "";
  for (let q0 = 0; q0 < UC; q0 += PP) {
    const us = Array.from({ length: Math.min(PP, UC - q0) }, (_, i) => q0 + i);
    const body = `${us.map((u, i) => rows((r) => `  ${P}_red[${(i * NACC + r) * WG}u + t] = g${u}_${r}; ${P}_red[${(i * NACC + ROWS + r) * WG}u + t] = u${u}_${r};`)).join("\n")}
${tree(WG, NACC * us.length, `${P}_red`)}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < dOut) {
${us.map((u, i) => `      ${guard(u, `let gg = ${P}_red[(${i * NACC}u + t) * ${WG}u]; ${P}_h[cs${u} * S.ys + row] = gg / (1.0 + exp(-gg)) * ${P}_red[(${i * NACC + ROWS}u + t) * ${WG}u];`)}`).join("\n")}
    }
  }
  workgroupBarrier();`;
    red += q0 ? `\n  if (n > ${q0}u) {\n${body}\n  }` : `\n${body}`;
  }
  return `
@group(1) @binding(0) var<storage, read> ${P}_gq: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_gs: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uq: array<u32>;
@group(1) @binding(3) var<storage, read> ${P}_us: array<u32>;
@group(1) @binding(4) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> ${P}_h: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_grp: array<u32>;
@group(1) @binding(7) var<storage, read> ${P}_sh: array<u32>;
@group(1) @binding(8) var<uniform> ${P}_s: MOEF;
var<workgroup> ${P}_red: array<f32, ${NACC * PP * WG}>;
var<workgroup> ${P}_n: u32;
@compute @workgroup_size(${WG})
fn moe_gusg_${fmt}_${sfmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x;
  let qt = t & 3u; let bl = t >> 2u; let nb = ${D("S.dIn", "32u")}; let row0 = wg.x * ${ROWS}u;
${chunkHead(P, UC, CO)}
${users(UC, (u) => `  let xc${u} = (${D(`cs${u}`, `${KS}u`)}) * (${D("S.xs", "4u")});`)}
${users(UC, (u) => rows((r) => `  var g${u}_${r}: f32 = 0.0; var u${u}_${r}: f32 = 0.0;`))}
  var dOut = S.dOut;
  if (e < S.nExp) {
${rows((r) => `    let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`)}
    for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${ld(fmt, "g", `${P}_gq`, "0u", `${P}_gs`, "0u", "er")}
${ld(fmt, "u", `${P}_uq`, "0u", `${P}_us`, "0u", "er")}
${acc(fmt)}
    }
  } else {
    dOut = S.sDim;
${rows((r) => `    let sr${r} = min(row0 + ${r}u, S.sDim - 1u);`)}
    for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${ld(sfmt, "g", `${P}_sh`, "0u", `${P}_sh`, "S.oGs", "sr")}
${ld(sfmt, "u", `${P}_sh`, "S.oUq", `${P}_sh`, "S.oUs", "sr")}
${acc(sfmt)}
    }
  }${red}
}`;
}

// ---- moe_dng: down projection for a chunk of pairs -> y[cs][row] (moe_dnc's per-slot sums, before combine) ----
export function dnGroupKernel(fmt, sfmt, K, R, UC, CO, WG = 64, O = FOPS) {
  const P = `dg${fmt}${sfmt}`, LANES = WG / 4, D = O.div;
  const rs = Array.from({ length: R }, (_, r) => r), rows = (f) => rs.map(f).join("\n");
  if (UC * R * WG * 4 > WG_MEM - 16) throw new Error(`moe_dng: ${UC} pairs x ${R} rows need more than ${WG_MEM} B of workgroup memory`);
  const ld = (f, Q, SC, er, nb) => rows((r) => { const [w0, w1] = wOff(f, Q, "0u", `${er}${r}`, nb);
    return `      let w${r} = ${w0};${w1 ? ` let v${r} = ${w1};` : ""} let s${r} = ${scOff(SC, "0u", `${er}${r}`, nb)};`; });
  const acc = (f) => users(UC, (u) => `      ${guard(u, rows((r) => `y${u}_${r} += ${termW(f, `s${r}`, `w${r}`, `v${r}`, `${P}_h`, `xc${u}`, "b", O)};`).replace(/\n/g, " "))}`);
  return `
@group(1) @binding(0) var<storage, read> ${P}_q: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_sc: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_h: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(4) var<storage, read> ${P}_grp: array<u32>;
@group(1) @binding(5) var<storage, read> ${P}_sq: array<u32>;
@group(1) @binding(6) var<storage, read> ${P}_ss: array<u32>;
@group(1) @binding(7) var<uniform> ${P}_s: MOEF;
var<workgroup> ${P}_red: array<f32, ${UC * R * WG}>;
var<workgroup> ${P}_n: u32;
@compute @workgroup_size(${WG})
fn moe_dng_${fmt}_${sfmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let qt = t & 3u; let bl = t >> 2u;
  let nb = ${D("S.dIn", "32u")}; let nbs = ${D("S.sDim", "32u")}; let row0 = wg.x * ${R}u; let hs4 = ${D("S.ys", "4u")};
${chunkHead(P, UC, CO)}
${rows((r) => `  let rr${r} = min(row0 + ${r}u, S.dOut - 1u);`)}
${users(UC, (u) => `  let xc${u} = cs${u} * hs4;`)}
${users(UC, (u) => `  ${rs.map((r) => `var y${u}_${r}: f32 = 0.0;`).join(" ")}`)}
  if (e < S.nExp) {
${rows((r) => `    let er${r} = e * S.dOut + rr${r};`)}
    for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${ld(fmt, `${P}_q`, `${P}_sc`, "er", "nb")}
${acc(fmt)}
    }
  } else {
    for (var b: u32 = bl; b < nbs; b += ${LANES}u) {
${ld(sfmt, `${P}_sq`, `${P}_ss`, "rr", "nbs")}
${acc(sfmt)}
    }
  }
${users(UC, (u) => rows((r) => `  ${P}_red[${(u * R + r) * WG}u + t] = y${u}_${r};`))}
${tree(WG, UC * R, `${P}_red`)}
  if (t < ${R}u) {
    let row = row0 + t;
    if (row < S.dOut) {
${users(UC, (u) => `      ${guard(u, `${P}_y[cs${u} * S.dOut + row] = ${P}_red[(${u * R}u + t) * ${WG}u];`)}`)}
    }
  }
}`;
}

// ---- moe_combw: moe_dnc's combine + residual, per (row, column), over the grouped y ----
export function combKernel(K) {
  const KS = K + 1;
  return `
@group(1) @binding(0) var<storage, read_write> gcb_x: array<f32>;
@group(1) @binding(1) var<storage, read> gcb_y: array<f32>;
@group(1) @binding(2) var<storage, read> gcb_w: array<f32>;
@group(1) @binding(3) var<uniform> gcb_s: MOEF;
@compute @workgroup_size(64)
fn moe_combw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let S = gcb_s; let row = gid.x; let col = gid.y;
  if (row >= S.dOut) { return; }
  let wb = col * ${KS}u;
  var o: f32 = 0.0;
  for (var k: u32 = 0u; k < ${K}u; k++) { o += gcb_w[wb + k] * gcb_y[(wb + k) * S.dOut + row]; }
  o += gcb_w[wb + ${K}u] * gcb_y[(wb + ${K}u) * S.dOut + row];
  gcb_x[col * S.xs + row] += o;
}`;
}

// Rows per moe_dng workgroup. A row's value does not depend on it (each row has its own accumulators and tree
// over the same thread map, exactly as moe_dnc's rows at any moeDnRows), so it is picked for speed: more rows
// per workgroup share each pair's h loads. The reduction scratch (UC * R * 64 floats) must fit in 16 KB.
export const dnGroupRows = (UC) => (UC <= 8 ? 4 : 2);

// The grouped kernels for one engine (appended after moeFusedWGSL, whose struct MOEF they use).
// K: top-k; R: rows per moe_dng workgroup (dnGroupRows(UC)); UC: pairs per chunk;
// U: ubatch width in tokens; nExp; gu / dn: the (routed, shared) format pairs.
export function moeGroupWGSL({ K, R = 1, UC = 8, U, nExp, gu = [], dn = [] }, O = FOPS) {
  if (!(K >= 1 && K <= 16) || ![1, 2, 4].includes(R)) throw new Error(`moeGroupWGSL: K ${K}, R ${R}`);
  if (![1, 2, 4, 8, 16].includes(UC)) throw new Error(`moeGroupWGSL: UC ${UC} is not 1, 2, 4, 8 or 16`);
  const { CO } = moeGroupSizes({ U, K, nExp, UC }), QM = Math.ceil((nExp + 1) / 256);
  return /* wgsl */ `
// ---------------- expert-grouped MoE prefill (engine/wgsl/moe_group.js) ----------------
struct MOEG { n: u32, nExp: u32, gx: u32, dx: u32 };
${sortKernel(K, UC, QM, CO, O)}
${gu.map(([f, s]) => gusGroupKernel(f, s, K, UC, CO, 256, O)).join("\n")}
${dn.map(([f, s]) => dnGroupKernel(f, s, K, R, UC, CO, 64, O)).join("\n")}
${combKernel(K)}
`;
}
