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

// ---- expert GEMV kernels: configurable layout (MOE_DEFAULT below; MOE_LEGACY reproduces the first coop build) ----
//
// A workgroup of WG threads is cut into WG / TPR row groups of TPR threads. Each group owns R consecutive
// rows (R accumulators per thread per matrix), so a workgroup covers RPW = (WG / TPR) * R rows and a
// (column, slot) pair needs ceil(dOut / RPW) workgroups in x. Inside a group:
//   wide = false: 4 threads share a 32-weight block (one Q4 word / two Q8 words each), TPR / 4 blocks per step
//   wide = true:  one thread owns a whole block, loaded as one vec4<u32> (Q4, 16 B) or two (Q8, 32 B),
//                 TPR blocks per step, with 16 B loads (research D4)
// U unrolls the block loop (U independent block loads in flight per row); tail steps past nb are clamped to
// the last block and dropped with select(), so every load stays in bounds and in one basic block.
// xsh = true stages the (column, slot)'s input vector in workgroup memory once, transposed to
// [slot-of-8][block] with a padded stride (nb + 1), so threads reading consecutive blocks hit different banks.
// The partial sums are reduced by a halving tree over the TPR lanes of each group (log2 TPR barriers).
//
// Every (column, slot) pair runs the same code whatever the pass width, so the batched path (verify /
// prefill) is bit-identical to the one-token path for any config. Different configs sum in different
// orders, so they give different (equally valid) MoE bits; the MoE goldens are llama.cpp text.
export const MOE_LEGACY = Object.freeze({
  gu: Object.freeze({ WG: 256, TPR: 256, R: 4, U: 1, wide: false, xsh: false }),
  dn: Object.freeze({ WG: 64, TPR: 64, R: 4, U: 1, wide: false, xsh: false }),
});
export const MOE_DEFAULT = Object.freeze({
  // Picked by tests/bench/moe_kernel_sweep.js on the GB10 (2026-09-26): in-model (prof_ts) moe_gu_q4 69.8 -> 60.7 µs,
  // moe_dn_q4 43.0 -> 38.4 µs vs legacy. The first guess (gu 128/16/2/2, dn 128/4/2/4) was no faster than legacy.
  // gate/up (512 x 2048 per expert): 32 threads x 16 B per row, 1 row per group, 4 rows per workgroup
  gu: Object.freeze({ WG: 128, TPR: 32, R: 1, U: 1, wide: true, xsh: true }),
  // down (2048 x 512 per expert, 16 blocks per row): 8 threads x 2 blocks per row, 16 rows per workgroup
  dn: Object.freeze({ WG: 128, TPR: 8, R: 1, U: 1, wide: true, xsh: true }),
});
const WG_MEM = 16384;   // WebGPU default maxComputeWorkgroupStorageSize
const pow2 = (n) => Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;

// opt: undefined | "default" | "legacy" | { gu?: {...}, dn?: {...} } (partial overrides of MOE_DEFAULT,
// or of MOE_LEGACY when opt.base === "legacy"). dims: { dim, inter } (inter = expert FFN width).
// Returns { gu, dn } with every field resolved plus rows (= RPW) and the baked input width dIn.
export function moeKernelConfig(opt, { dim, inter }) {
  if (typeof opt === "string") opt = opt === "legacy" ? { base: "legacy" } : opt === "default" ? {} : (() => { throw new Error(`moeKernel: unknown preset ${opt}`); })();
  opt = opt || {};
  const base = opt.base === "legacy" ? MOE_LEGACY : MOE_DEFAULT;
  const one = (kind, dIn) => {
    const c = { ...base[kind], ...(opt[kind] || {}) };
    const { WG, TPR, R, U } = c;
    c.wide = !!c.wide; c.xsh = !!c.xsh;
    if (!pow2(WG) || WG > 256 || WG < 4) throw new Error(`moeKernel.${kind}.WG must be a power of two in [4, 256] (got ${WG})`);
    if (!pow2(TPR) || TPR > WG || (!c.wide && TPR < 4)) throw new Error(`moeKernel.${kind}.TPR must be a power of two <= WG${c.wide ? "" : " and >= 4"} (got ${TPR})`);
    if (!Number.isInteger(R) || R < 1 || R > TPR || R > 8) throw new Error(`moeKernel.${kind}.R must be an integer in [1, min(8, TPR)] (got ${R})`);
    if (!Number.isInteger(U) || U < 1 || U > 8) throw new Error(`moeKernel.${kind}.U must be an integer in [1, 8] (got ${U})`);
    if (dIn % 32) throw new Error(`MoE ${kind} input width ${dIn} is not a multiple of 32`);
    const NA = kind === "gu" ? 2 : 1, red = NA * R * WG * 4, xs = 8 * (dIn / 32 + 1) * 16;
    if (red > WG_MEM) throw new Error(`moeKernel.${kind}: ${red} B of reduction scratch exceeds ${WG_MEM} B of workgroup memory`);
    if (c.xsh && red + xs > WG_MEM) c.xsh = false;   // input too wide to stage: read it from the storage buffer
    c.rows = (WG / TPR) * R; c.dIn = dIn;
    return c;
  };
  return { gu: one("gu", dim), dn: one("dn", inter) };
}

// Code emitters. js = true emits a JavaScript generator body with the same control flow and index math
// (tests/unit/moe_kernels_test.js runs it on the CPU); the WGSL-only constructs go through these helpers.
function emit(js) {
  return {
    q4lo: (w) => js ? `q4lo(${w})` : `(vec4<f32>(unpack4xU8(${w} & 0x0F0F0F0Fu)) - vec4<f32>(8.0))`,
    q4hi: (w) => js ? `q4hi(${w})` : `(vec4<f32>(unpack4xU8((${w} >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0))`,
    i8x4: (w) => js ? `i8x4(${w})` : `vec4<f32>(unpack4xI8(${w}))`,
    div: (a, b) => js ? `Math.floor((${a}) / (${b}))` : `((${a}) / (${b}))`,
  };
}
export function wgslToJs(body) {
  return body.replace(/\bvar (\w+): [^=;]+ = /g, "let $1 = ").replace(/\bvar (\w+) = /g, "let $1 = ")
    .replace(/\b(0x[0-9A-Fa-f]+)u\b/g, "$1").replace(/\b(\d+)u\b/g, "$1").replace(/workgroupBarrier\(\);/g, "yield;");
}

// One expert GEMV kernel. kind "gu": h[cs][row] = silu(Wg[e] x) * (Wu[e] x); kind "dn": y[cs][row] = Wd[e] h[cs].
// Returns { name, decl, body, P }: decl = bindings + workgroup arrays (WGSL only), body = the function body.
export function expertKernel(kind, fmt, c, js = false) {
  const E = emit(js);
  const P = `m${kind === "gu" ? "g" : "d"}${fmt}`, name = `moe_${kind}_${fmt}`;
  const { WG, TPR, R, U, wide, xsh } = c;
  const LANES = wide ? TPR : TPR / 4, RPW = (WG / TPR) * R, NA = kind === "gu" ? 2 : 1;
  const mats = kind === "gu" ? [["g", `${P}_gq`, `${P}_gs`], ["u", `${P}_uq`, `${P}_us`]] : [["y", `${P}_q`, `${P}_sc`]];
  const X = `${P}_x`, XS = `${P}_xs`, RED = `${P}_red`;
  const rs = Array.from({ length: R }, (_, r) => r);
  const xl = (s) => xsh ? `${XS}[(${s}) * nbp + b]` : `${X}[xc + b * 8u + ${s}]`;
  const scale = (SC, r) => `unpack2x16float(${SC}[(er${r} * nb + b) >> 1u])[(er${r} * nb + b) & 1u]`;
  // one row's contribution of block b
  const term = (Q, SC, r, m) => {
    const sc = scale(SC, r);
    if (!wide && fmt === "q4") return `${sc} * (dot(${E.q4lo(`${Q}[(er${r} * nb + b) * 4u + qt]`)}, xa) + dot(${E.q4hi(`${Q}[(er${r} * nb + b) * 4u + qt]`)}, xb))`;
    if (!wide) return `${sc} * (dot(${E.i8x4(`${Q}[(er${r} * nb + b) * 8u + qt * 2u]`)}, xa) + dot(${E.i8x4(`${Q}[(er${r} * nb + b) * 8u + qt * 2u + 1u]`)}, xb))`;
    const w = `w${m}${r}`;
    if (fmt === "q4") {
      const p = (i) => `(dot(${E.q4lo(`${w}[${i}u]`)}, x${i}) + dot(${E.q4hi(`${w}[${i}u]`)}, x${i + 4}))`;
      return `${sc} * ((${p(0)} + ${p(1)}) + (${p(2)} + ${p(3)}))`;
    }
    const d = (v, i, s) => `dot(${E.i8x4(`${v}[${i}u]`)}, x${s})`;
    const h = (v, o) => `((${d(v, 0, o)} + ${d(v, 1, o + 1)}) + (${d(v, 2, o + 2)} + ${d(v, 3, o + 3)}))`;
    return `${sc} * (${h(`${w}a`, 0)} + ${h(`${w}b`, 4)})`;
  };
  const wload = (Q, r, m) => !wide ? "" : fmt === "q4" ? `let w${m}${r} = ${Q}[er${r} * nb + b];`
    : `let w${m}${r}a = ${Q}[(er${r} * nb + b) * 2u]; let w${m}${r}b = ${Q}[(er${r} * nb + b) * 2u + 1u];`;
  const xloads = wide ? Array.from({ length: 8 }, (_, s) => `let x${s} = ${xl(`${s}u`)};`).join(" ")
    : fmt === "q4" ? `let xa = ${xl("qt")}; let xb = ${xl("qt + 4u")};` : `let xa = ${xl("qt * 2u")}; let xb = ${xl("qt * 2u + 1u")};`;
  const step = (u) => {
    const head = U === 1 ? `let b = b0;` : `let bu = b0 + ${u * LANES}u; let ok = bu < nb; let b = min(bu, nb - 1u);`;
    const loads = mats.map(([m, Q]) => rs.map((r) => wload(Q, r, m)).join(" ")).join(" ");
    const adds = mats.map(([m, Q, SC]) => rs.map((r) => U === 1 ? `      ${m}${r} += ${term(Q, SC, r, m)};` : `      ${m}${r} += select(0.0, ${term(Q, SC, r, m)}, ok);`).join("\n")).join("\n");
    return `    {\n      ${head}\n      ${xloads}\n      ${loads}\n${adds}\n    }`;
  };
  const k = (mi, r) => (mi * R + r) * WG;
  const out = kind === "gu"
    ? `let gg = ${RED}[lane * ${WG}u + grp * ${TPR}u]; ${P}_h[cs * S.ys + row] = gg / (1.0 + exp(-gg)) * ${RED}[(${R}u + lane) * ${WG}u + grp * ${TPR}u];`
    : `${P}_y[cs * S.ys + row] = ${RED}[lane * ${WG}u + grp * ${TPR}u];`;
  const body = `
  let S = ${P}_s; let t = lid.x; let cs = wg.y; let lane = t % ${TPR}u; let grp = ${E.div("t", `${TPR}u`)};
  let e = ${P}_sel[cs]; let nb = ${E.div("S.dIn", "32u")}; let row0 = wg.x * ${RPW}u + grp * ${R}u;
  let xc = ${kind === "gu" ? `${E.div("cs", "S.K")} * ${E.div("S.xs", "4u")}` : `cs * ${E.div("S.xs", "4u")}`};   // ${kind === "gu" ? "the column's x" : "each (column, slot) has its own input h"}
${wide ? "" : "  let qt = lane & 3u; let bl = lane >> 2u;\n"}${xsh ? `  let nbp = nb + 1u;
  for (var i: u32 = t; i < nb * 8u; i += ${WG}u) { ${XS}[(i & 7u) * nbp + (i >> 3u)] = ${X}[xc + i]; }
  workgroupBarrier();
` : ""}${mats.map(([m]) => rs.map((r) => `  var ${m}${r}: f32 = 0.0;`).join("\n")).join("\n")}
${rs.map((r) => `  let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`).join("\n")}
  for (var b0: u32 = ${wide ? "lane" : "bl"}; b0 < nb; b0 += ${LANES * U}u) {
${Array.from({ length: U }, (_, u) => step(u)).join("\n")}
  }
${mats.map(([m], mi) => rs.map((r) => `  ${RED}[${k(mi, r)}u + t] = ${m}${r};`).join("\n")).join("\n")}
  workgroupBarrier();
${TPR === 1 ? "" : `  for (var st: u32 = ${TPR / 2}u; st > 0u; st >>= 1u) {
    if (lane < st) {
${mats.map((_, mi) => rs.map((r) => `      ${RED}[${k(mi, r)}u + t] += ${RED}[${k(mi, r)}u + t + st];`).join("\n")).join("\n")}
    }
    workgroupBarrier();
  }
`}  if (lane < ${R}u) { let row = row0 + lane; if (row < S.dOut) { ${out} } }
`;
  const qT = wide ? "array<vec4<u32>>" : "array<u32>";
  const decl = kind === "gu" ? `
@group(1) @binding(0) var<storage, read> ${P}_gq: ${qT};
@group(1) @binding(1) var<storage, read> ${P}_gs: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uq: ${qT};
@group(1) @binding(3) var<storage, read> ${P}_us: array<u32>;
@group(1) @binding(4) var<storage, read> ${X}: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> ${P}_h: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(7) var<uniform> ${P}_s: MOE;` : `
@group(1) @binding(0) var<storage, read> ${P}_q: ${qT};
@group(1) @binding(1) var<storage, read> ${P}_sc: array<u32>;
@group(1) @binding(2) var<storage, read> ${X}: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(4) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(5) var<uniform> ${P}_s: MOE;`;
  const wgDecl = `
var<workgroup> ${RED}: array<f32, ${NA * R * WG}>;${xsh ? `\nvar<workgroup> ${XS}: array<vec4<f32>, ${8 * (c.dIn / 32 + 1)}>;` : ""}`;
  return { name, P, decl: decl + wgDecl, body: js ? wgslToJs(body) : body, wgslBody: body };
}

function expertWGSL(kind, fmt, c) {
  const { name, decl, body } = expertKernel(kind, fmt, c);
  return `${decl}
@compute @workgroup_size(${c.WG})
fn ${name}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {${body}}
`;
}

// cfg: moeKernelConfig(...) result (MOE_DEFAULT at dim 2048, expert width 512 when omitted)
export function moeWGSL(cfg = moeKernelConfig(undefined, { dim: 2048, inter: 512 })) {
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
${expertWGSL("gu", "q4", cfg.gu)}
${expertWGSL("gu", "q8", cfg.gu)}
${expertWGSL("dn", "q4", cfg.dn)}
${expertWGSL("dn", "q8", cfg.dn)}

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
