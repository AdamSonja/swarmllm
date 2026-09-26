// Run tests/bench/bench.html in Chrome with the real GPU. node tests/bench/chrome_bench.mjs <model path under repo> [tokens]
// Loading (not part of the tok/s numbers) is made cheap for repeated runs:
//   WCACHE=1 (default): the page takes pre-converted tensors from serve.mjs's weight cache
//     (tests/weight_cache.js, ~/.cache/swarmllm-weights) instead of repacking in JS; WCACHE=0 = old path.
//   CHROME_PROFILE=<dir> (default ~/.cache/swarmllm-chrome-bench): a persistent Chrome profile, so
//     Chrome's on-disk GPU/shader caches survive between runs; CHROME_PROFILE=0 = a fresh profile each run.
// Decode is untouched either way: same page, same engine, same kernels.
import { chromium } from "playwright"; import { spawn } from "node:child_process"; import os from "node:os"; import path from "node:path";
const root = new URL("../..", import.meta.url).pathname, model = process.argv[2], N = process.argv[3] || 40;
const GOLD = { "q36moe": ["```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:", "A hash map is a data structure that stores key-value pairs, allowing for efficient retrieval, insertion, and deletion operations. It uses a hash function to compute an index into an array of buckets or slots"] };
const wcache = process.env.WCACHE !== "0", prof = process.env.CHROME_PROFILE ?? path.join(os.homedir(), ".cache", "swarmllm-chrome-bench");
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, "8791"], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=65536"];
let b, ctx;
if (prof && prof !== "0") { ctx = await chromium.launchPersistentContext(prof, { headless: false, args }); console.log("chrome profile:", prof); }
else { b = await chromium.launch({ headless: false, args }); ctx = await b.newContext(); }
const p = await ctx.newPage(); p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
const gold = GOLD[Object.keys(GOLD).find((k) => model.includes(k))] || [];
await p.goto(`http://127.0.0.1:8791/tests/bench/bench.html?model=/${model}&tokens=${N}&wcache=${wcache ? 1 : 0}&gold=${encodeURIComponent(JSON.stringify(gold))}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 30 * 60e3, polling: 2000 }).catch((e) => console.log("timeout", e.message));
await ctx.close(); if (b) await b.close(); srv.kill();
