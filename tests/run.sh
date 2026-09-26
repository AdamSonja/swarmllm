#!/usr/bin/env bash
# GPU test runner. Usage: tests/run.sh [quick|q38|q38once|all]
# quick: small-model goldens (Qwen3 0.6B / SmolLM); q38: 27B suites, one process per file;
# q38once: the same 27B checks in one process sharing one upload (run_q38_once.js); all: quick + q38.
# The 27B tests load through the converted-weights cache (tests/weight_cache.js, docs/testing-fast.md):
# WEIGHT_CACHE=0 disables it, WEIGHT_CACHE=<dir> moves it.
set -uo pipefail
cd "$(dirname "$0")"
WC="${WEIGHT_CACHE:-$HOME/.cache/swarmllm-weights}"
D="deno run --unstable-webgpu --allow-read --allow-env --allow-write=$WC"
quick=(test_qwen.js test_smollm.js test_stream.js test_batch.js test_reset.js test_qwen_split.js test_qwen_stream.js test_batch_split.js)
q38=(test_q38.js test_batch_q38.js test_mtp.js test_b4.js test_twins.js test_gemm.js test_q38_split.js test_mtp_split.js test_ctx.js)
case "${1:-quick}" in quick) list=("${quick[@]}");; q38) list=("${q38[@]}");; all) list=("${quick[@]}" "${q38[@]}");;
  q38once) $D run_q38_once.js 2>&1 | grep -v "^TU:\|^MESA"; exit "${PIPESTATUS[0]}";;
  *) echo "unknown suite"; exit 2;; esac
fail=0
for t in "${list[@]}"; do
  echo "=== $t"
  if ! $D "$t" 2>&1 | grep -v "^TU:\|^MESA" | tail -4; then fail=1; fi
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
done
exit $fail
