#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required. Run 'hf auth whoami' and export HF_TOKEN without printing it.}"
command -v vllm >/dev/null 2>&1 || {
  echo "ERROR: vLLM executable not found" >&2
  exit 1
}

GLM_MODEL_ID="${GLM_MODEL_ID:-zai-org/GLM-4.5-Air}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "Starting GLM-4.5-Air teacher server"
echo "source model: ${GLM_MODEL_ID}"
echo "served model: zai-org/GLM-4.5-Air"
echo "endpoint: http://127.0.0.1:${VLLM_PORT}/v1"

# GLM_MODEL_ID만 local/quantized checkpoint로 바꿀 수 있다.
# 메모리 및 generation parameter는 launcher가 자동 변경하지 않는다.
exec vllm serve "${GLM_MODEL_ID}" \
  --served-model-name "zai-org/GLM-4.5-Air" \
  --host "127.0.0.1" \
  --port "${VLLM_PORT}"
