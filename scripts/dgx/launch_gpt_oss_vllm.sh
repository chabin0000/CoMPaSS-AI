#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required. Run 'hf auth whoami' and export HF_TOKEN without printing it.}"
command -v vllm >/dev/null 2>&1 || {
  echo "ERROR: vLLM executable not found" >&2
  exit 1
}

GPT_OSS_MODEL_ID="${GPT_OSS_MODEL_ID:-openai/gpt-oss-120b}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "Starting GPT-OSS teacher server"
echo "source model: ${GPT_OSS_MODEL_ID}"
echo "served model: openai/gpt-oss-120b"
echo "endpoint: http://127.0.0.1:${VLLM_PORT}/v1"
echo "gpu-memory-utilization: 0.7"

# 모델 파일은 vLLM과 Hugging Face cache가 필요할 때 처리한다.
# 실험 generation parameter는 teacher config가 관리하며 launcher에서 변경하지 않는다.
exec vllm serve "${GPT_OSS_MODEL_ID}" \
  --served-model-name "openai/gpt-oss-120b" \
  --host "127.0.0.1" \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization 0.7
