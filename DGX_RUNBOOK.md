# ARGOS-v1 DGX 실행 Runbook

아래 명령은 저장소 루트에서 실행한다. `RUN_ID`는 한 실험 동안 유지하며, 새 실험에서는 새 값을 쓴다. 서버 실행 명령은 DGX 확인 뒤 vLLM/SGLang/NIM 중 선택한다. 모든 Linux DGX 실행 블록은 첫 블록의 `set -euo pipefail`이 유지되는 같은 shell에서 실행한다. 한 단계라도 실패하면 다음 단계로 자동 진행하지 않고, 원인을 해결한 뒤 같은 `RUN_ID`로 실패한 단계부터 재개한다.

현재 pre-DGX 작업트리는 이 Runbook 보완 후 사용자가 commit하고 `pre-dgx-argos-v1` tag로 고정한다. 현재 미커밋 상태 자체는 코드 오류가 아니다. 목요일에는 아래 확인이 모두 통과한 frozen 버전에서만 실행한다.

```bash
set -euo pipefail

export EXPECTED_TAG="pre-dgx-argos-v1"
git rev-parse HEAD
git status --short
test -z "$(git status --short)" || { echo "ERROR: working tree is not clean" >&2; exit 1; }
git describe --tags --exact-match
test "$(git describe --tags --exact-match)" = "$EXPECTED_TAG" || { echo "ERROR: expected tag $EXPECTED_TAG" >&2; exit 1; }

uv sync --python 3.11 --dev --extra student
# QLoRA를 시험할 때만 별도 환경에서: uv sync --python 3.11 --dev --extra student --extra student-qlora
export RUN_ID="argos-v1-$(date -u +%Y%m%dT%H%M%SZ)"

uv run python -m compass_ai dgx-system-report --root . --report-dir "artifacts/dgx_runs/$RUN_ID/system/report"
uv run python -m compass_ai dgx-plan --root . --config configs/dgx/argos_v1.yaml
uv run python -m compass_ai pre-dgx-final --root .
```

## Teacher A: GPT-OSS

OpenAI-compatible localhost 서버를 별도 터미널에서 시작한 뒤 실행한다.

```bash
uv run python -m compass_ai teacher-check --root . --config configs/teachers/gpt_oss_dgx.yaml --smoke --output-root "artifacts/dgx_runs/$RUN_ID/teachers/smoke"
uv run python -m compass_ai run-pilot --root . --teacher gpt_oss --config configs/teachers/gpt_oss_dgx.yaml --output-root "artifacts/dgx_runs/$RUN_ID/teachers"
uv run python -m compass_ai teacher-check --root . --config configs/teachers/gpt_oss_core_v2_dgx.yaml --questions data/questions/core_v2/train_questions.jsonl
uv run python -m compass_ai run-teacher-dataset --root . --teacher gpt_oss --config configs/teachers/gpt_oss_core_v2_dgx.yaml --questions data/questions/core_v2/train_questions.jsonl --output-root "artifacts/dgx_runs/$RUN_ID/teachers"
```

GPT 서버를 정상 종료하고 메모리 해제를 확인한 뒤 GLM 서버를 시작한다.

## Teacher B: GLM-4.5-Air

```bash
uv run python -m compass_ai teacher-check --root . --config configs/teachers/glm45_air_dgx.yaml --smoke --output-root "artifacts/dgx_runs/$RUN_ID/teachers/smoke"
uv run python -m compass_ai run-pilot --root . --teacher glm45_air --config configs/teachers/glm45_air_dgx.yaml --output-root "artifacts/dgx_runs/$RUN_ID/teachers"
uv run python -m compass_ai teacher-check --root . --config configs/teachers/glm45_air_core_v2_dgx.yaml --questions data/questions/core_v2/train_questions.jsonl
uv run python -m compass_ai run-teacher-dataset --root . --teacher glm45_air --config configs/teachers/glm45_air_core_v2_dgx.yaml --questions data/questions/core_v2/train_questions.jsonl --output-root "artifacts/dgx_runs/$RUN_ID/teachers"
```

## 독립 검증과 수동 검토

```bash
export GPT_RUN="artifacts/dgx_runs/$RUN_ID/teachers/gpt_oss/core-v2-train-gpt-oss-dgx/responses.jsonl"
export GLM_RUN="artifacts/dgx_runs/$RUN_ID/teachers/glm45_air/core-v2-train-glm45-air-dgx/responses.jsonl"
export TRAIN_SHA="22652b4f1e34e7f3ec9bbc0fa18c3114204e8dce4d6171b3a49967ead51f4715"

uv run python -m compass_ai comparison-readiness --left "$GPT_RUN" --right "$GLM_RUN" --dataset-checksum "$TRAIN_SHA"
uv run python -m compass_ai verification-readiness --questions data/questions/core_v2/train_questions.jsonl --left "$GPT_RUN" --right "$GLM_RUN" --dataset-checksum "$TRAIN_SHA" --output-root "artifacts/dgx_runs/$RUN_ID/verification/initial"
```

`needs_review`이면 teacher 다수결로 넘기지 않는다. 같은 `RUN_ID`를 유지하고 `artifacts/dgx_runs/$RUN_ID/verification/manual_review.jsonl`에 unresolved 항목만 한 줄당 하나의 JSON으로 기록한다.

```json
{"question_id":"...","decision":"accept_a|accept_b|reject|human_corrected","reviewer":"...","evidence":"...","final_answer":"human_corrected일 때만"}
```

`decision`이 `human_corrected`일 때만 `final_answer`를 넣는다. 파일을 저장한 뒤 다음 명령으로 검토 적용 단계부터 재개한다.

```bash
uv run python -m compass_ai apply-verification-review --questions data/questions/core_v2/train_questions.jsonl --verification-file "artifacts/dgx_runs/$RUN_ID/verification/initial/verification_records.jsonl" --eligible-file "artifacts/dgx_runs/$RUN_ID/verification/initial/eligible_teacher_only.jsonl" --review-file "artifacts/dgx_runs/$RUN_ID/verification/manual_review.jsonl" --output-root "artifacts/dgx_runs/$RUN_ID/verification/reviewed"
```

검토를 적용했다면 `export VERIFIED_DIR="reviewed"`, 처음부터 `ready`였다면 `export VERIFIED_DIR="initial"`로 둔다. 이후 명령도 처음 설정한 같은 `RUN_ID`를 사용한다.

## 조건 데이터와 학생 학습

S5는 1차 핵심 본실험에서 실행하지 않는다. 기존 config/schema는 후속 확장 실험 placeholder로만 보존한다. 실행 순서는 아래 배열로 고정하며 glob 순서에 의존하지 않는다.

```bash
uv run python -m compass_ai build-core-conditions --root . --left "$GPT_RUN" --right "$GLM_RUN" --teacher-eligible "artifacts/dgx_runs/$RUN_ID/verification/$VERIFIED_DIR/eligible_teacher_only.jsonl" --human-eligible "artifacts/dgx_runs/$RUN_ID/verification/$VERIFIED_DIR/eligible_with_human_correction.jsonl" --output-root "artifacts/dgx_runs/$RUN_ID/student_datasets"

CONDITIONS=(s0 s1 s2 s3a s3b s3_matched s4_teacher_only s4_with_human_correction)
for CONDITION in "${CONDITIONS[@]}"; do
  CFG="artifacts/dgx_runs/$RUN_ID/student_datasets/configs/$CONDITION.json"
  uv run python -m compass_ai student-preflight --root . --config "$CFG" --runtime-model-check
done
for CONDITION in "${CONDITIONS[@]}"; do
  CFG="artifacts/dgx_runs/$RUN_ID/student_datasets/configs/$CONDITION.json"
  uv run python -m compass_ai train-student --root . --config "$CFG"
done
```

## Private 평가와 결과표

private key는 별도 전달된 `artifacts/private/core_v2_private_key.jsonl`만 사용한다. 생성되는 runtime 정답 파일은 Git에 넣지 않는다.

```bash
uv run python -m compass_ai prepare-core-evaluation --root . --config "artifacts/dgx_runs/$RUN_ID/student_datasets/configs" --output-root "artifacts/dgx_runs/$RUN_ID/evaluation/private"
for CONDITION in "${CONDITIONS[@]}"; do
  CFG="artifacts/dgx_runs/$RUN_ID/evaluation/private/configs/$CONDITION.json"
  uv run python -m compass_ai evaluate-student --root . --config "$CFG"
done
uv run python -m compass_ai collect-core-results --config "artifacts/dgx_runs/$RUN_ID/evaluation/private/configs" --evaluation-file "artifacts/dgx_runs/$RUN_ID/evaluation/private/private_runtime.jsonl" --output-root "artifacts/dgx_runs/$RUN_ID/evaluation/scored" --run-id "$RUN_ID"
export SCORED_DIR="scored"
```

rubric 항목이 `needs_review`이면 같은 `RUN_ID`를 유지하고 blind reviewer가 `artifacts/dgx_runs/$RUN_ID/evaluation/blind_rubric_review.jsonl`에 unresolved 항목만 기록한다. 형식은 `{"condition":"S4_teacher_only","evaluation_id":"...","correct":true}`이며 `correct`는 JSON boolean `true` 또는 `false`다. 저장 후 다음 명령으로 결과 수집 단계부터 새 output 경로에 재개한다.

```bash
uv run python -m compass_ai collect-core-results --config "artifacts/dgx_runs/$RUN_ID/evaluation/private/configs" --evaluation-file "artifacts/dgx_runs/$RUN_ID/evaluation/private/private_runtime.jsonl" --review-file "artifacts/dgx_runs/$RUN_ID/evaluation/blind_rubric_review.jsonl" --output-root "artifacts/dgx_runs/$RUN_ID/evaluation/scored_reviewed" --run-id "$RUN_ID"
export SCORED_DIR="scored_reviewed"
```

수동 검토가 없으면 `SCORED_DIR=scored`, 있었다면 `SCORED_DIR=scored_reviewed`인 상태에서 분석하고 외부 저장소로 백업한다. 실제 외부 경로는 DGX에서 `BACKUP_DEST`에 입력한다. 빈 값이거나 존재하지 않는 디렉터리이면 최종 백업 단계에서 중단한다.

```bash
uv run python -m compass_ai analyze-results --input "artifacts/dgx_runs/$RUN_ID/evaluation/$SCORED_DIR" --output "artifacts/dgx_runs/$RUN_ID/summary/analysis"

export BACKUP_DEST="${BACKUP_DEST:-}"
test -n "$BACKUP_DEST" || { echo "ERROR: set BACKUP_DEST to external storage" >&2; exit 1; }
test -d "$BACKUP_DEST" || { echo "ERROR: BACKUP_DEST is not an existing directory" >&2; exit 1; }
tar -czf "$RUN_ID-artifacts.tar.gz" "artifacts/dgx_runs/$RUN_ID"
cp "$RUN_ID-artifacts.tar.gz" "$BACKUP_DEST/"
cmp "$RUN_ID-artifacts.tar.gz" "$BACKUP_DEST/$RUN_ID-artifacts.tar.gz"
echo "Backup verified: $BACKUP_DEST/$RUN_ID-artifacts.tar.gz"
```

## 실패 시 재개 지점

- GPT/GLM 서버 실패: `/v1/models`와 model ID를 확인하고 같은 teacher 명령을 재실행한다.
- CUDA/bf16: `dgx-system-report`와 `student-preflight` 결과를 확인하고 하드웨어 문제가 해결되기 전 학습하지 않는다.
- bitsandbytes/QLoRA: 별도 QLoRA 환경이 실패하면 기본 student 환경으로 돌아간다. `quantization: auto`가 bf16 LoRA를 선택한 이유를 metadata로 확인하고 임의 조건 변경은 하지 않는다.
- EXAONE target mismatch: runtime preflight의 실제 module 목록과 config를 대조한다. 자동 변경하지 않는다.
- Pilot/Core teacher 중단: 같은 `RUN_ID`·config로 재실행하면 완료 ID를 보존한다.
- manual review: unresolved만 사람이 독립 검토하고 review artifact부터 재개한다.
- student OOM: 실패 metadata를 보존하고 batch/sequence/accumulation 변경은 새 config와 새 output 경로로 기록한다.
- disk 부족: artifact를 백업·이동한 뒤 같은 단계에서 재개하며 기존 run을 삭제하거나 덮어쓰지 않는다.
