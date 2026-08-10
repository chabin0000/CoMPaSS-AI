# CoMPaSS-AI

다중 교사 지식 통합과 독립 검증을 통해 한국형 학생 AI를 연구하기 위한 재현 가능한 데이터 파이프라인이다.

현재 ARGOS-v1은 Mock E2E뿐 아니라 OpenAI-compatible 실제 teacher API, teacher별 중단·재개, 독립 검증과 수동 검토, frozen Core-v2 데이터·평가 프로토콜, EXAONE 학생 LoRA·QLoRA 학습 및 평가 파이프라인, DGX 실행 Runbook까지 pre-DGX 준비가 완료된 상태다. 단, 실제 대형 teacher 추론과 ARGOS-v1 본학습·본평가는 아직 DGX에서 수행하지 않았으므로 현재 결과를 실제 모델 성능으로 해석할 수 없다.

## 실행

```powershell
cd C:\CoMPaSS-AI
uv sync --dev
uv run pytest -q
uv run python -m compass_ai system-check --root .
uv run python -m compass_ai mock-e2e --root .
uv run python -m compass_ai validate --file data/questions/mock_questions.jsonl
uv run python -m compass_ai leakage-check --root .
uv run python -m compass_ai pre-dgx-check --root .
```

## 구조

- `data/questions/`: Pilot-v1 및 frozen Core-v2 질문, `data/raw/`: 교사별 원본 응답, `data/verified/`: 독립 검증 결과
- `data/conditions/`: 학습 조건, `evaluation/protocols/`: frozen 평가 프로토콜, `evaluation/private_final/`: 비공개 평가 보호 경계
- `src/compass_ai/`: Mock/실제 teacher API, 독립 검증, 조건 구성, student LoRA·QLoRA, 평가·분석, 누출 검사와 CLI
- `logs/`: 실행마다 생성되는 재현성 로그(추적 제외), `reports/`: 요약 결과
- `docs/`: 실험 프로토콜과 DGX 실행 지침

## 결과 위치

`mock-e2e` 후 `data/raw/`, `data/merged/mock_merged.jsonl`, `data/verified/mock_verified.jsonl`, 조건 데이터, `reports/mock_summary.json`, `logs/<run_id>/`가 생성된다. S5는 ARGOS-v1 핵심 본실험에서 실행하지 않는 후속 확장용 placeholder이며 실험 결과가 아니다.

주실험용 파일은 `data/conditions/S3_matched.jsonl`과 `S4_teacher_only.jsonl`이다. E2E는 두 파일의 question_id 집합·표본 수·영역 분포가 다르면 실패한다.

## DGX 본실험

DGX에서는 frozen Git tag를 확인한 뒤 GPT-OSS와 GLM을 순차 실행하고, 독립 검증·수동 검토, S0~S4 학생 학습, private 평가, 결과 분석과 외부 artifact 백업을 수행한다. 환경변수, 정확한 실행 순서, 중단·재개 및 manual review 형식은 [`DGX_RUNBOOK.md`](DGX_RUNBOOK.md)를 따른다. GPU/CUDA/bf16, bitsandbytes 4bit, 실제 endpoint, EXAONE runtime과 모델 로딩 성능은 DGX 실행 전까지 미확인 상태다.
