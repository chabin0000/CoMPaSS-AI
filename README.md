# CoMPaSS-AI

다중 교사 지식 통합과 독립 검증을 통해 한국형 학생 AI를 연구하기 위한 재현 가능한 데이터 파이프라인이다.

현재 구현은 **Mock 교사 데이터로 소프트웨어 흐름을 검증**한다. 실제 모델 다운로드, LoRA/QLoRA 학습, 비공개 최종 평가 문항은 포함하지 않는다.

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

- `data/questions/`: 질문 JSONL, `data/raw/`: 교사별 원본 응답, `data/verified/`: 독립 검증 결과
- `data/conditions/`: S0~S5 학습 조건, `evaluation/private_final/`: 실제 문항 없이 보호 규칙만 둔 위치
- `src/compass_ai/`: 스키마, Mock 교사, 병합, 검증, 조건 구성, 누출 검사, 로그, CLI
- `logs/`: 실행마다 생성되는 재현성 로그(추적 제외), `reports/`: 요약 결과
- `docs/`: 실험 프로토콜과 DGX 실행 지침

## 결과 위치

`mock-e2e` 후 `data/raw/`, `data/merged/mock_merged.jsonl`, `data/verified/mock_verified.jsonl`, `data/conditions/S1.jsonl`, `S2.jsonl`, `S3a.jsonl`, `S3b.jsonl`, `S4_teacher_only.jsonl`, `S4_with_human_correction.jsonl`, `reports/mock_summary.json`, `logs/<run_id>/`가 생성된다. 한국어 원천 데이터가 없으면 `S5.jsonl`은 빈 placeholder이며 실험 결과가 아니다.

주실험용 파일은 `data/conditions/S3_matched.jsonl`과 `S4_teacher_only.jsonl`이다. E2E는 두 파일의 question_id 집합·표본 수·영역 분포가 다르면 실패한다.

## DGX Spark에서 다음에 할 일

먼저 `uv sync --dev`와 `uv run python -m compass_ai system-check --root .`를 실행한다. 라이선스·체크포인트·추론 엔진 호환성을 문서로 확인한 뒤에만 실제 `GPTOSSBackend`, `GLMBackend`를 연결한다. 두 교사는 동시에 적재하지 않고 gpt-oss 응답 저장 및 프로세스 종료 후 GLM을 순차 실행한다.

## 실제 파일럿 작성

`data/questions/pilot/pilot_questions.template.jsonl`을 참고해 `data/questions/pilot/pilot_questions.jsonl`에 실제 문항을 작성한다. 이후 `uv run python -m compass_ai pilot-check --root .`로 품질을 확인하고, `pre-dgx-check`가 manifest를 생성한다. 파일럿은 별도 승인 전 학습 데이터로 전환되지 않는다.
