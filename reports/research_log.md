# 연구 로그

## 2026-08-09: Mock 연구 기반 구축 및 본실험 전 설계 감사

- Python 3.14.6, uv 0.11.21, Windows 11 환경에서 저장소를 구성했다.
- `uv run pytest -q` 결과: 13 passed, 0 failed.
- `uv run python -m compass_ai mock-e2e --root .` 결과: Mock 질문 20개, 교사 원본 응답 38개, 독립 검증 뒤 레코드 42개(사람 보정 4개 포함), 병합 충돌은 agree 8·answer_conflict 6·one_missing 2·format_conflict 2·both_suspect 2건이었다.
- 생성 조건 수: S0 0, S1 20, S2 18, S3a 38, S3b 20, S4_teacher_only 16, S4_with_human_correction 20, S5 0이다. S3b는 정렬된 question_id 교대 규칙으로 A 11/B 9를 선택했으며, 누락 교사 응답 1건에서 A 대체가 발생했다.
- S4_with_human_correction의 human_corrected 비율은 20%(4/20)이다. 이 조건의 결과는 독립 검증 효과와 사람 보정 효과를 분리해 해석해야 한다.
- S5는 한국어 원천 데이터가 없으므로 `placeholder` 상태이며, S4 복사 조건이나 실험 결과가 아니다.
- 주실험 matched comparison은 S3_matched 16개와 S4_teacher_only 16개로 동결했다. 두 조건의 question_id 집합과 영역 분포는 같고, S3_matched는 품질 비의존 결정론 규칙으로 MockTeacherA 8개/MockTeacherB 8개를 선택했다.
- 집 데스크톱 준비 단계로 실제 파일럿 템플릿, 품질 검사, manifest checksum, pre-DGX 점검을 추가했다. 실제 `pilot_questions.jsonl`은 의도적으로 아직 없으므로 pre-DGX 결과는 오류가 아닌 `not_ready: actual pilot dataset missing`이다.
- `data/questions/mock_questions.jsonl`과 `data/verified/mock_verified.jsonl`의 JSONL validation을 통과했고, private_final 접근 차단도 통과했다.

Mock 결과는 소프트웨어 파이프라인 구현 검증이며 실제 AI 모델 성능 결과가 아니다.

실제 연구 전에는 모델·생성 응답·원천 데이터의 라이선스를 확인하고, DGX Spark에서 실제 교사 백엔드를 파일럿한 뒤 비공개 평가셋을 학습과 분리해 제작한다.
