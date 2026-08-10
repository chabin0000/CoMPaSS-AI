# DGX Spark 실행 순서

본실험용 복사·붙여넣기 명령과 실패별 재개 절차의 canonical 문서는 저장소 루트 `DGX_RUNBOOK.md`이다. 이 파일은 기존 pre-DGX 호환 경로로 유지한다.

집 데스크톱: 개발·파일럿 준비 → `uv run python -m compass_ai pre-dgx-check --root .` → Git commit/push.

게이밍 노트북: 저장소 clone/pull → 필요 시 같은 pre-DGX 검사 → 학교로 이동.

DGX: 저장소 clone/pull → `uv sync --dev` → `uv run python -m compass_ai pre-dgx-check --root .` → `pilot_manifest.json`의 `source_file_sha256` 확인 → 환경 결과 저장 → 실제 inference backend 결정 → 교사 모델 파일럿 실행.

DGX에서는 코드를 새로 개발하지 않고, 이 저장소에서 검증한 코드 중 GPU 의존 부분만 실행한다.

1. 저장소를 옮긴 뒤 `uv sync --dev`, `uv run pytest -q`, `uv run python -m compass_ai system-check --root .`와 `uv run python -m compass_ai pre-dgx-check --root .`를 실행한다. Pilot-v1의 reference answer/rubric을 수정해야 하면 기존 manifest를 덮어쓰지 말고 dataset version을 올린다.
2. 모델 카드, 생성물 학습·배포 조건, 양자화 체크포인트 제작 주체와 엔진 호환성을 `docs/license_checklist.md`로 확인한다.
3. 실제 백엔드 연결 전 작은 파일럿 질문으로 공식 채팅 템플릿, ARM64/Blackwell, MoE, 컨텍스트 길이와 배치 크기를 확인한다.
4. gpt-oss-120b를 실행해 응답을 저장한 뒤 서버·프로세스를 종료하고 메모리 해제를 확인한다. 그 뒤에만 GLM-4.5-Air를 실행한다.
5. `nvidia-smi`만으로 통합 메모리 사용량을 단정하지 않는다. Linux의 `/proc/meminfo`, 프로세스 로그, 추론 엔진 지표를 함께 저장한다.

실제 `GPTOSSBackend`와 `GLMBackend`는 `TeacherBackend` 인터페이스에 연결할 예정이며, 이번 저장소에는 대형 모델·엔진·학습 코드를 넣지 않는다.
