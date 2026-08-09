# 재현성 기록

각 실행은 `logs/<run_id>/`에 설정, 운영체제·Python·아키텍처·Git 해시·명령이 담긴 환경 정보, 지표, 이벤트를 기록한다. 실제 실험에서는 random seed, checkpoint, quantization, generation settings, dataset version을 반드시 추가한다.

같은 질문 ID와 파일 버전을 보존해야 S1~S5의 차이가 데이터 변경이 아니라 검증 여부인지 확인할 수 있다. 실패도 `events.jsonl`에 남겨 선택적 성공 보고를 막는다.
