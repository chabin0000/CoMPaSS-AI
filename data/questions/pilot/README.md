# 실제 파일럿 질문

실제 질문은 `pilot_questions.jsonl`에만 작성한다. Pilot-v1은 60문항으로 동결하며, manifest의 checksum이 reference answer와 rubric을 포함한 이 파일의 기준점이다.

Mock 질문을 복사하거나 `mock-` ID를 사용하면 검증이 실패한다. 파일럿은 자동으로 학습 데이터가 되지 않는다. 정치 문항의 `response_status`는 질문 파일이 아니라 이후 교사 응답 레코드에 `answered`, `partial`, `evasive`, `refusal`, `no_answer` 중 하나로 기록한다.
