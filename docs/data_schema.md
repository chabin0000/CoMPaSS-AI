# 데이터 스키마

질문 JSONL 필수 필드는 `question_id`, `split`, `domain`, `task_type`, `question`, `reference_answer`, `verification_method`, `evidence_source`, `difficulty`, `requires_current_info`, `notes`이다.

교사 응답 JSONL은 위 질문 맥락과 함께 `teacher_model`, `checkpoint`, `quantization`, `generation_config`, `raw_response`, `final_answer`, `verification_status`, `verification_method`, `evidence`, `error_type`, `tag`, `reviewer`, `verification_outcome`, `timestamp`, `training_approved`를 보존한다. 추가 필드는 허용한다.

- `teacher_raw`: 생성 직후, 아직 학습 승인 전
- `teacher_verified`: 독립 검증 통과
- `human_corrected`: 사람이 수정한 최종 답
- `korean_source`: 검증된 한국어 원천 데이터
- `excluded`: 검증 실패 또는 제외 대상

`verification_outcome`은 최소 `teacher_a_selected`, `teacher_b_selected`, `both_agree_verified`, `human_corrected`, `excluded`를 사용한다. `human_corrected` 표본은 교사 답을 단순 선택한 표본과 별도로 집계한다.

현재 자동 실행되는 검증은 `answer_key`이다. `calculation`, `unit_check`, `code_execution`, `official_source`, `human_review`는 상태와 근거를 보존하는 확장 지점이며 실제 연구 전 각각의 검증기를 추가해야 한다.

실제 교사 응답에는 가능한 경우 `revision`, `inference_engine`, `question_dataset_checksum`, `run_id`, `seed`, `environment_log_reference`를 추가한다. 값이 없으면 추측하거나 Mock 값으로 채우지 않는다.
