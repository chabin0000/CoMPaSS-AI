# Teacher Pilot 실행 규격

Pilot-v1의 checksum은 `data/questions/pilot/pilot_manifest.json`의 `source_file_sha256`로 고정한다. 두 교사는 같은 질문 원문, system prompt, temperature, top_p, max_new_tokens, seed(지원 시), num_repeats를 사용한다. checkpoint·revision·quantization·inference engine·chat template·reasoning/output 형식 차이는 실행 로그에 기록하며, 한 모델만 위한 추가 prompt는 금지한다.

`configs/teacher_pilot.template.json`의 값은 DGX 호환성 확인 전 **proposed**다. 실제 실행 직전 final config를 별도 파일로 저장하고 checksum을 생성한다. checkpoint/revision/engine은 확인 전 `null`을 유지한다.

순서: Pilot checksum 확인 → GPT-OSS load·60문항 실행·raw/log 검증 → 완전 종료·메모리 확인 → GLM load·동일 실행·raw/log 검증 → 종료. 중간에 dataset/config가 달라지면 실패 또는 새 experiment version으로 기록한다.

Raw JSONL은 `data/raw/gpt_oss/` 및 `data/raw/glm/`에 별도로 저장한다. 각 행은 run_id, dataset/config checksum, 시작·종료 시각, latency/TTFT/token(측정 불가 시 null), response_status, error, metadata를 가진다. 오류 없는 완료 ID만 resume 시 건너뛴다. 실제 파일이 둘 다 생긴 뒤에만 comparison-readiness가 dataset checksum, question ID 집합, config reference, 문항 수를 검사한다.
