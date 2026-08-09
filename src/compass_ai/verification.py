"""독립 검증의 공통 진입점. 현재 answer_key만 자동 실행한다."""

from __future__ import annotations

from datetime import UTC, datetime

from .comparison import normalize_answer


def _human_correction(question: dict) -> dict:
    """검증 실패 시 사람 검수 기록의 구조를 Mock으로 재현한다.

    실제 연구에서는 사람이 작성한 근거·검수자를 반드시 기록하며, 이 표본은 교사 전용 S4와 분리한다.
    """
    return {
        "question_id": question["question_id"], "domain": question["domain"], "task_type": question["task_type"],
        "question": question["question"], "teacher_model": "HumanReviewer", "checkpoint": "human-correction",
        "quantization": "not_applicable", "generation_config": {"source": "mock answer key"},
        "raw_response": question["reference_answer"], "final_answer": question["reference_answer"],
        "verification_status": "passed", "verification_method": question["verification_method"],
        "evidence": f"human_corrected_from_answer_key:{question['reference_answer']}", "error_type": None,
        "tag": "human_corrected", "reviewer": "mock_reviewer", "verification_outcome": "human_corrected",
        "timestamp": datetime.now(UTC).isoformat(), "training_approved": True,
    }


def verify_responses(records: list[dict], questions: list[dict]) -> list[dict]:
    keys = {item["question_id"]: item for item in questions}
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["question_id"], []).append(dict(record))
    verified: list[dict] = []
    for question_id, candidates in grouped.items():
        question = keys[question_id]
        if question["verification_method"] != "answer_key":
            for item in candidates:
                item.update({"verification_status": "requires_review", "tag": "teacher_raw", "verification_outcome": "excluded", "training_approved": False})
            verified.extend(candidates)
            continue
        passed = []
        for item in candidates:
            is_correct = item.get("error_type") is None and normalize_answer(item["final_answer"]) == normalize_answer(question["reference_answer"])
            item.update({"verification_status": "passed" if is_correct else "failed", "evidence": f"answer_key:{question['reference_answer']}", "tag": "teacher_verified" if is_correct else "excluded", "verification_outcome": "excluded", "training_approved": False})
            if is_correct:
                passed.append(item)
        if not passed:
            verified.extend(candidates)
            verified.append(_human_correction(question))
            continue
        a = next((item for item in passed if item["teacher_model"] == "MockTeacherA"), None)
        b = next((item for item in passed if item["teacher_model"] == "MockTeacherB"), None)
        selected = a or b or passed[0]
        same_answer = len(passed) >= 2 and len({normalize_answer(item["final_answer"]) for item in passed}) == 1
        selected["verification_outcome"] = "both_agree_verified" if same_answer else ("teacher_a_selected" if selected["teacher_model"] == "MockTeacherA" else "teacher_b_selected")
        selected["training_approved"] = True
        for item in passed:
            if item is not selected:
                item["verification_outcome"] = "both_agree_verified" if same_answer else "excluded"
        verified.extend(candidates)
    return verified


def verification_interface_note() -> dict:
    return {method: "not implemented; preserve record for reviewer" for method in ["calculation", "unit_check", "code_execution", "official_source", "human_review"]}
