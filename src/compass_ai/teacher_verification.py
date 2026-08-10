"""실제 teacher run JSONL을 독립 검증 결과와 학생 학습 후보로 변환한다."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .comparison import classify_pair, normalize_answer
from .jsonl import read_jsonl, write_jsonl
from .schemas import validate_questions
from .teacher_run import validate_teacher_raw_responses


EXACT_VERIFICATION_METHODS = {"answer_key", "exact_match"}


def _load_verification_questions(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if rows and rows[0].get("dataset_version") == "core-v2":
        normalized = []
        for row in rows:
            reference = row.get("reference", {})
            rubric = row.get("rubric", {})
            if row.get("split") == "private_eval" or not reference.get("answer"):
                raise ValueError("Core-v2 private_eval/hidden reference cannot be used for teacher verification")
            normalized.append({
                **row,
                "reference_answer": reference["answer"],
                "evidence_source": "core-v2 frozen verification-only reference",
                "rubric": rubric.get("rubric_id"),
                "auto_verifiable": bool(rubric.get("auto_verifiable")),
            })
        return normalized
    return validate_questions(rows)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _raw_reference(path: Path, line: int, row: dict) -> dict:
    return {"path": str(path), "line": line, "record_sha256": _sha256_json(row)}


def _load_side(path: Path, expected_checksum: str) -> tuple[dict[str, dict], dict[str, dict]]:
    rows = validate_teacher_raw_responses(read_jsonl(path)) if path.exists() else []
    by_id: dict[str, dict] = {}
    for line, row in enumerate(rows, 1):
        if row["dataset_checksum"] != expected_checksum:
            raise ValueError(f"teacher dataset checksum mismatch: {path}")
        enriched = dict(row)
        enriched["raw_response_reference"] = _raw_reference(path, line, row)
        by_id[row["question_id"]] = enriched
    error_path = path.parent / "errors.jsonl"
    errors: dict[str, dict] = {}
    if error_path.exists():
        for line, row in enumerate(read_jsonl(error_path), 1):
            if isinstance(row, dict) and isinstance(row.get("question_id"), str):
                errors[row["question_id"]] = {
                    **row,
                    "error_reference": {"path": str(error_path), "line": line, "record_sha256": _sha256_json(row)},
                }
    return by_id, errors


def _candidate_view(row: dict | None, error: dict | None) -> dict:
    if row is not None:
        response_error = row.get("error") if isinstance(row.get("error"), dict) else {}
        return {
            "present": True,
            "teacher_model": row["teacher_model"],
            "run_id": row["run_id"],
            "final_answer": row["final_answer"],
            "response_status": row["response_status"],
            "error_type": row.get("error_type") or response_error.get("type"),
            "raw_response_reference": row["raw_response_reference"],
            "error_reference": None,
        }
    return {
        "present": False,
        "teacher_model": None,
        "run_id": error.get("run_id") if error else None,
        "final_answer": None,
        "response_status": "no_answer",
        "error_type": error.get("error_type") if error else "missing_response",
        "raw_response_reference": None,
        "error_reference": error.get("error_reference") if error else None,
    }


def _evaluate_candidate(candidate: dict, question: dict) -> tuple[str, str]:
    if not candidate["present"]:
        return "rejected", candidate["error_type"] or "missing_response"
    if candidate["response_status"] in {"refusal", "no_answer"} or candidate["error_type"]:
        return "rejected", candidate["error_type"] or candidate["response_status"]
    method = question["verification_method"]
    if method in EXACT_VERIFICATION_METHODS or question.get("auto_verifiable") is True:
        passed = normalize_answer(candidate["final_answer"]) == normalize_answer(question["reference_answer"])
        return ("passed", "independent_reference_exact_match") if passed else ("failed", "reference_mismatch")
    # 전용 checker가 없는 계산/코드/제약/출처/rubric 항목은 정답으로 추정하지 않는다.
    return "unresolved", f"independent_checker_not_implemented:{method}"


def _eligible_record(question: dict, selected: dict, audit: dict, outcome: str) -> dict:
    verifier_metadata = {
        key: value for key, value in audit["verifier_metadata"].items()
        if key not in {"reference_answer_sha256", "rubric_id"}
    }
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "final_answer": selected["final_answer"],
        "teacher_model": selected["teacher_model"],
        "checkpoint": None,
        "revision": None,
        "quantization": None,
        "verification_status": "passed",
        "verification_method": question["verification_method"],
        "verification_result": "verified",
        "verification_outcome": outcome,
        "training_approved": True,
        "tag": "teacher_verified",
        "error_type": None,
        "response_status": "answered",
        "evidence": "independent verification completed; reference content intentionally omitted",
        "conflict_type": audit["conflict_type"],
        "teacher_a_raw_response_reference": audit["teacher_a"]["raw_response_reference"],
        "teacher_b_raw_response_reference": audit["teacher_b"]["raw_response_reference"],
        "rejection_reason": None,
        "verifier_metadata": verifier_metadata,
        "source_run_ids": audit["source_run_ids"],
        "run_id": selected["run_id"],
        "dataset_version": audit["dataset_version"],
        "dataset_checksum": audit["dataset_checksum"],
        "timestamp": audit["verified_at"],
        "domain": question["domain"],
        "task_type": question["task_type"],
    }


def verify_teacher_runs(
    questions_path: Path,
    teacher_a_path: Path,
    teacher_b_path: Path,
    output_dir: Path,
    expected_dataset_checksum: str,
) -> dict:
    """교사 일치 여부와 무관하게 각 답을 reference/method로 독립 검증한다."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"verification output collision: {output_dir}")
    if _sha256_file(questions_path) != expected_dataset_checksum:
        raise ValueError("verification questions checksum does not match teacher dataset checksum")
    questions = _load_verification_questions(questions_path)
    teacher_a, errors_a = _load_side(teacher_a_path, expected_dataset_checksum)
    teacher_b, errors_b = _load_side(teacher_b_path, expected_dataset_checksum)
    allowed_ids = {row["question_id"] for row in questions}
    unknown = (set(teacher_a) | set(teacher_b) | set(errors_a) | set(errors_b)) - allowed_ids
    if unknown:
        raise ValueError(f"teacher output contains unknown question_id: {', '.join(sorted(unknown))}")

    records: list[dict] = []
    eligible: list[dict] = []
    for question in questions:
        question_id = question["question_id"]
        a = _candidate_view(teacher_a.get(question_id), errors_a.get(question_id))
        b = _candidate_view(teacher_b.get(question_id), errors_b.get(question_id))
        a_result, a_reason = _evaluate_candidate(a, question)
        b_result, b_reason = _evaluate_candidate(b, question)
        pair_a = {"final_answer": a["final_answer"], "error_type": a["error_type"]} if a["present"] else None
        pair_b = {"final_answer": b["final_answer"], "error_type": b["error_type"]} if b["present"] else None
        conflict_type = classify_pair(pair_a, pair_b)
        passed = [("A", a), ("B", b)]
        passed = [(side, candidate) for side, candidate in passed if (a_result if side == "A" else b_result) == "passed"]
        if passed:
            selected_side, selected = passed[0]
            same_verified_answer = len(passed) == 2 and normalize_answer(a["final_answer"]) == normalize_answer(b["final_answer"])
            verification_result = "verified"
            rejection_reason = None
            outcome = "both_agree_verified" if same_verified_answer else ("teacher_a_selected" if selected_side == "A" else "teacher_b_selected")
        elif "unresolved" in {a_result, b_result}:
            selected_side, selected, verification_result, outcome = None, None, "unresolved", "needs_review"
            rejection_reason = "; ".join(sorted({a_reason, b_reason}))
        else:
            selected_side, selected, verification_result, outcome = None, None, "rejected", "excluded"
            rejection_reason = "; ".join(sorted({a_reason, b_reason}))
        run_ids = sorted({value for value in [a["run_id"], b["run_id"]] if value})
        verified_at = datetime.now(UTC).isoformat()
        audit = {
            "question_id": question_id,
            "teacher_a": {**a, "independent_result": a_result, "independent_reason": a_reason},
            "teacher_b": {**b, "independent_result": b_result, "independent_reason": b_reason},
            "conflict": conflict_type != "agree",
            "conflict_type": conflict_type,
            "verification_method": question["verification_method"],
            "verification_result": verification_result,
            "selected_teacher_side": selected_side,
            "selected_final_answer": selected["final_answer"] if selected else None,
            "rejection_reason": rejection_reason,
            "verifier_metadata": {
                "verifier": "compass_ai.independent_verifier",
                "version": 1,
                "mode": "reference_exact" if question["verification_method"] in EXACT_VERIFICATION_METHODS or question.get("auto_verifiable") is True else "manual_review_required",
                "reference_answer_sha256": _sha256_json(question["reference_answer"]),
                "rubric_id": question.get("rubric"),
                "evidence_source": question["evidence_source"],
            },
            "source_run_ids": run_ids,
            "dataset_version": next((row["dataset_version"] for row in [teacher_a.get(question_id), teacher_b.get(question_id)] if row), "pilot-v1"),
            "dataset_checksum": expected_dataset_checksum,
            "verified_at": verified_at,
            "training_eligible": verification_result == "verified",
        }
        records.append(audit)
        if selected is not None:
            eligible.append(_eligible_record(question, selected, audit, outcome))

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "verification_records.jsonl"
    eligible_path = output_dir / "eligible_teacher_only.jsonl"
    eligible_human_path = output_dir / "eligible_with_human_correction.jsonl"
    write_jsonl(records_path, records)
    write_jsonl(eligible_path, eligible)
    write_jsonl(eligible_human_path, eligible)
    aligned = sum(row["teacher_a"]["present"] and row["teacher_b"]["present"] for row in records)
    summary = {
        "status": "needs_review" if any(row["verification_result"] == "unresolved" for row in records) else "ready",
        "teacher_a_completed": len(teacher_a),
        "teacher_b_completed": len(teacher_b),
        "aligned": aligned,
        "conflicts": sum(row["conflict_type"] in {"answer_conflict", "format_conflict", "both_suspect"} for row in records),
        "independently_verified": sum(row["verification_result"] == "verified" for row in records),
        "rejected": sum(row["verification_result"] == "rejected" for row in records),
        "unresolved": sum(row["verification_result"] == "unresolved" for row in records),
        "student_training_eligible": len(eligible),
        "dataset_checksum": expected_dataset_checksum,
        "verification_records_checksum": _sha256_file(records_path),
        "eligible_checksum": _sha256_file(eligible_path),
        "verification_records_path": str(records_path),
        "eligible_teacher_only_path": str(eligible_path),
        "eligible_with_human_correction_path": str(eligible_human_path),
    }
    (output_dir / "verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def verification_readiness(verification_records_path: Path, eligible_path: Path) -> dict:
    records = read_jsonl(verification_records_path)
    eligible = read_jsonl(eligible_path)
    eligible_ids = {row.get("question_id") for row in eligible}
    verified_ids = {row.get("question_id") for row in records if row.get("verification_result") == "verified"}
    errors = []
    if len(eligible_ids) != len(eligible):
        errors.append("duplicate_eligible_question_id")
    if eligible_ids != verified_ids:
        errors.append("eligible_question_id_mismatch")
    if any(
        row.get("training_approved") is not True
        or row.get("verification_result") != "verified"
        or row.get("verification_status") != "passed"
        for row in eligible
    ):
        errors.append("ineligible_record_present")
    def contains_verification_material(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                key in {"reference_answer", "rubric"} or contains_verification_material(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_verification_material(item) for item in value)
        return False

    if contains_verification_material(eligible):
        errors.append("verification_material_leakage")
    unresolved = sum(row.get("verification_result") == "unresolved" for row in records)
    return {
        "status": "error" if errors else ("needs_review" if unresolved else "ready"),
        "errors": errors,
        "teacher_a_completed": sum(bool(row.get("teacher_a", {}).get("present")) for row in records),
        "teacher_b_completed": sum(bool(row.get("teacher_b", {}).get("present")) for row in records),
        "aligned": sum(bool(row.get("teacher_a", {}).get("present")) and bool(row.get("teacher_b", {}).get("present")) for row in records),
        "conflicts": sum(row.get("conflict_type") in {"answer_conflict", "format_conflict", "both_suspect"} for row in records),
        "independently_verified": len(verified_ids),
        "rejected": sum(row.get("verification_result") == "rejected" for row in records),
        "unresolved": unresolved,
        "student_training_eligible": len(eligible),
        "checksum": _sha256_file(eligible_path),
    }


def apply_verification_review(
    questions_path: Path,
    verification_records_path: Path,
    eligible_path: Path,
    review_path: Path,
    output_dir: Path,
) -> dict:
    """사람의 명시적 결정을 원본과 분리된 reviewed artifact에 적용한다."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"reviewed verification output collision: {output_dir}")
    questions = {row["question_id"]: row for row in _load_verification_questions(questions_path)}
    records = read_jsonl(verification_records_path)
    initial_eligible = read_jsonl(eligible_path)
    decisions: dict[str, dict] = {}
    for index, row in enumerate(read_jsonl(review_path), 1):
        required = {"question_id", "decision", "reviewer", "evidence"}
        missing = required - row.keys() if isinstance(row, dict) else required
        if missing:
            raise ValueError(f"review decision {index} missing: {', '.join(sorted(missing))}")
        if row["decision"] not in {"accept_a", "accept_b", "reject", "human_corrected"}:
            raise ValueError(f"review decision {index} is unsupported")
        if not str(row["reviewer"]).strip() or not str(row["evidence"]).strip():
            raise ValueError(f"review decision {index} requires reviewer and evidence")
        if row["decision"] == "human_corrected" and not str(row.get("final_answer", "")).strip():
            raise ValueError(f"review decision {index} requires final_answer")
        if row["question_id"] in decisions:
            raise ValueError(f"duplicate review decision: {row['question_id']}")
        decisions[row["question_id"]] = row
    unresolved_ids = {row["question_id"] for row in records if row.get("verification_result") == "unresolved"}
    extra = set(decisions) - unresolved_ids
    if extra:
        raise ValueError(f"review decisions target non-unresolved IDs: {', '.join(sorted(extra))}")

    reviewed_records: list[dict] = []
    teacher_eligible = list(initial_eligible)
    with_human = list(initial_eligible)
    for original in records:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        decision = decisions.get(row["question_id"])
        if decision is None:
            reviewed_records.append(row)
            continue
        manual_metadata = {
            "reviewer": decision["reviewer"], "evidence": decision["evidence"],
            "decision": decision["decision"], "reviewed_at": datetime.now(UTC).isoformat(),
            "review_file": str(review_path), "review_file_checksum": _sha256_file(review_path),
        }
        row["verifier_metadata"] = {**row.get("verifier_metadata", {}), "manual_review": manual_metadata}
        if decision["decision"] == "reject":
            row.update({
                "verification_result": "rejected", "selected_teacher_side": None,
                "selected_final_answer": None, "rejection_reason": f"manual_review:{decision['evidence']}",
                "training_eligible": False, "verified_at": manual_metadata["reviewed_at"],
            })
        else:
            if decision["decision"] in {"accept_a", "accept_b"}:
                side = "A" if decision["decision"] == "accept_a" else "B"
                selected = row["teacher_a"] if side == "A" else row["teacher_b"]
                if not selected.get("present") or not str(selected.get("final_answer", "")).strip():
                    raise ValueError(f"review selected unavailable teacher for {row['question_id']}")
                outcome = "teacher_a_selected" if side == "A" else "teacher_b_selected"
                row["selected_teacher_side"] = side
                eligible_row = _eligible_record(questions[row["question_id"]], selected, row, outcome)
                teacher_eligible.append(eligible_row)
                with_human.append(eligible_row)
            else:
                selected = {
                    "final_answer": decision["final_answer"], "teacher_model": "HumanReviewer",
                    "run_id": None,
                }
                row["selected_teacher_side"] = "HUMAN"
                eligible_row = _eligible_record(questions[row["question_id"]], selected, row, "human_corrected")
                eligible_row.update({"tag": "human_corrected", "reviewer": decision["reviewer"]})
                with_human.append(eligible_row)
            row.update({
                "verification_result": "verified", "selected_final_answer": selected["final_answer"],
                "rejection_reason": None, "training_eligible": True, "verified_at": manual_metadata["reviewed_at"],
            })
        reviewed_records.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = output_dir / "verification_records.jsonl"
    teacher_path = output_dir / "eligible_teacher_only.jsonl"
    human_path = output_dir / "eligible_with_human_correction.jsonl"
    write_jsonl(reviewed_path, reviewed_records)
    write_jsonl(teacher_path, teacher_eligible)
    write_jsonl(human_path, with_human)
    readiness = verification_readiness(reviewed_path, human_path)
    result = {
        **readiness,
        "verification_records_path": str(reviewed_path),
        "eligible_teacher_only_path": str(teacher_path),
        "eligible_with_human_correction_path": str(human_path),
        "review_file_checksum": _sha256_file(review_path),
    }
    (output_dir / "verification_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
