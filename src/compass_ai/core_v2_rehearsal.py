"""Core-v2 실제 구조로 S0~S5를 연결하는 대형 모델 없는 synthetic rehearsal."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from .conditions import assert_matched_comparison, build_conditions
from .core_v2_dataset import check_core_v2_dataset
from .dgx import collect_dgx_system_report
from .jsonl import read_jsonl, write_jsonl
from .result_analysis import CONDITIONS as ANALYSIS_CONDITIONS, CORE_V2_CHECKSUM, PROTOCOL_CHECKSUM, analyze_results
from .student_data import build_student_dataset
from .student_tiny import create_tiny_causal_lm
from .student_training import run_student
from .teacher import merge_responses


EXPECTED_CONDITION_COUNTS = {
    "S0": 0, "S1": 120, "S2": 120, "S3a": 240, "S3b": 120,
    "S3_matched": 114, "S4": 120, "S4_teacher_only": 114,
    "S4_with_human_correction": 120, "S5": 125,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(path: Path, run_id: str, artifact_type: str, config_snapshot: object = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    value = {
        "run_id": run_id, "artifact_type": artifact_type, "created_at": datetime.now(UTC).isoformat(),
        "dataset_checksum": CORE_V2_CHECKSUM, "protocol_checksum": PROTOCOL_CHECKSUM,
        "config_snapshot": config_snapshot,
    }
    (path / "artifact_metadata.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _raw(question: dict, teacher: str, final_answer: str, run_id: str) -> dict:
    return {
        "question_id": question["question_id"], "domain": question["domain"], "task_type": question["task_type"],
        "question": question["question"], "teacher_model": teacher, "checkpoint": "synthetic-no-model",
        "quantization": "none", "generation_config": {"temperature": 0, "synthetic": True},
        "raw_response": {"choices": [{"message": {"reasoning_content": "excluded from targets", "content": final_answer}}]},
        "final_answer": final_answer, "verification_status": "pending",
        "verification_method": question["verification_method"], "evidence": "synthetic teacher output",
        "error_type": None, "tag": "teacher_raw", "reviewer": None, "verification_outcome": "pending",
        "timestamp": datetime.now(UTC).isoformat(), "training_approved": False,
        "response_status": "answered", "run_id": run_id, "dataset_version": "core-v2",
        "dataset_checksum": CORE_V2_CHECKSUM,
    }


def _korean_source(index: int, run_id: str) -> dict:
    prompts = [
        ("다음 표현을 자연스럽게 고치시오: 계획을 실시하다.", "계획을 실행하다."),
        ("'빛이 밝아서 눈이 부셨다'의 원인 부분을 쓰시오.", "빛이 밝아서"),
        ("'안전 점검을 마친 뒤 장치를 켰다'에서 먼저 일어난 일을 쓰시오.", "안전 점검을 마친 일"),
        ("'모두가 동의한 것은 아니다'가 전체 동의를 뜻하는지 쓰시오.", "아니다"),
        ("'전력 소비를 줄이는 장치'에서 관형절을 쓰시오.", "전력 소비를 줄이는"),
    ]
    question, answer = prompts[index]
    return {
        "question_id": f"SYN-KOR-SOURCE-{index + 1:03d}", "domain": "korean", "task_type": "verified_korean_source",
        "question": question, "teacher_model": "KoreanSource", "final_answer": answer,
        "verification_status": "passed", "verification_method": "exact_match", "verification_result": "verified",
        "verification_outcome": "teacher_a_selected", "training_approved": True, "tag": "korean_source",
        "source_type": "korean_source", "error_type": None, "response_status": "answered",
        "run_id": run_id, "dataset_version": "core-v2", "dataset_checksum": CORE_V2_CHECKSUM,
    }


def _tiny_config(path: Path, dataset: Path, evaluation: Path, base_model: Path, output: Path) -> Path:
    value = {
        "run_name": "core-v2-synthetic-tiny", "condition": "S3b", "base_model": str(base_model),
        "dataset": str(dataset), "evaluation_dataset": str(evaluation), "output_dir": str(output),
        "training_enabled": True, "seed": 17, "epochs": 1, "learning_rate": 0.001,
        "batch_size": 1, "gradient_accumulation_steps": 1, "max_sequence_length": 64,
        "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.0, "target_modules": ["c_attn"],
        "fp16": False, "bf16": False, "quantization": "none", "allow_quantization_fallback": True,
        "max_steps": 1, "save_steps": 1, "resume_from_checkpoint": "auto", "prompt_format": "plain",
        "chat_template_required": False, "trust_remote_code": False, "local_files_only": True,
        "dataset_checksum": None, "expected_sample_count": 120, "minimum_disk_free_gb": 0,
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_core_v2_synthetic_rehearsal(repo_root: Path, output_root: Path, status_report: Path | None = None) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"synthetic rehearsal output collision: {output_root}")
    run_id = output_root.name
    dirs = {name: output_root / name for name in ["system", "teachers", "verification", "student_datasets", "training", "evaluation", "summary"]}
    for name, path in dirs.items(): _metadata(path, run_id, name)
    core_report = check_core_v2_dataset(repo_root)
    collect_dgx_system_report(repo_root, dirs["system"] / "report")
    questions = read_jsonl(repo_root / "data" / "questions" / "core_v2" / "train_questions.jsonl")

    raw, answer_by_id = [], {row["question_id"]: row["reference"]["answer"] for row in questions}
    for index, question in enumerate(questions):
        correct, wrong = answer_by_id[question["question_id"]], f"synthetic-wrong-{index:03d}"
        if index % 20 == 0:
            answer_a, answer_b = wrong + "-a", wrong + "-b"
        elif index % 4 == 1:
            answer_a, answer_b = correct, wrong
        elif index % 4 == 2:
            answer_a, answer_b = wrong, correct
        else:
            answer_a = answer_b = correct
        raw.extend([_raw(question, "MockTeacherA", answer_a, f"{run_id}-teacher-a"), _raw(question, "MockTeacherB", answer_b, f"{run_id}-teacher-b")])
    write_jsonl(dirs["teachers"] / "teacher_a.jsonl", [row for row in raw if row["teacher_model"] == "MockTeacherA"])
    write_jsonl(dirs["teachers"] / "teacher_b.jsonl", [row for row in raw if row["teacher_model"] == "MockTeacherB"])
    merged = merge_responses(raw)
    conflict_by_id = {row["question_id"]: row["conflict_type"] for row in merged}
    write_jsonl(dirs["verification"] / "conflicts.jsonl", merged)

    by_id = {question["question_id"]: [row for row in raw if row["question_id"] == question["question_id"]] for question in questions}
    verification_records, manual_decisions, human_rows = [], [], []
    for question in questions:
        qid, reference = question["question_id"], answer_by_id[question["question_id"]]
        candidates = by_id[qid]
        correct = [row for row in candidates if row["final_answer"] == reference]
        manual_gate = question["verification_method"] == "rubric_review"
        initial_result = "unresolved" if manual_gate else "verified" if correct else "rejected"
        if not correct:
            human = {
                **_raw(question, "HumanReviewer", reference, f"{run_id}-manual-review"),
                "raw_response": "manual correction; no teacher reasoning", "verification_status": "passed",
                "verification_result": "verified", "verification_outcome": "human_corrected",
                "training_approved": True, "tag": "human_corrected", "reviewer": "synthetic-independent-reviewer",
                "conflict_type": conflict_by_id[qid],
            }
            human_rows.append(human)
            manual_decisions.append({"question_id": qid, "decision": "human_corrected", "reason": "both synthetic teachers failed independent reference"})
            selected = human
        else:
            selected = correct[0]
            selected.update({
                "verification_status": "passed", "verification_result": "verified", "training_approved": True,
                "tag": "teacher_verified", "conflict_type": conflict_by_id[qid],
                "verification_outcome": "both_agree_verified" if len(correct) == 2 else ("teacher_a_selected" if selected["teacher_model"] == "MockTeacherA" else "teacher_b_selected"),
            })
            if manual_gate:
                manual_decisions.append({"question_id": qid, "decision": "accept_teacher", "teacher": selected["teacher_model"], "reason": "synthetic rubric gate applied"})
        verification_records.append({
            "question_id": qid, "conflict_type": conflict_by_id[qid], "initial_result": initial_result,
            "final_result": "verified", "selected_source": selected["teacher_model"],
            "manual_gate": manual_gate or not correct, "dataset_checksum": CORE_V2_CHECKSUM,
        })
    write_jsonl(dirs["verification"] / "verification_records.jsonl", verification_records)
    write_jsonl(dirs["verification"] / "manual_review_decisions.jsonl", manual_decisions)

    verified_records = raw + human_rows
    korean_source = [_korean_source(index, run_id) for index in range(5)]
    conditions = build_conditions(questions, verified_records, korean_source=korean_source)
    assert_matched_comparison(conditions["S3_matched"], conditions["S4_teacher_only"])
    actual_counts = {name: len(rows) for name, rows in conditions.items() if isinstance(rows, list)}
    if actual_counts != EXPECTED_CONDITION_COUNTS:
        raise AssertionError(f"condition count mismatch: {actual_counts}")

    student_outputs = {}
    for condition in ["S1", "S2", "S3a", "S3b", "S3_matched", "S4", "S4_teacher_only", "S4_with_human_correction", "S5"]:
        source_path = dirs["student_datasets"] / f"{condition}.source.jsonl"
        output_path = dirs["student_datasets"] / f"{condition}.jsonl"
        write_jsonl(source_path, conditions[condition])
        student_outputs[condition] = build_student_dataset(source_path, output_path, condition)
    (dirs["student_datasets"] / "S0.json").write_text(json.dumps({"condition": "S0", "training_enabled": False, "sample_count": 0}, indent=2), encoding="utf-8")
    private_ids = {row["question_id"] for row in read_jsonl(repo_root / "data" / "questions" / "core_v2" / "private_eval_questions.jsonl")}
    for condition, metadata in student_outputs.items():
        rows = read_jsonl(Path(metadata["output_path"]))
        if private_ids & {row["source_question_id"] for row in rows}:
            raise AssertionError(f"private_eval leakage in {condition}")
        serialized = json.dumps(rows, ensure_ascii=False)
        if '"reference"' in serialized or '"rubric"' in serialized or "reasoning_content" in serialized:
            raise AssertionError(f"verification/reasoning leakage in {condition}")

    tiny_base = dirs["training"] / "tiny-base"
    tiny_output = dirs["training"] / "tiny-s3b"
    tiny_eval = dirs["evaluation"] / "tiny_public_eval.jsonl"
    write_jsonl(tiny_eval, [{"evaluation_id": "SYN-PUBLIC-EVAL-001", "prompt": "실행 경로 확인", "reference_answer": "정답"}])
    create_tiny_causal_lm(tiny_base, seed=17)
    tiny_config = _tiny_config(dirs["training"] / "tiny_config.json", dirs["student_datasets"] / "S3b.jsonl", tiny_eval, tiny_base, tiny_output)
    tiny = run_student(tiny_config, repo_root)

    private_questions = read_jsonl(repo_root / "data" / "questions" / "core_v2" / "private_eval_questions.jsonl")
    thresholds = {"S0": 38, "S1": 45, "S2": 46, "S3a": 50, "S3b": 51, "S3_matched": 55, "S4_teacher_only": 62, "S4_with_human_correction": 65, "S5": 66}
    scored = []
    for condition in ANALYSIS_CONDITIONS:
        for question in private_questions:
            value = int(hashlib.sha256(f"{condition}:{question['question_id']}".encode()).hexdigest()[:8], 16) % 100
            invalid = value == 99
            unresolved = not question["rubric"]["auto_verifiable"] and value % 17 == 0
            scored.append({
                "run_id": run_id, "condition": condition, "evaluation_id": question["question_id"],
                "domain": question["domain"], "correct": value < thresholds[condition] and not invalid and not unresolved,
                "invalid": invalid, "unresolved": unresolved, "mode": "synthetic_pipeline_validation",
                "dataset_checksum": CORE_V2_CHECKSUM, "protocol_checksum": PROTOCOL_CHECKSUM,
            })
    write_jsonl(dirs["evaluation"] / "scored_items.jsonl", scored)
    analysis = analyze_results(dirs["evaluation"], dirs["summary"] / "analysis", allow_synthetic=True)

    summary = {
        "status": "complete", "mode": "synthetic_pipeline_validation", "performance_claim": False,
        "run_id": run_id, "dataset_checksum": CORE_V2_CHECKSUM, "protocol_checksum": PROTOCOL_CHECKSUM,
        "core_integrity": core_report["status"], "teacher_counts": {"A": 120, "B": 120},
        "conflict_counts": dict(Counter(row["conflict_type"] for row in merged)),
        "manual_review_count": len(manual_decisions), "condition_counts": actual_counts,
        "matched_comparison": "PASS", "private_eval_leakage": "PASS",
        "tiny_training": {"status": tiny["status"], "adapter_reload_verified": tiny["adapter_reload_verified"]},
        "analysis": {"status": analysis["status"], "mode": analysis["mode"], "performance_claim": analysis["performance_claim"]},
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (dirs["summary"] / "rehearsal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if status_report is not None:
        status_report.parent.mkdir(parents=True, exist_ok=True)
        status_report.write_text(json.dumps({**summary, "output_root": str(output_root)}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**summary, "output_root": str(output_root)}
