"""검증된 실제 Core-v2 teacher artifact에서 S0~S5 dataset을 만든다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import yaml

from .conditions import assert_matched_comparison, build_conditions
from .core_v2_dataset import check_core_v2_dataset
from .jsonl import read_jsonl, write_jsonl
from .student_data import build_student_dataset
from .teacher_run import validate_teacher_raw_responses


def build_core_condition_datasets(
    root: Path, teacher_a_path: Path, teacher_b_path: Path,
    teacher_eligible_path: Path, human_eligible_path: Path,
    korean_source_path: Path | None, output_root: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"condition output collision: {output_root}")
    report = check_core_v2_dataset(root)
    questions_path = root / "data" / "questions" / "core_v2" / "train_questions.jsonl"
    questions = read_jsonl(questions_path)
    private_ids = {row["question_id"] for row in read_jsonl(root / "data" / "questions" / "core_v2" / "private_eval_questions.jsonl")}
    a = validate_teacher_raw_responses(read_jsonl(teacher_a_path))
    b = validate_teacher_raw_responses(read_jsonl(teacher_b_path))
    if len(a) != 120 or len(b) != 120:
        raise ValueError("both Core-v2 teacher runs must contain 120 completed train responses")
    teacher_models = (a[0]["teacher_model"], b[0]["teacher_model"])
    if teacher_models[0] == teacher_models[1]:
        raise ValueError("teacher A/B model identifiers must differ")
    qmap = {row["question_id"]: row for row in questions}
    raw = []
    for row in [*a, *b]:
        if row["dataset_version"] != "core-v2" or row["question_id"] not in qmap:
            raise ValueError("teacher output is not aligned to Core-v2 train")
        question = qmap[row["question_id"]]
        raw.append({**row, "question": question["question"], "domain": question["domain"],
                    "task_type": question["task_type"], "verification_status": "pending"})
    teacher_eligible = read_jsonl(teacher_eligible_path)
    eligible_by_pair = {(row["question_id"], row["teacher_model"]): row for row in teacher_eligible}
    raw = [{**row, **eligible_by_pair.get((row["question_id"], row["teacher_model"]), {})} for row in raw]
    human_rows = [row for row in read_jsonl(human_eligible_path) if row.get("verification_outcome") == "human_corrected"]
    korean = read_jsonl(korean_source_path) if korean_source_path is not None else []
    conditions = build_conditions(questions, raw + human_rows, korean_source=korean, teacher_models=teacher_models)
    assert_matched_comparison(conditions["S3_matched"], conditions["S4_teacher_only"])
    output_root.mkdir(parents=True, exist_ok=True)
    evaluation_dir = output_root.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    validation_path = evaluation_dir / "validation_runtime.jsonl"
    validation_rows = read_jsonl(root / "data" / "questions" / "core_v2" / "validation_questions.jsonl")
    write_jsonl(validation_path, [{
        "evaluation_id": row["question_id"], "prompt": row["question"],
        "reference_answer": row["reference"]["answer"], "domain": row["domain"],
        "verification_method": row["verification_method"],
    } for row in validation_rows])
    counts = {"S0": 0}
    built_metadata: dict[str, dict] = {}
    active_conditions = ["S1", "S2", "S3a", "S3b", "S3_matched", "S4", "S4_teacher_only", "S4_with_human_correction"]
    if korean:
        active_conditions.append("S5")
    for condition in active_conditions:
        source = output_root / f"{condition}.source.jsonl"
        target = output_root / f"{condition}.jsonl"
        write_jsonl(source, conditions[condition])
        built = build_student_dataset(source, target, condition)
        ids = set(built["question_ids"])
        if ids & private_ids:
            raise ValueError(f"private_eval leakage in {condition}")
        counts[condition] = built["sample_count"]
        built_metadata[condition] = built
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_names = {
        "S0": "s0", "S1": "s1", "S2": "s2", "S3a": "s3a", "S3b": "s3b",
        "S3_matched": "s3_matched", "S4": "s4", "S4_teacher_only": "s4_teacher_only",
        "S4_with_human_correction": "s4_with_human_correction", "S5": "s5",
    }
    generated_configs = {}
    for condition, stem in config_names.items():
        if condition == "S5" and not korean:
            continue
        canonical = yaml.safe_load((root / "configs" / "student" / f"{stem}.yaml").read_text(encoding="utf-8"))
        canonical["evaluation_dataset"] = str(validation_path.resolve())
        if condition != "S0":
            canonical["dataset"] = str((output_root / f"{condition}.jsonl").resolve())
            canonical["dataset_checksum"] = built_metadata[condition]["dataset_checksum"]
            canonical["expected_sample_count"] = counts[condition]
        canonical["output_dir"] = str((output_root.parent / "training" / stem).resolve())
        config_path = config_dir / f"{stem}.json"
        config_path.write_text(json.dumps(canonical, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        generated_configs[condition] = str(config_path)
    metadata = {
        "status": "ready", "core_v2_checksum": report["dataset_checksum"],
        "train_file_checksum": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "teacher_models": list(teacher_models), "condition_counts": counts,
        "matched_comparison": "PASS", "generated_student_configs": generated_configs,
        "s5_status": "ready" if korean else "deferred_follow_up_placeholder",
    }
    (output_root / "condition_build_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
