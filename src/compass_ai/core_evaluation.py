"""Core-v2 private key를 training과 분리해 runtime 평가/채점 artifact로 연결한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata

from .jsonl import read_jsonl, write_jsonl
from .result_analysis import CONDITIONS, PRIMARY_CONDITIONS, CORE_V2_CHECKSUM, PROTOCOL_CHECKSUM
from .student_training import load_student_config


def _normalize(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split()).casefold()


def prepare_core_private_evaluation(root: Path, config_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"private evaluation output collision: {output_dir}")
    manifest = json.loads((root / "data" / "questions" / "core_v2" / "core_manifest.json").read_text(encoding="utf-8"))
    key_path = root / "artifacts" / "private" / "core_v2_private_key.jsonl"
    if not key_path.is_file():
        raise FileNotFoundError("Core-v2 private key is missing; transfer it separately on DGX")
    if hashlib.sha256(key_path.read_bytes()).hexdigest() != manifest["private_key_sha256"]:
        raise ValueError("Core-v2 private key checksum mismatch")
    questions = read_jsonl(root / "data" / "questions" / "core_v2" / "private_eval_questions.jsonl")
    keys = {row["question_id"]: row for row in read_jsonl(key_path)}
    if set(keys) != {row["question_id"] for row in questions}:
        raise ValueError("private key question IDs do not match Core-v2 private_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "private_runtime.jsonl"
    write_jsonl(runtime_path, [{
        "evaluation_id": row["question_id"], "prompt": row["question"],
        "reference_answer": keys[row["question_id"]]["answer"], "domain": row["domain"],
        "verification_method": keys[row["question_id"]]["verification_method"],
    } for row in questions])
    eval_configs = output_dir / "configs"
    eval_configs.mkdir()
    generated = {}
    for condition in CONDITIONS:
        stem = condition.lower()
        candidates = [config_dir / f"{stem}.json"]
        if condition == "S4_with_human_correction": candidates.insert(0, config_dir / "s4_with_human_correction.json")
        config_path = next((path for path in candidates if path.exists()), None)
        if config_path is None and condition == "S5":
            continue
        if config_path is None:
            raise FileNotFoundError(f"student config snapshot missing for {condition}")
        config = load_student_config(config_path)
        config["evaluation_dataset"] = str(runtime_path.resolve())
        target = eval_configs / config_path.name
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        generated[condition] = str(target)
    summary = {
        "status": "ready", "dataset_checksum": CORE_V2_CHECKSUM,
        "protocol_checksum": PROTOCOL_CHECKSUM, "runtime_path": str(runtime_path),
        "evaluation_configs": generated, "contains_private_answers": True,
    }
    (output_dir / "private_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def collect_core_scored_results(config_dir: Path, private_runtime: Path, output_dir: Path, run_id: str, review_file: Path | None = None) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"scored output collision: {output_dir}")
    runtime = {row["evaluation_id"]: row for row in read_jsonl(private_runtime)}
    if len(runtime) != 48:
        raise ValueError("Core-v2 private runtime must contain 48 items")
    decisions = {}
    if review_file is not None:
        for row in read_jsonl(review_file):
            key = (row.get("condition"), row.get("evaluation_id"))
            if key in decisions or not isinstance(row.get("correct"), bool):
                raise ValueError("manual scoring review has duplicate or malformed decision")
            decisions[key] = row
    scored = []
    for config_path in sorted(config_dir.glob("*.json")):
        config = load_student_config(config_path)
        condition = config["condition"]
        if condition not in CONDITIONS:
            continue
        evaluation = Path(config["output_dir"]) / "evaluation.rerun.json"
        if not evaluation.is_file():
            raise FileNotFoundError(f"evaluation result missing for {condition}: {evaluation}")
        result_rows = json.loads(evaluation.read_text(encoding="utf-8"))["results"]
        if {row["evaluation_id"] for row in result_rows} != set(runtime):
            raise ValueError(f"private evaluation IDs mismatch for {condition}")
        for row in result_rows:
            item = runtime[row["evaluation_id"]]
            prediction = str(row.get("prediction", ""))
            invalid = not prediction.strip()
            manual = decisions.get((condition, row["evaluation_id"]))
            rubric = item["verification_method"] == "rubric_review"
            unresolved = rubric and manual is None
            correct = bool(manual["correct"]) if manual else (
                not rubric and not invalid and _normalize(prediction) == _normalize(item["reference_answer"])
            )
            scored.append({
                "run_id": run_id, "condition": condition, "evaluation_id": row["evaluation_id"],
                "domain": item["domain"], "correct": correct, "invalid": invalid,
                "unresolved": unresolved, "mode": "actual", "dataset_checksum": CORE_V2_CHECKSUM,
                "protocol_checksum": PROTOCOL_CHECKSUM,
            })
    if not set(PRIMARY_CONDITIONS).issubset({row["condition"] for row in scored}):
        raise ValueError("not all frozen primary evaluation conditions are present")
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_items.jsonl"
    write_jsonl(scored_path, scored)
    summary = {"status": "needs_review" if any(row["unresolved"] for row in scored) else "ready",
               "scored_items": len(scored), "unresolved": sum(row["unresolved"] for row in scored),
               "scored_items_path": str(scored_path)}
    (output_dir / "scoring_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary
