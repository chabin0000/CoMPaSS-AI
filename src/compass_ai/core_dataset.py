"""Pilot-v1과 분리된 Core-v1 질문 풀의 계약과 무결성을 검사한다."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .jsonl import read_jsonl


CORE_VERSION = "core-v1"
CORE_FILES = {
    "train": "train_questions.jsonl",
    "validation": "validation_questions.jsonl",
    "private_eval": "private_eval_questions.jsonl",
}
EXPECTED_SPLIT_COUNTS = {"train": 120, "validation": 24, "private_eval": 48}
DOMAINS = {"math", "coding", "science", "logic", "korean"}
DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {
    "question_id", "dataset_version", "split", "domain", "difficulty",
    "task_type", "question", "verification_method", "reference", "rubric", "provenance",
}


class CoreDatasetError(ValueError):
    """Core-v1 schema, split 또는 checksum 위반."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_checksum(file_checksums: dict[str, str]) -> str:
    payload = json.dumps(file_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _validate_record(row: dict, split: str, index: int) -> None:
    if not isinstance(row, dict):
        raise CoreDatasetError(f"{split} row {index}: object required")
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise CoreDatasetError(f"{split} row {index}: missing {', '.join(sorted(missing))}")
    if row["dataset_version"] != CORE_VERSION or row["split"] != split:
        raise CoreDatasetError(f"{row.get('question_id', index)}: dataset/split mismatch")
    if row["domain"] not in DOMAINS or row["difficulty"] not in DIFFICULTIES:
        raise CoreDatasetError(f"{row['question_id']}: invalid domain or difficulty")
    for field in ("question_id", "task_type", "question", "verification_method"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise CoreDatasetError(f"{row['question_id']}: empty {field}")
    reference, rubric = row["reference"], row["rubric"]
    if not isinstance(reference, dict) or not isinstance(rubric, dict):
        raise CoreDatasetError(f"{row['question_id']}: reference/rubric must be objects")
    if not isinstance(rubric.get("auto_verifiable"), bool):
        raise CoreDatasetError(f"{row['question_id']}: auto_verifiable must be boolean")
    if split == "private_eval":
        if set(reference) != {"visibility", "key_id", "answer_sha256"} or reference["visibility"] != "private":
            raise CoreDatasetError(f"{row['question_id']}: private reference exposure")
        if rubric.get("visibility") != "private" or "criteria" in rubric:
            raise CoreDatasetError(f"{row['question_id']}: private rubric exposure")
    else:
        if reference.get("visibility") != "verification_only" or not str(reference.get("answer", "")).strip():
            raise CoreDatasetError(f"{row['question_id']}: public verification answer missing")
        if rubric.get("visibility") != "verification_only" or not isinstance(rubric.get("criteria"), list):
            raise CoreDatasetError(f"{row['question_id']}: verification rubric missing")


def check_core_dataset(root: Path) -> dict:
    core_dir = root / "data" / "questions" / "core_v1"
    manifest_path = core_dir / "core_manifest.json"
    if not manifest_path.exists():
        raise CoreDatasetError("Core-v1 manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hashes = {name: file_sha256(core_dir / name) for name in CORE_FILES.values()}
    if actual_hashes != manifest.get("file_sha256"):
        raise CoreDatasetError("Core-v1 file checksum mismatch")
    actual_dataset_checksum = dataset_checksum(actual_hashes)
    if actual_dataset_checksum != manifest.get("dataset_checksum"):
        raise CoreDatasetError("Core-v1 dataset checksum mismatch")

    all_rows: list[dict] = []
    split_counts: dict[str, int] = {}
    split_ids: dict[str, set[str]] = {}
    for split, filename in CORE_FILES.items():
        rows = read_jsonl(core_dir / filename)
        split_counts[split] = len(rows)
        split_ids[split] = set()
        for index, row in enumerate(rows, 1):
            _validate_record(row, split, index)
            if row["question_id"] in split_ids[split]:
                raise CoreDatasetError(f"duplicate question_id: {row['question_id']}")
            split_ids[split].add(row["question_id"])
        all_rows.extend(rows)
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise CoreDatasetError(f"split count mismatch: {split_counts}")
    if any(split_ids[left] & split_ids[right] for left in CORE_FILES for right in CORE_FILES if left < right):
        raise CoreDatasetError("question_id overlap across splits")
    normalized = [_normalized_question(row["question"]) for row in all_rows]
    if len(normalized) != len(set(normalized)):
        raise CoreDatasetError("duplicate question text across splits")

    pilot_path = root / "data" / "questions" / "pilot" / "pilot_questions.jsonl"
    if pilot_path.exists():
        pilot_ids = {row["question_id"] for row in read_jsonl(pilot_path)}
        overlap = pilot_ids & {row["question_id"] for row in all_rows}
        if overlap:
            raise CoreDatasetError(f"Pilot-v1 question_id overlap: {sorted(overlap)}")

    domain_counts = dict(sorted(Counter(row["domain"] for row in all_rows).items()))
    auto_count = sum(bool(row["rubric"]["auto_verifiable"]) for row in all_rows)
    report = {
        "status": "ready",
        "dataset_version": CORE_VERSION,
        "question_count": len(all_rows),
        "split_counts": split_counts,
        "domain_counts": domain_counts,
        "auto_verifiable_count": auto_count,
        "auto_verifiable_ratio": auto_count / len(all_rows),
        "dataset_checksum": actual_dataset_checksum,
        "file_sha256": actual_hashes,
        "private_key_status": "separate_not_required_for_training",
    }
    expected_summary = manifest.get("summary", {})
    for key in ("question_count", "split_counts", "domain_counts", "auto_verifiable_count"):
        if expected_summary.get(key) != report[key]:
            raise CoreDatasetError(f"manifest summary mismatch: {key}")
    return report
