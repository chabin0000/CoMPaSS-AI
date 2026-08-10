"""동결된 Core-v2와 평가 protocol snapshot의 무결성을 검사한다."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from .jsonl import read_jsonl


FILES = ("train_questions.jsonl", "validation_questions.jsonl", "private_eval_questions.jsonl")
EXPECTED_SPLITS = {"train": 120, "validation": 24, "private_eval": 48}
EXPECTED_DOMAINS = {"math": 39, "coding": 39, "science": 39, "logic": 38, "korean": 37}
REQUIRED = {
    "question_id", "dataset_version", "split", "domain", "difficulty", "task_type",
    "template_family", "question", "verification_method", "reference", "rubric", "provenance",
}


class CoreV2DatasetError(ValueError):
    """Core-v2 freeze 계약 위반."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_checksum(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def check_core_v2_dataset(root: Path) -> dict:
    core = root / "data" / "questions" / "core_v2"
    manifest = json.loads((core / "core_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "core-v2" or manifest.get("status") != "frozen":
        raise CoreV2DatasetError("Core-v2 manifest is not frozen")
    hashes = {filename: _sha(core / filename) for filename in FILES}
    if hashes != manifest.get("file_sha256"):
        raise CoreV2DatasetError("Core-v2 file checksum mismatch")
    checksum = _dataset_checksum(hashes)
    if checksum != manifest.get("dataset_checksum"):
        raise CoreV2DatasetError("Core-v2 dataset checksum mismatch")

    rows = []
    ids, families, questions = set(), set(), set()
    for split, filename in zip(EXPECTED_SPLITS, FILES):
        for index, row in enumerate(read_jsonl(core / filename), 1):
            missing = REQUIRED - row.keys()
            if missing:
                raise CoreV2DatasetError(f"{filename}:{index} missing {sorted(missing)}")
            if row["dataset_version"] != "core-v2" or row["split"] != split:
                raise CoreV2DatasetError(f"{row['question_id']}: split/version mismatch")
            if row["question_id"] in ids or row["template_family"] in families:
                raise CoreV2DatasetError(f"{row['question_id']}: duplicate id or family")
            normalized = _normalize(row["question"])
            if normalized in questions:
                raise CoreV2DatasetError(f"{row['question_id']}: normalized duplicate")
            ids.add(row["question_id"]); families.add(row["template_family"]); questions.add(normalized)
            steps = row["rubric"].get("reasoning_steps")
            difficulty_ok = (
                row["difficulty"] == "easy" and steps == 1
                or row["difficulty"] == "medium" and isinstance(steps, int) and 2 <= steps <= 3
                or row["difficulty"] == "hard" and isinstance(steps, int) and steps >= 3
            )
            if not difficulty_ok:
                raise CoreV2DatasetError(f"{row['question_id']}: difficulty mismatch")
            if split == "private_eval":
                if set(row["reference"]) != {"visibility", "key_id", "answer_sha256"} or row["reference"]["visibility"] != "private":
                    raise CoreV2DatasetError(f"{row['question_id']}: private reference exposure")
                if "criteria" in row["rubric"] or row["rubric"].get("visibility") != "private":
                    raise CoreV2DatasetError(f"{row['question_id']}: private rubric exposure")
            else:
                if row["reference"].get("visibility") != "verification_only" or not row["reference"].get("answer"):
                    raise CoreV2DatasetError(f"{row['question_id']}: verification answer missing")
            rows.append(row)

    split_counts = dict(Counter(row["split"] for row in rows))
    domain_counts = dict(Counter(row["domain"] for row in rows))
    if split_counts != EXPECTED_SPLITS or domain_counts != EXPECTED_DOMAINS:
        raise CoreV2DatasetError("Core-v2 split/domain distribution mismatch")
    auto_count = sum(row["rubric"]["auto_verifiable"] for row in rows)
    if auto_count / len(rows) < 0.85:
        raise CoreV2DatasetError("Core-v2 auto verification ratio below 85%")

    audit_meta = manifest["quality_audit"]
    audit_path = root / audit_meta["path"]
    if _sha(audit_path) != audit_meta["sha256"]:
        raise CoreV2DatasetError("Core-v2 quality audit checksum mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["summary"]["freeze_ready"] or audit["summary"]["status_counts"]["FAIL"]:
        raise CoreV2DatasetError("Core-v2 quality audit did not pass freeze gate")
    review_ids = sorted(
        finding["question_id"] for finding in audit["findings"] if finding["status"] == "REVIEW"
    )
    acceptance_meta = manifest.get("review_acceptance", {})
    acceptance_path = root / acceptance_meta.get("path", "")
    if not acceptance_path.is_file() or _sha(acceptance_path) != acceptance_meta.get("sha256"):
        raise CoreV2DatasetError("Core-v2 REVIEW acceptance record mismatch")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    accepted_ids = sorted(row["question_id"] for row in acceptance.get("reviews", []))
    if (
        review_ids != accepted_ids
        or accepted_ids != sorted(acceptance_meta.get("question_ids", []))
        or acceptance.get("decision") != "accepted_without_dataset_change"
        or acceptance.get("dataset_checksum") != checksum
    ):
        raise CoreV2DatasetError("Core-v2 REVIEW acceptance does not cover the audit findings")

    protocol_meta = manifest["evaluation_protocol"]
    protocol_manifest_path = root / protocol_meta["manifest_path"]
    protocol = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    protocol_dir = protocol_manifest_path.parent
    protocol_paths = {
        "protocol.yaml": root / "configs" / "evaluation" / "argos_v1_core_v2_private_eval_v1.yaml",
        "results_template.csv": protocol_dir / "results_template.csv",
        "comparison_template.csv": protocol_dir / "comparison_template.csv",
    }
    actual_protocol_hashes = {name: _sha(path) for name, path in protocol_paths.items()}
    snapshot = _dataset_checksum(actual_protocol_hashes)
    if actual_protocol_hashes != protocol["files"] or snapshot != protocol["snapshot_checksum"] or snapshot != protocol_meta["snapshot_checksum"]:
        raise CoreV2DatasetError("Core-v2 evaluation protocol snapshot mismatch")

    private_key = root / "artifacts" / "private" / "core_v2_private_key.jsonl"
    private_status = "UNKNOWN"
    if private_key.exists():
        if _sha(private_key) != manifest["private_key_sha256"]:
            raise CoreV2DatasetError("Core-v2 private key checksum mismatch")
        private_status = "PASS"
    return {
        "status": "ready",
        "dataset_version": "core-v2",
        "question_count": len(rows),
        "split_counts": split_counts,
        "domain_counts": domain_counts,
        "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
        "auto_verifiable_count": auto_count,
        "auto_verifiable_ratio": auto_count / len(rows),
        "dataset_checksum": checksum,
        "quality_audit_status_counts": audit["summary"]["status_counts"],
        "review_acceptance": {"decision": acceptance["decision"], "question_ids": accepted_ids},
        "near_duplicate_template_groups": audit["summary"]["near_duplicate_template_groups"],
        "cross_split_template_leakage_pairs": audit["summary"]["cross_split_template_leakage_pairs"],
        "evaluation_protocol_checksum": snapshot,
        "private_key_status": private_status,
    }
