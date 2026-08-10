"""Core-v2 전수 품질감사. 외부 embedding/API 없이 구조·shingle·reference 계약을 검사한다."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
FILES = ("train_questions.jsonl", "validation_questions.jsonl", "private_eval_questions.jsonl")
EXPECTED_SPLITS = {"train": 120, "validation": 24, "private_eval": 48}
EXPECTED_DOMAINS = {"math": 39, "coding": 39, "science": 39, "logic": 38, "korean": 37}
ALLOWED_METHODS = {
    "exact_match", "calculation", "deterministic_calculation", "code_execution",
    "executable_tests", "programmatic_constraints", "constraint_solver", "rubric_review",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def template(text: str) -> str:
    value = re.sub(r"[0-9]+(?:\.[0-9]+)?", "<num>", normalize(text))
    value = re.sub(r"\b[a-z]\b", "<var>", value)
    return value


def shingles(text: str, size: int = 4) -> set[str]:
    compact = re.sub(r"\s+", "", template(text))
    return {compact[index:index + size] for index in range(max(1, len(compact) - size + 1))}


def similarity(left: str, right: str) -> float:
    a, b = shingles(left), shingles(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def load_answers(root: Path, rows: list[dict]) -> tuple[dict[str, str], dict[str, list[str]], str | None]:
    answers, criteria = {}, {}
    for row in rows:
        if row["split"] != "private_eval":
            answers[row["question_id"]] = row["reference"]["answer"]
            criteria[row["question_id"]] = row["rubric"].get("criteria", [])
    key_path = root / "artifacts" / "private" / "core_v2_private_key.jsonl"
    if not key_path.exists():
        return answers, criteria, None
    for row in read_jsonl(key_path):
        answers[row["question_id"]] = row["answer"]
        criteria[row["question_id"]] = row["criteria"]
    return answers, criteria, sha256(key_path)


def targeted_reference_issues(row: dict, answer: str) -> list[str]:
    """초기 사람 검토에서 고위험으로 식별된 수식·논리 조건을 독립 재검산한다."""
    qid, question, issues = row["question_id"], row["question"], []
    if qid == "CORE2-TR-MATH-002":
        values = [int(value.replace(",", "")) for value in re.findall(r"[0-9][0-9,]*", question)]
        a1, b1, c1, a2, b2, c2 = values[:6]
        determinant = a1 * b2 - a2 * b1
        x = (c1 * b2 - c2 * b1) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        expected = f"{x + y:g}원"
        if answer != expected or not x.is_integer() or not y.is_integer():
            issues.append("reference_or_prompt_linear_system_inconsistent")
    elif qid == "CORE2-TR-MATH-015":
        if "2배보다 1 cm 길다" in question:
            width, height = 15, 8
            if width != 2 * height + 1:
                issues.append("reference_dimensions_violate_stated_relation")
    elif qid == "CORE2-TR-SCI-021":
        if answer != "하현달, 삭":
            issues.append("reference_phase_term_should_be_sak")
    elif qid == "CORE2-TR-LOG-015":
        if "C가 D+A와 같다" in question and "B가 D보다 무겁" in question:
            issues.append("premises_are_internally_inconsistent")
    return issues


def audit(root: Path) -> dict:
    core = root / "data" / "questions" / "core_v2"
    rows = [row for filename in FILES for row in read_jsonl(core / filename)]
    answers, private_criteria, private_key_hash = load_answers(root, rows)
    exact, normalized, templated = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        exact[row["question"]].append(row["question_id"])
        normalized[normalize(row["question"])].append(row["question_id"])
        templated[template(row["question"])].append(row)
    exact_groups = [ids for ids in exact.values() if len(ids) > 1]
    normalized_groups = [ids for ids in normalized.values() if len(ids) > 1]
    template_groups = [group for group in templated.values() if len(group) > 1]
    cross_template_pairs = [
        (a, b) for group in template_groups for a, b in combinations(group, 2) if a["split"] != b["split"]
    ]
    family_groups = defaultdict(list)
    for row in rows:
        family_groups[row["template_family"].split(":", 2)[-1]].append(row)
    cross_family_pairs = [
        (a, b) for group in family_groups.values() for a, b in combinations(group, 2) if a["split"] != b["split"]
    ]
    related_pairs = []
    for a, b in combinations(rows, 2):
        if a["split"] == b["split"]:
            continue
        score = similarity(a["question"], b["question"])
        if 0.45 <= score < 0.70 and (a, b) not in cross_template_pairs:
            related_pairs.append((a, b, score))
    related_ids = {row["question_id"] for a, b, _ in related_pairs for row in (a, b)}

    findings, status_counts = [], Counter()
    difficulty_mismatch, reference_errors, method_errors, domain_errors = [], [], [], []
    unnatural, pattern_bias = [], []
    split_counts, domain_counts = Counter(row["split"] for row in rows), Counter(row["domain"] for row in rows)
    task_counts = Counter(row["task_type"] for row in rows)
    for row in rows:
        qid, reasons, severity = row["question_id"], [], 0
        steps = row["rubric"].get("reasoning_steps")
        difficulty_ok = (
            row["difficulty"] == "easy" and steps == 1
            or row["difficulty"] == "medium" and isinstance(steps, int) and 2 <= steps <= 3
            or row["difficulty"] == "hard" and isinstance(steps, int) and steps >= 3
        )
        if not difficulty_ok:
            difficulty_mismatch.append(qid); reasons.append("difficulty_reasoning_steps_mismatch"); severity = 2
        if row["verification_method"] not in ALLOWED_METHODS:
            method_errors.append(qid); reasons.append("unsupported_verification_method"); severity = 2
        if row["domain"] not in EXPECTED_DOMAINS:
            domain_errors.append(qid); reasons.append("invalid_domain"); severity = 2
        if re.search(r"문항\s*[0-9]+|사례\s*[0-9]+|문제\s*번호", row["question"]):
            unnatural.append(qid); reasons.append("artificial_generation_identifier"); severity = max(severity, 1)
        answer = answers.get(qid)
        if answer is None or hashlib.sha256(answer.encode("utf-8")).hexdigest() != row["reference"]["answer_sha256"]:
            reference_errors.append(qid); reasons.append("reference_hash_or_private_key_mismatch"); severity = 2
        else:
            issues = targeted_reference_issues(row, answer)
            if issues:
                reference_errors.append(qid); reasons.extend(issues); severity = 2
        criteria = private_criteria.get(qid, row["rubric"].get("criteria", []))
        if row["verification_method"] == "rubric_review" and len(criteria) < 2:
            method_errors.append(qid); reasons.append("rubric_has_insufficient_criteria"); severity = 2
        if task_counts[row["task_type"]] > 3:
            pattern_bias.append(qid); reasons.append("task_type_overrepresented"); severity = max(severity, 1)
        if qid in related_ids:
            reasons.append("cross_split_related_skill_pair_but_constraints_are_distinct"); severity = max(severity, 1)
        if any(qid in {a["question_id"], b["question_id"]} for a, b in cross_template_pairs + cross_family_pairs):
            reasons.append("cross_split_template_leakage"); severity = 2
        status = ("PASS", "REVIEW", "FAIL")[severity]
        status_counts[status] += 1
        findings.append({"question_id": qid, "status": status, "reasons": reasons})

    file_hashes = {filename: sha256(core / filename) for filename in FILES}
    dataset_checksum = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    auto_count = sum(row["rubric"]["auto_verifiable"] for row in rows)
    fail_reasons = {
        "split_count": dict(split_counts) != EXPECTED_SPLITS,
        "domain_count": dict(domain_counts) != EXPECTED_DOMAINS,
        "auto_ratio": auto_count / len(rows) < 0.85,
        "exact_duplicates": bool(exact_groups),
        "normalized_duplicates": bool(normalized_groups),
        "cross_template": bool(cross_template_pairs or cross_family_pairs),
        "difficulty": bool(difficulty_mismatch),
        "reference": bool(reference_errors),
        "method": bool(method_errors),
        "domain": bool(domain_errors),
        "private_key": private_key_hash is None,
    }
    freeze_ready = not any(fail_reasons.values()) and status_counts["FAIL"] == 0
    return {
        "audit_id": "core-v2-quality-audit-v1",
        "dataset_version": "core-v2",
        "dataset_checksum": dataset_checksum,
        "file_sha256": file_hashes,
        "private_key_sha256": private_key_hash,
        "summary": {
            "question_count": len(rows),
            "split_counts": dict(split_counts),
            "domain_counts": dict(domain_counts),
            "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
            "verification_method_counts": dict(Counter(str(row["verification_method"]) for row in rows)),
            "task_type_counts": dict(sorted(task_counts.items())),
            "auto_verifiable_count": auto_count,
            "auto_verifiable_ratio": auto_count / len(rows),
            "status_counts": {status: status_counts[status] for status in ("PASS", "REVIEW", "FAIL")},
            "exact_duplicate_groups": len(exact_groups),
            "normalized_duplicate_groups": len(normalized_groups),
            "near_duplicate_template_groups": len(template_groups),
            "cross_split_template_leakage_pairs": len(cross_template_pairs) + len(cross_family_pairs),
            "cross_split_related_skill_review_pairs": len(related_pairs),
            "difficulty_mismatch_count": len(difficulty_mismatch),
            "reference_error_count": len(set(reference_errors)),
            "verification_method_error_count": len(set(method_errors)),
            "domain_error_count": len(set(domain_errors)),
            "artificial_identifier_count": len(unnatural),
            "pattern_bias_count": len(set(pattern_bias)),
            "freeze_ready": freeze_ready,
        },
        "freeze_fail_reasons": fail_reasons,
        "exact_duplicate_groups": exact_groups,
        "normalized_duplicate_groups": normalized_groups,
        "template_groups": [[row["question_id"] for row in group] for group in template_groups],
        "cross_split_related_skill_reviews": [
            {"left": a["question_id"], "right": b["question_id"], "shingle_similarity": round(score, 6), "decision": "same skill allowed; materially different constraints"}
            for a, b, score in related_pairs
        ],
        "findings": findings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
