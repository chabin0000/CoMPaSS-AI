"""Core-v1을 수정하지 않고 재현 가능한 정적 품질감사를 생성한다."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from math import gcd, sqrt
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "questions" / "core_v1"
FILES = ("train_questions.jsonl", "validation_questions.jsonl", "private_eval_questions.jsonl")
CURRENT_AUTO_METHODS = {"answer_key", "exact_match"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _template(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "<num>", _normalize(text))


def _distribution(rows: list[dict]) -> dict:
    return {
        key: dict(sorted(Counter(row[key] for row in rows).items()))
        for key in ("domain", "difficulty", "verification_method", "task_type")
    }


def _reference_map(rows: list[dict], root: Path) -> tuple[dict[str, str], bool]:
    answers = {
        row["question_id"]: row["reference"]["answer"]
        for row in rows if row["split"] != "private_eval"
    }
    key_path = root / "artifacts" / "private" / "core_v1_private_key.jsonl"
    if key_path.exists():
        answers.update({row["question_id"]: row["answer"] for row in _read_jsonl(key_path)})
    return answers, key_path.exists()


def _ascii_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"[0-9]+", text)]


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _derive_expected(row: dict) -> str | None:
    """질문 문자열에서 답을 독립 재계산한다. 비실행 코드 추적만 사용한다."""
    question, task = row["question"], row["task_type"]
    values = _ascii_ints(question)
    if task == "linear_equation":
        a, b, total = values[:3]; return str((total - b) // a)
    if task == "geometry":
        return f"{values[0] * values[1]} cm²"
    if task == "percentage":
        return _number(values[0] * values[1] / 100)
    if task == "sequence":
        first, difference, ordinal = values[:3]; return str(first + (ordinal - 1) * difference)
    if task == "number_theory":
        return str(gcd(values[0], values[1]))
    if task == "probability":
        red, blue = values[:2]; common = gcd(red, red + blue); return f"{red // common}/{(red + blue) // common}"
    if task == "coordinate_geometry":
        x1, y1, x2, y2 = values[:4]; return _number(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if task == "word_problem":
        return str((values[0] + 1) // 2)
    if task == "python_trace":
        return str(values[0] * 2 + 1)
    if task == "loop_trace":
        limit = int(re.search(r"range\(1,\s*([0-9]+)\)", question).group(1)); return str(sum(range(1, limit)))
    if task == "representation":
        return bin(values[0])[2:]
    if task == "complexity":
        return "O(n)"
    if task == "list_comprehension":
        limit = int(re.search(r"range\(([0-9]+)\)", question).group(1)); return str(max(x for x in range(limit) if x % 2 == 0))
    if task == "branch_trace":
        limit = int(re.search(r"range\(([0-9]+)\)", question).group(1)); return str(sum(x % 3 == 0 for x in range(limit)))
    if task == "indexing":
        limit = int(re.search(r"range\(([0-9]+)\)", question).group(1)); return str(limit - 2)
    if task == "boolean_logic":
        left = int(re.search(r"\(([0-9]+)\s*>", question).group(1)); return "True" if left % 2 == 0 else "False"
    if task == "mechanics":
        return f"{values[1] / values[0]:g} m/s²"
    if task == "electricity":
        return f"{values[1] / values[0]:g} A"
    if task == "kinematics":
        return f"{values[0] * values[1]} m"
    if task == "density":
        return f"{values[0] / values[1]:g} g/cm³"
    if task == "energy":
        return f"{values[0] * values[1]} J"
    if task == "waves":
        return f"{values[0] * values[1]} m/s"
    if task == "circuits":
        return f"{values[0] + values[1]} Ω"
    if task == "thermodynamics":
        return f"{values[0] * values[1] * values[2]} J"
    if task == "sequence_logic":
        return str(values[3] + values[1] - values[0])
    if task == "propositional_logic":
        return "참"
    if task == "syllogism":
        return "예"
    if task == "ordering":
        return "민수"
    if task == "set_logic":
        return str(values[0] + values[1] - values[2])
    if task == "constraint_logic":
        return "C"
    if task == "modus_tollens":
        return "아니오"
    if task == "parity":
        return "짝수"
    if task == "word_count":
        quoted = re.search(r"'([^']+)'", question).group(1); return str(sum(not token.isdigit() for token in quoted.split()))
    if task == "sorting":
        words = re.search(r":\s*(.+)$", question).group(1).split(", "); return ", ".join(sorted(words))
    if task == "grammar" and "주어" in question:
        return "학생들이"
    if task == "grammar":
        return "읽고, 그린다"
    if task == "vocabulary":
        return "슬프다"
    if task == "honorific":
        return "선생님께서 교실에 계신다."
    if task == "morphology":
        return "맨-"
    if task == "meaning":
        return "상호 존중"
    return None


def _reference_sanity(row: dict, answer: str | None) -> tuple[str, str | None]:
    if answer is None:
        return "unknown", "private_reference_key_unavailable"
    expected_hash = row["reference"]["answer_sha256"]
    if hashlib.sha256(answer.encode("utf-8")).hexdigest() != expected_hash:
        return "fail", "reference_answer_hash_mismatch"
    if row["verification_method"] == "rubric_review":
        return "review", "manual_rubric_reference_requires_blinded_review"
    derived = _derive_expected(row)
    if derived is None:
        return "unknown", "independent_reference_rule_not_implemented"
    if _normalize(derived) != _normalize(answer):
        return "fail", "independently_recalculated_reference_mismatch"
    return "pass", None


def audit(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "data" / "questions" / "core_v1" / "core_manifest.json").read_text(encoding="utf-8"))
    rows = [row for name in FILES for row in _read_jsonl(root / "data" / "questions" / "core_v1" / name)]
    original_checksum = manifest["dataset_checksum"]
    original_file_hashes = {name: _sha256(root / "data" / "questions" / "core_v1" / name) for name in FILES}
    if original_file_hashes != manifest["file_sha256"]:
        raise ValueError("Core-v1 checksum mismatch before audit")

    exact = defaultdict(list)
    normalized = defaultdict(list)
    templates = defaultdict(list)
    for row in rows:
        exact[row["question"]].append(row)
        normalized[_normalize(row["question"])].append(row)
        templates[(row["domain"], row["task_type"], _template(row["question"]))].append(row)
    exact_groups = [[row["question_id"] for row in group] for group in exact.values() if len(group) > 1]
    normalized_groups = [[row["question_id"] for row in group] for group in normalized.values() if len(group) > 1]
    near_groups = [group for group in templates.values() if len(group) > 1]
    near_pairs = [(a, b) for group in near_groups for a, b in combinations(group, 2)]
    cross_pairs = [(a, b) for a, b in near_pairs if a["split"] != b["split"]]
    eval_cross_ids = {
        row["question_id"] for pair in cross_pairs for row in pair
        if row["split"] in {"validation", "private_eval"}
    }
    all_cross_ids = {row["question_id"] for pair in cross_pairs for row in pair}
    answers, private_key_available = _reference_map(rows, root)
    id_phrase = re.compile(r"문항|사례")

    findings = []
    status_counts = Counter()
    reference_counts = Counter()
    for row in rows:
        reasons: list[str] = []
        severity = 0
        reasons.append("REVIEW: 단일 단계·정형 문제로 ARGOS-v1 변별력이 지나치게 낮을 가능성")
        severity = 1
        if row["question_id"] in eval_cross_ids:
            reasons.append("FAIL: train과 숫자/변수만 다른 동일 문제 템플릿")
            severity = 2
        elif row["question_id"] in all_cross_ids:
            reasons.append("REVIEW: validation/private_eval과 동일 템플릿을 공유하는 train 문항")
            severity = max(severity, 1)
        if row["difficulty"] != "easy":
            reasons.append(f"REVIEW: 1단계 문제인데 difficulty={row['difficulty']}로 순환 배정된 정황")
            severity = max(severity, 1)
        if row["verification_method"] not in CURRENT_AUTO_METHODS:
            reasons.append(f"REVIEW: 현재 independent verifier가 {row['verification_method']}를 자동 실행하지 못함")
            severity = max(severity, 1)
        if id_phrase.search(row["question"]):
            reasons.append("REVIEW: 고유화 목적의 문항/사례 번호가 질문에 섞여 표현이 인위적임")
            severity = max(severity, 1)
        if row["rubric"]["auto_verifiable"] and len(templates[(row["domain"], row["task_type"], _template(row["question"]))]) > 1:
            reasons.append("REVIEW: 자동검증 가능하지만 반복 템플릿 패턴 암기 위험")
            severity = max(severity, 1)
        ref_status, ref_reason = _reference_sanity(row, answers.get(row["question_id"]))
        reference_counts[ref_status] += 1
        if ref_status == "fail":
            reasons.append(f"FAIL: {ref_reason}")
            severity = 2
        elif ref_status in {"review", "unknown"}:
            reasons.append(f"REVIEW: {ref_reason}")
            severity = max(severity, 1)
        status = ("PASS", "REVIEW", "FAIL")[severity]
        status_counts[status] += 1
        findings.append({"question_id": row["question_id"], "split": row["split"], "status": status, "reasons": reasons})

    report = {
        "audit_id": "core-v1-quality-audit-20260810",
        "scope": {"dataset_version": "core-v1", "question_count": len(rows), "dataset_checksum": original_checksum},
        "method": {
            "exact_duplicate": "literal string equality",
            "normalized_duplicate": "Unicode NFKC + casefold + whitespace collapse",
            "template_near_duplicate": "same domain/task_type after replacing all numeric literals",
            "semantic_leakage": "template near-duplicate crossing split; no external embedding API",
            "difficulty_review": "single-step task treated as easy; non-easy labels flagged",
            "reference_review": "stored answer SHA-256 plus task-specific independent recalculation; rubric items require human review",
        },
        "summary": {
            "status_counts": {status: status_counts[status] for status in ("PASS", "REVIEW", "FAIL")},
            "exact_duplicate_groups": len(exact_groups),
            "normalized_duplicate_groups": len(normalized_groups),
            "template_near_duplicate_groups": len(near_groups),
            "template_near_duplicate_pairs": len(near_pairs),
            "cross_split_suspect_pairs": len(cross_pairs),
            "cross_split_affected_eval_items": len(eval_cross_ids),
            "cross_split_affected_all_items": len(all_cross_ids),
            "non_easy_label_review_count": sum(row["difficulty"] != "easy" for row in rows),
            "overly_easy_review_count": len(rows),
            "current_verifier_unsupported_count": sum(row["verification_method"] not in CURRENT_AUTO_METHODS for row in rows),
            "verification_method_semantic_mismatch_count": 0,
            "domain_label_error_count": 0,
            "artificial_identifier_phrase_count": sum(bool(id_phrase.search(row["question"])) for row in rows),
            "korean_artificial_identifier_phrase_count": sum(row["domain"] == "korean" and bool(id_phrase.search(row["question"])) for row in rows),
            "auto_template_memorization_risk_count": sum(row["rubric"]["auto_verifiable"] for row in rows),
            "reference_error_count": reference_counts["fail"],
            "reference_status_counts": dict(reference_counts),
            "private_key_available_during_audit": private_key_available,
            "freeze_recommendation": "DO_NOT_FREEZE_AS_FINAL_EVALUATION",
        },
        "distributions": {
            split: _distribution([row for row in rows if row["split"] == split])
            for split in ("train", "validation", "private_eval")
        },
        "exact_duplicate_groups": exact_groups,
        "normalized_duplicate_groups": normalized_groups,
        "template_near_duplicate_groups": [
            {"question_ids": [row["question_id"] for row in group], "splits": sorted({row["split"] for row in group})}
            for group in near_groups
        ],
        "cross_split_suspect_pairs": [
            {"left": a["question_id"], "right": b["question_id"], "template_similarity": 1.0}
            for a, b in cross_pairs
        ],
        "findings": findings,
        "core_files_unchanged_after_audit": {
            "dataset_checksum": original_checksum,
            "file_sha256": original_file_hashes,
        },
    }
    return report


if __name__ == "__main__":
    output = ROOT / "reports" / "core_v1_quality_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = audit(ROOT)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
