"""S0~S5 학습 조건 데이터 구성 및 비교 통계."""

from __future__ import annotations

from collections import Counter
from typing import Any


class MatchedComparisonError(ValueError):
    """주실험 paired comparison의 통제 조건이 깨졌을 때 발생한다."""


def _by_question(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in records:
        grouped.setdefault(item["question_id"], []).append(item)
    return grouped


def assert_matched_comparison(s3_matched: list[dict], s4_teacher_only: list[dict]) -> None:
    """주실험 비교군의 질문 ID·표본 수·영역 분포가 같은지 검증한다."""
    s3_ids = {item["question_id"] for item in s3_matched}
    s4_ids = {item["question_id"] for item in s4_teacher_only}
    if s3_ids != s4_ids:
        raise MatchedComparisonError("question_id 집합이 S4_teacher_only와 다릅니다")
    if len(s3_matched) != len(s4_teacher_only):
        raise MatchedComparisonError("sample count가 S4_teacher_only와 다릅니다")
    s3_domains = Counter(item["domain"] for item in s3_matched)
    s4_domains = Counter(item["domain"] for item in s4_teacher_only)
    if s3_domains != s4_domains:
        raise MatchedComparisonError("domain distribution이 S4_teacher_only와 다릅니다")


def build_conditions(questions: list[dict], records: list[dict], korean_source: list[dict] | None = None) -> dict[str, Any]:
    """S3a/S3b와 사람 보정 포함 여부가 다른 S4 조건을 함께 만든다."""
    grouped = _by_question(records)
    s1 = [item for item in records if item["teacher_model"] == "MockTeacherA"]
    s2 = [item for item in records if item["teacher_model"] == "MockTeacherB"]
    teacher_candidates = [item for item in records if item["teacher_model"] in {"MockTeacherA", "MockTeacherB"}]
    s3a = list(teacher_candidates)
    s3b, selections = [], []
    for index, question in enumerate(sorted(questions, key=lambda row: row["question_id"])):
        candidates = sorted([item for item in grouped.get(question["question_id"], []) if item["teacher_model"] in {"MockTeacherA", "MockTeacherB"}], key=lambda item: item["teacher_model"])
        if candidates:
            # 답의 품질·검증 결과와 무관한 질문 ID 순서 기반 고정 규칙이다.
            preferred = "MockTeacherA" if index % 2 == 0 else "MockTeacherB"
            selected = next((item for item in candidates if item["teacher_model"] == preferred), candidates[0])
            row = dict(selected)
            row["condition_selection"] = {"condition": "S3b", "rule": "alternating_sorted_question_id_no_quality_signal", "preferred_teacher": preferred, "fallback_used": selected["teacher_model"] != preferred}
            s3b.append(row)
            selections.append({"question_id": question["question_id"], "preferred_teacher": preferred, "selected_teacher": selected["teacher_model"], "fallback_used": selected["teacher_model"] != preferred})
    s4_with_human = [item for item in records if item.get("training_approved") and item.get("verification_outcome") in {"teacher_a_selected", "teacher_b_selected", "both_agree_verified", "human_corrected"}]
    s4_teacher_only = [item for item in s4_with_human if item["verification_outcome"] != "human_corrected"]
    matched_ids = {item["question_id"] for item in s4_teacher_only}
    s3_matched, matched_selections = [], []
    matched_counts = Counter()
    for index, question in enumerate(sorted((row for row in questions if row["question_id"] in matched_ids), key=lambda row: row["question_id"])):
        candidates = sorted([item for item in grouped.get(question["question_id"], []) if item["teacher_model"] in {"MockTeacherA", "MockTeacherB"}], key=lambda item: item["teacher_model"])
        if not candidates:
            raise MatchedComparisonError(f"S3_matched 후보가 없는 question_id: {question['question_id']}")
        # 검증 결과가 아닌 이미 선택된 교사 수만 사용해 가능한 균형을 맞춘다.
        lowest = min(matched_counts[item["teacher_model"]] for item in candidates)
        eligible = [item for item in candidates if matched_counts[item["teacher_model"]] == lowest]
        preferred = "MockTeacherA" if index % 2 == 0 else "MockTeacherB"
        selected = next((item for item in eligible if item["teacher_model"] == preferred), eligible[0])
        matched_counts[selected["teacher_model"]] += 1
        row = dict(selected)
        row["condition_selection"] = {"condition": "S3_matched", "rule": "balanced_count_then_alternating_sorted_question_id_no_quality_signal", "preferred_teacher": preferred, "fallback_used": selected["teacher_model"] != preferred}
        s3_matched.append(row)
        matched_selections.append({"question_id": question["question_id"], "preferred_teacher": preferred, "selected_teacher": selected["teacher_model"], "fallback_used": selected["teacher_model"] != preferred})
    assert_matched_comparison(s3_matched, s4_teacher_only)
    source = list(korean_source or [])
    s5 = s4_with_human + source if source else []
    return {
        "S0": [], "S1": s1, "S2": s2, "S3a": s3a, "S3b": s3b, "S3_matched": s3_matched,
        "S4": s4_with_human, "S4_teacher_only": s4_teacher_only,
        "S4_with_human_correction": s4_with_human, "S5": s5,
        "selection_metadata": {"rule": "alternating_sorted_question_id_no_quality_signal", "random_used": False, "seed": None, "selections": selections},
        "condition_metadata": {"S3_matched": {"primary_comparison": True, "reference_scope": "S4_teacher_only_question_ids_only", "selection_rule": "balanced_count_then_alternating_sorted_question_id_no_quality_signal", "selection_results": matched_selections, "validation": "passed"}, "S5": {"status": "executed" if source else "placeholder", "reason": "verified korean_source input is required" if not source else None}, "S4": {"human_correction_separated": True}},
    }


def condition_stats(records: list[dict]) -> dict:
    lengths = [len(str(item.get("final_answer", ""))) for item in records]
    return {
        "question_ids": sorted({item["question_id"] for item in records}), "question_id_coverage": len({item["question_id"] for item in records}), "sample_count": len(records),
        "teacher_counts": dict(Counter(item.get("teacher_model", "korean_source") for item in records)),
        "domain_counts": dict(Counter(item.get("domain", "unknown") for item in records)),
        "character_count": sum(lengths), "average_answer_length": sum(lengths) / len(lengths) if lengths else 0,
        "teacher_composition": {teacher: count / len(records) for teacher, count in Counter(item.get("teacher_model", "korean_source") for item in records).items()} if records else {},
        "approximate_token_count": sum(max(1, (length + 3) // 4) for length in lengths), "token_count_kind": "approximate",
        "human_corrected_count": sum(item.get("verification_outcome") == "human_corrected" for item in records),
        "human_corrected_ratio": sum(item.get("verification_outcome") == "human_corrected" for item in records) / len(records) if records else 0,
    }
