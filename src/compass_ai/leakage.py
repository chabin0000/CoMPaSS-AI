"""비공개 최종 평가셋에 대한 구조적 접근 차단과 중복 검사."""

from __future__ import annotations

from pathlib import Path


def assert_private_final_is_safe(path: Path) -> None:
    allowed = {"README.md", ".gitkeep", "schema.json"}
    disallowed = [item.name for item in path.iterdir() if item.name not in allowed]
    if disallowed:
        raise ValueError(f"private_final에는 평가 문항을 둘 수 없습니다: {', '.join(disallowed)}")


def find_id_overlap(training_ids: list[str], evaluation_ids: list[str]) -> list[str]:
    return sorted(set(training_ids) & set(evaluation_ids))


def assert_no_evaluation_leakage(training_ids: list[str], evaluation_ids: list[str]) -> None:
    overlap = find_id_overlap(training_ids, evaluation_ids)
    if overlap:
        raise ValueError(f"학습/평가 question_id 중복: {', '.join(overlap)}")
