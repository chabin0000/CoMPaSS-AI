"""초기 Core-v2 감사에서 확인된 ID만 1회 수정한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "questions" / "core_v2"

PATCHES = {
    "CORE2-TR-MATH-002": {"question": "공책 2권과 펜 3자루는 8,000원이고, 공책 3권과 펜 1자루는 7,800원이다. 공책 한 권과 펜 한 자루의 가격 합은 얼마인가?", "answer": "3400원"},
    "CORE2-TR-MATH-015": {"question": "직사각형의 둘레는 46 cm이고 가로는 세로의 2배보다 1 cm 짧다. 가로와 세로를 구하시오."},
    "CORE2-TR-CODE-010": {"task_type": "loop_boundary_repair"},
    "CORE2-TR-CODE-022": {"task_type": "mutation_bug_repair"},
    "CORE2-VA-CODE-004": {"task_type": "method_call_bug"},
    "CORE2-PE-CODE-005": {"task_type": "representation_bug"},
    "CORE2-PE-CODE-010": {"task_type": "initialization_bug"},
    "CORE2-TR-SCI-007": {"task_type": "lever_balance", "question": "받침점에서 왼쪽 0.4 m 지점에 30 N의 힘이 작용한다. 오른쪽 0.6 m 지점에서 평형을 이루는 힘을 구하시오.", "verification_method": "deterministic_calculation", "answer": "20 N"},
    "CORE2-TR-SCI-021": {"answer": "하현달, 삭"},
    "CORE2-TR-LOG-004": {"task_type": "two_person_truth"},
    "CORE2-TR-LOG-012": {"task_type": "clue_truth"},
    "CORE2-TR-LOG-015": {"question": "저울에서 A+B가 C와 같고, C가 D+A보다 무겁다. B와 D 중 더 무거운 것을 쓰시오."},
    "CORE2-VA-LOG-002": {"task_type": "two_speaker_truth"},
    "CORE2-PE-LOG-001": {"task_type": "three_speaker_truth"},
    "CORE2-TR-KOR-002": {"difficulty": "medium"},
    "CORE2-TR-KOR-016": {"task_type": "lexical_choice", "question": "'새 규칙은 다음 달부터 시행된다'에서 '시행된다'와 가장 가까운 뜻을 '실행된다'와 '취소된다' 중 고르시오.", "verification_method": "exact_match", "answer": "실행된다", "reasoning_steps": 1},
    "CORE2-TR-KOR-023": {"difficulty": "medium"},
    "CORE2-PE-KOR-008": {"difficulty": "medium"},
}


def main() -> None:
    found: set[str] = set()
    for filename in ("train_questions.jsonl", "validation_questions.jsonl", "private_eval_questions.jsonl"):
        path = CORE / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            patch = PATCHES.get(row["question_id"])
            if patch is None:
                continue
            found.add(row["question_id"])
            for field in ("question", "task_type", "verification_method", "difficulty"):
                if field in patch:
                    row[field] = patch[field]
            if "reasoning_steps" in patch:
                row["rubric"]["reasoning_steps"] = patch["reasoning_steps"]
            if "answer" in patch:
                if row["split"] == "private_eval":
                    raise ValueError(f"private answer repair requires key update: {row['question_id']}")
                answer = patch["answer"]
                row["reference"]["answer"] = answer
                row["reference"]["answer_sha256"] = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            row["provenance"]["audit_repair"] = "core-v2-initial-audit"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    missing = set(PATCHES) - found
    if missing:
        raise ValueError(f"repair targets missing: {sorted(missing)}")
    print(json.dumps({"repaired_question_ids": sorted(found), "count": len(found)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
