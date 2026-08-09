"""명백한 표기 차이만 정규화하는 답변 비교기.

의미 기반 비교는 향후 별도 비교기를 연결한다. 이 모듈은 의미 동등성을 추측하지 않는다.
"""

from __future__ import annotations

import re


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", "", str(value).strip().lower())
    text = text.replace(",", "")
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else format(number, "g")
    except ValueError:
        return text


def classify_pair(a: dict | None, b: dict | None) -> str:
    if a is None or b is None:
        return "one_missing"
    if a.get("error_type") == "format_error" or b.get("error_type") == "format_error":
        return "format_conflict"
    a_answer, b_answer = normalize_answer(a.get("final_answer")), normalize_answer(b.get("final_answer"))
    if not a_answer or not b_answer or a_answer in {"unknown", "추가확인필요"} and b_answer in {"unknown", "추가확인필요"}:
        return "both_suspect"
    return "agree" if a_answer == b_answer else "answer_conflict"
