"""교사 백엔드 인터페이스와 소프트웨어 검증용 Mock 교사."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from .comparison import classify_pair


class TeacherBackend(ABC):
    model_name: str

    @abstractmethod
    def generate(self, question: dict) -> dict | None:
        """질문 하나에 대한 표준 교사 응답을 생성한다."""


class _MockTeacher(TeacherBackend):
    def __init__(self, model_name: str, side: str):
        self.model_name, self.side = model_name, side

    def generate(self, question: dict) -> dict | None:
        case, answer = question.get("mock_case", "same"), question["reference_answer"]
        if case == "missing" and self.side == "B":
            return None
        wrong = "오답"
        response_answer, error_type = answer, None
        if case == "normalized" and self.side == "B":
            response_answer = f" {answer}.0 " if answer.isdigit() else answer
        elif case == "a_correct" and self.side == "B":
            response_answer = wrong
        elif case == "b_correct" and self.side == "A":
            response_answer = wrong
        elif case == "conflict" and self.side == "B":
            response_answer = wrong
        elif case == "format" and self.side == "B":
            error_type = "format_error"
        elif case in {"both_wrong", "plausible"}:
            response_answer = "추가 확인 필요" if case == "plausible" else wrong
        return {
            "question_id": question["question_id"], "domain": question["domain"], "task_type": question["task_type"],
            "question": question["question"], "teacher_model": self.model_name, "checkpoint": "mock-v1",
            "quantization": "none", "generation_config": {"temperature": 0, "mock_case": case},
            "raw_response": response_answer, "final_answer": response_answer, "verification_status": "pending",
            "verification_method": question["verification_method"], "evidence": "mock backend output",
            "error_type": error_type, "tag": "teacher_raw", "reviewer": None, "verification_outcome": "pending",
            "timestamp": datetime.now(UTC).isoformat(), "training_approved": False,
        }


class MockTeacherA(_MockTeacher):
    def __init__(self):
        super().__init__("MockTeacherA", "A")


class MockTeacherB(_MockTeacher):
    def __init__(self):
        super().__init__("MockTeacherB", "B")


class _FutureBackend(TeacherBackend):
    """실제 모델을 Mock으로 대체하지 않기 위한 명시적 확장 지점."""

    def generate(self, question: dict) -> dict | None:
        raise NotImplementedError(f"{self.model_name} backend is not implemented; do not fall back to MockTeacher")


class GPTOSSBackend(_FutureBackend):
    model_name = "openai/gpt-oss-120b"


class GLMBackend(_FutureBackend):
    model_name = "zai-org/GLM-4.5-Air"


def generate_responses(questions: list[dict], backends: list[TeacherBackend] | None = None) -> list[dict]:
    result: list[dict] = []
    for question in questions:
        for backend in backends or [MockTeacherA(), MockTeacherB()]:
            generated = backend.generate(question)
            if generated is not None:
                result.append(generated)
    return result


def merge_responses(records: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = {}
    for record in records:
        grouped.setdefault(record["question_id"], {})[record["teacher_model"]] = record
    merged = []
    for question_id, pair in sorted(grouped.items()):
        a, b = pair.get("MockTeacherA"), pair.get("MockTeacherB")
        merged.append({"question_id": question_id, "teacher_a": a, "teacher_b": b, "conflict_type": classify_pair(a, b)})
    return merged
