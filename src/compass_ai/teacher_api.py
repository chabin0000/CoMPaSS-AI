"""OpenAI-compatible 교사 endpoint 통신과 사전 점검."""

from __future__ import annotations

import json
import hashlib
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .jsonl import read_jsonl, write_jsonl
from .schemas import SchemaError


class TeacherAPIError(RuntimeError):
    """교사 API 연결 또는 응답 계약 오류."""

    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type


def load_teacher_api_config(path: Path) -> dict:
    """JSON 또는 YAML 설정을 읽고 실제 실행에 필요한 필드를 검증한다."""
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - 배포 환경 진단용
            raise SchemaError("YAML config requires PyYAML") from error
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise SchemaError("teacher config must be an object")
    required = {
        "teacher_name", "experiment_id", "dataset_name", "dataset_version", "dataset_checksum",
        "base_url", "model", "timeout_seconds", "system_prompt", "generation", "extra_body", "model_metadata",
    }
    missing = required - config.keys()
    if missing:
        raise SchemaError(f"teacher config missing fields: {', '.join(sorted(missing))}")
    supported = {
        ("compass-ai-pilot", "pilot-v1"),
        ("compass-ai-core", "core-v2"),
    }
    if (config["dataset_name"], config["dataset_version"]) not in supported:
        raise SchemaError("teacher config must target frozen Pilot-v1 or Core-v2")
    if not isinstance(config["dataset_checksum"], str) or len(config["dataset_checksum"]) != 64:
        raise SchemaError("teacher config dataset_checksum must be SHA-256")
    if not isinstance(config["generation"], dict) or not isinstance(config["extra_body"], dict):
        raise SchemaError("generation and extra_body must be objects")
    if not isinstance(config["timeout_seconds"], (int, float)) or config["timeout_seconds"] <= 0:
        raise SchemaError("timeout_seconds must be positive")
    return config


@dataclass(frozen=True)
class TeacherAnswer:
    raw_response: dict
    final_answer: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None


class OpenAICompatibleTeacher:
    """vLLM/SGLang/NIM에 종속되지 않는 OpenAI Chat Completions 클라이언트."""

    def __init__(self, config: dict):
        self.config = config
        self.base_url = str(config["base_url"]).rstrip("/")
        self.timeout = float(config["timeout_seconds"])

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, float]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        started = perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise TeacherAPIError(f"HTTP {error.code}: {detail}", "http_error") from error
        except (TimeoutError, socket.timeout) as error:
            raise TeacherAPIError(f"request timeout after {self.timeout:g}s", "timeout") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise TeacherAPIError(f"request timeout after {self.timeout:g}s", "timeout") from error
            raise TeacherAPIError(f"endpoint connection failed: {error.reason}", "connection_error") from error
        latency = perf_counter() - started
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TeacherAPIError("response is not valid JSON", "malformed_response") from error
        if not isinstance(decoded, dict):
            raise TeacherAPIError("response JSON must be an object", "malformed_response")
        return decoded, latency

    def models(self) -> list[str]:
        raw, _latency = self._request("GET", "/models")
        data = raw.get("data")
        if not isinstance(data, list):
            raise TeacherAPIError("/v1/models response has no data list", "malformed_response")
        return [str(item["id"]) for item in data if isinstance(item, dict) and "id" in item]

    def generate(self, question: dict) -> TeacherAnswer:
        payload = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": question["question"]},
            ],
            **self.config["generation"],
            **self.config["extra_body"],
        }
        raw, latency = self._request("POST", "/chat/completions", payload)
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise TeacherAPIError("chat response has no choices[0].message.content", "malformed_response") from error
        if not isinstance(content, str) or not content.strip():
            raise TeacherAPIError("chat response final content is empty", "malformed_response")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return TeacherAnswer(
            raw_response=raw,
            final_answer=content,
            latency_seconds=latency,
            input_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            output_tokens=usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
        )


def teacher_check(root: Path, config_path: Path, smoke: bool = False, output_root: Path | None = None, questions_path: Path | None = None) -> dict:
    """서버, model 목록, 선택적 Pilot 첫 문항 smoke 응답 구조를 검사한다."""
    from .teacher_pilot import load_frozen_pilot

    config = load_teacher_api_config(config_path)
    if config["dataset_version"] == "pilot-v1":
        questions, _manifest = load_frozen_pilot(root, config["dataset_checksum"])
    else:
        if questions_path is None:
            questions_path = root / "data" / "questions" / "core_v2" / "train_questions.jsonl"
        if hashlib.sha256(questions_path.read_bytes()).hexdigest() != config["dataset_checksum"]:
            raise TeacherAPIError("Core-v2 train checksum mismatch", "dataset_checksum_mismatch")
        questions = read_jsonl(questions_path)
    client = OpenAICompatibleTeacher(config)
    models = client.models()
    report = {
        "status": "ready",
        "endpoint_reachable": True,
        "models": models,
        "configured_model": config["model"],
        "model_available": config["model"] in models,
        "smoke": None,
        "smoke_output": None,
    }
    if not report["model_available"]:
        report["status"] = "error"
    if smoke:
        answer = client.generate(questions[0])
        smoke_row = {
            "question_id": questions[0]["question_id"],
            "raw_response": answer.raw_response,
            "final_answer": answer.final_answer,
            "latency_seconds": answer.latency_seconds,
            "ttft_seconds": None,
            "tokens_per_second": None,
        }
        smoke_root = output_root or root / "logs" / "teacher_smoke"
        smoke_path = smoke_root / config["teacher_name"] / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        write_jsonl(smoke_path / "smoke_response.jsonl", [smoke_row])
        report["smoke"] = smoke_row
        report["smoke_output"] = str(smoke_path)
    return report
