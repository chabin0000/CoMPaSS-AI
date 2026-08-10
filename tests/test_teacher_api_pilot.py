from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from compass_ai.jsonl import read_jsonl, write_jsonl
from compass_ai.pilot import create_pilot_manifest, write_pilot_manifest
from compass_ai.teacher_api import teacher_check
from compass_ai.teacher_api import load_teacher_api_config
from compass_ai.teacher_pilot import DatasetChecksumError, run_teacher_pilot
from compass_ai.teacher_run import comparison_readiness


def _question(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "split": "pilot",
        "domain": "stem",
        "task_type": "calculation",
        "question": f"{question_id} 질문",
        "reference_answer": "정답",
        "verification_method": "answer_key",
        "evidence_source": "test fixture",
        "difficulty": "easy",
        "requires_current_info": False,
        "notes": "rubric=R-CALC; origin=original",
    }


def _pilot_root(tmp_path: Path, count: int = 3) -> tuple[Path, str]:
    pilot_dir = tmp_path / "data" / "questions" / "pilot"
    questions_path = pilot_dir / "pilot_questions.jsonl"
    rows = [_question(f"P-{index:03d}") for index in range(1, count + 1)]
    write_jsonl(questions_path, rows)
    manifest = create_pilot_manifest(questions_path, rows)
    write_pilot_manifest(pilot_dir / "pilot_manifest.json", manifest)
    return tmp_path, manifest["source_file_sha256"]


def _write_config(path: Path, base_url: str, checksum: str, **overrides: object) -> Path:
    config = {
        "teacher_name": "gpt_oss",
        "experiment_id": "pilot-test",
        "dataset_name": "compass-ai-pilot",
        "dataset_version": "pilot-v1",
        "dataset_checksum": checksum,
        "base_url": base_url,
        "model": "fake/teacher-model",
        "timeout_seconds": 1,
        "system_prompt": "공통 시스템 프롬프트",
        "generation": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 7},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "model_metadata": {
            "checkpoint": "fake-checkpoint",
            "revision": "test-revision",
            "quantization": None,
            "inference_engine": None,
            "server_config": {"tensor_parallel_size": None},
        },
    }
    config.update(overrides)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@contextmanager
def _fake_openai_server(mode: str = "ok"):
    state = {"requests": [], "failed_once": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "fake/teacher-model", "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            state["requests"].append(request)
            question = request["messages"][-1]["content"]
            if mode == "timeout":
                time.sleep(0.15)
            if mode == "http_error":
                self._json(500, {"error": {"message": "fake failure"}})
                return
            if mode == "malformed":
                self._json(200, {"id": "bad", "choices": []})
                return
            if mode == "fail_once" and "P-002" in question and not state["failed_once"]:
                state["failed_once"] = True
                self._json(500, {"error": {"message": "retry me"}})
                return
            self._json(
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": "fake/teacher-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "학습 답안에 섞이면 안 되는 사고 과정",
                                "content": f"최종 답변: {question}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_three_question_run_saves_full_raw_json_and_separate_final_answer(tmp_path: Path):
    root, checksum = _pilot_root(tmp_path)
    with _fake_openai_server() as (base_url, state):
        config = _write_config(tmp_path / "teacher.yaml", base_url, checksum)
        report = run_teacher_pilot(root, "gpt_oss", config, output_root=tmp_path / "teacher-runs")

    assert report["status"] == "complete"
    assert report["completed_count"] == 3
    rows = read_jsonl(Path(report["responses_path"]))
    assert [row["question_id"] for row in rows] == ["P-001", "P-002", "P-003"]
    assert rows[0]["raw_response"]["choices"][0]["message"]["reasoning_content"].startswith("학습")
    assert rows[0]["final_answer"] == "최종 답변: P-001 질문"
    assert "사고 과정" not in rows[0]["final_answer"]
    assert rows[0]["ttft_seconds"] is None
    assert rows[0]["tokens_per_second"] is None
    assert state["requests"][0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_resume_keeps_completed_ids_and_retries_only_failed_id(tmp_path: Path):
    root, checksum = _pilot_root(tmp_path)
    with _fake_openai_server("fail_once") as (base_url, state):
        config = _write_config(tmp_path / "teacher.yaml", base_url, checksum)
        first = run_teacher_pilot(root, "gpt_oss", config, output_root=tmp_path / "teacher-runs")
        second = run_teacher_pilot(root, "gpt_oss", config, output_root=tmp_path / "teacher-runs")

    assert first["status"] == "incomplete"
    assert first["completed_count"] == 2
    assert second["status"] == "complete"
    assert second["completed_count"] == 3
    assert len(state["requests"]) == 4
    assert len(read_jsonl(Path(second["responses_path"]))) == 3
    errors = read_jsonl(Path(second["errors_path"]))
    assert errors[0]["question_id"] == "P-002"
    assert "HTTP 500" in errors[0]["error"]


@pytest.mark.parametrize("mode", ["timeout", "http_error", "malformed"])
def test_timeout_http_error_and_malformed_response_are_logged(tmp_path: Path, mode: str):
    root, checksum = _pilot_root(tmp_path, count=1)
    with _fake_openai_server(mode) as (base_url, _state):
        timeout = 0.03 if mode == "timeout" else 1
        config = _write_config(tmp_path / "teacher.yaml", base_url, checksum, timeout_seconds=timeout)
        report = run_teacher_pilot(root, "gpt_oss", config, output_root=tmp_path / "teacher-runs")

    assert report["status"] == "incomplete"
    assert report["completed_count"] == 0
    error = read_jsonl(Path(report["errors_path"]))[0]
    assert error["question_id"] == "P-001"
    assert error["error_type"] in {"timeout", "http_error", "malformed_response"}


def test_dataset_checksum_mismatch_stops_before_any_request(tmp_path: Path):
    root, _checksum = _pilot_root(tmp_path)
    with _fake_openai_server() as (base_url, state):
        config = _write_config(tmp_path / "teacher.yaml", base_url, "0" * 64)
        with pytest.raises(DatasetChecksumError, match="checksum"):
            run_teacher_pilot(root, "gpt_oss", config, output_root=tmp_path / "teacher-runs")
    assert state["requests"] == []


def test_teacher_check_models_and_smoke_use_separate_output(tmp_path: Path):
    root, checksum = _pilot_root(tmp_path, count=1)
    with _fake_openai_server() as (base_url, _state):
        config = _write_config(tmp_path / "teacher.yaml", base_url, checksum)
        report = teacher_check(root, config, smoke=True, output_root=tmp_path / "teacher-smoke")

    assert report["status"] == "ready"
    assert report["model_available"] is True
    assert report["smoke"]["final_answer"] == "최종 답변: P-001 질문"
    assert "reasoning_content" in report["smoke"]["raw_response"]["choices"][0]["message"]
    assert "teacher-runs" not in report["smoke_output"]


def _raw_row(question_id: str, teacher: str, common_checksum: str, full_checksum: str) -> dict:
    return {
        "question_id": question_id,
        "teacher_model": teacher,
        "dataset_version": "pilot-v1",
        "dataset_checksum": "a" * 64,
        "run_id": f"run-{teacher}",
        "raw_response": {"choices": [{"message": {"content": "answer"}}]},
        "final_answer": "answer",
        "generation_config_reference": common_checksum,
        "response_started_at": "2026-08-10T00:00:00+00:00",
        "response_finished_at": "2026-08-10T00:00:01+00:00",
        "latency_seconds": 1.0,
        "ttft_seconds": None,
        "tokens_per_second": None,
        "input_tokens": None,
        "output_tokens": None,
        "response_status": "answered",
        "error": None,
        "metadata": {"full_config_checksum": full_checksum, "model_specific": {"revision": teacher}},
    }


def test_teacher_comparison_allows_documented_model_specific_settings_but_rejects_common_generation_mismatch(tmp_path: Path):
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a_path, [_raw_row("P-001", "teacher-a", "c" * 64, "1" * 64)])
    write_jsonl(b_path, [_raw_row("P-001", "teacher-b", "c" * 64, "2" * 64)])
    ready = comparison_readiness(a_path, b_path, expected_dataset_checksum="a" * 64)
    assert ready["status"] == "ready"
    assert ready["model_specific_settings_recorded"] is True

    write_jsonl(b_path, [_raw_row("P-001", "teacher-b", "d" * 64, "2" * 64)])
    mismatch = comparison_readiness(a_path, b_path, expected_dataset_checksum="a" * 64)
    assert mismatch["status"] == "error"
    assert "generation_config_mismatch" in mismatch["errors"]


def test_pre_dgx_check_does_not_rewrite_frozen_manifest(tmp_path: Path):
    from compass_ai.pre_dgx import pre_dgx_check

    root, _checksum = _pilot_root(tmp_path, count=1)
    (root / "docs").mkdir()
    (root / "docs" / "dgx_runbook.md").write_text("runbook", encoding="utf-8")
    private = root / "evaluation" / "private_final"
    private.mkdir(parents=True)
    (private / "README.md").write_text("placeholder", encoding="utf-8")
    manifest_path = root / "data" / "questions" / "pilot" / "pilot_manifest.json"
    before = manifest_path.read_bytes()

    result = pre_dgx_check(root, run_tests=False)

    assert result["status"] == "ready"
    assert manifest_path.read_bytes() == before


def test_dgx_teacher_configs_share_frozen_dataset_and_common_generation():
    root = Path(__file__).resolve().parents[1]
    gpt = load_teacher_api_config(root / "configs" / "teachers" / "gpt_oss_dgx.yaml")
    glm = load_teacher_api_config(root / "configs" / "teachers" / "glm45_air_dgx.yaml")

    assert gpt["dataset_checksum"] == glm["dataset_checksum"] == "07cae48e5e5e0ca020fba3455beb33dd45128273fbf4b9c6ad2ce053fe983c92"
    assert gpt["system_prompt"] == glm["system_prompt"]
    assert gpt["generation"] == glm["generation"]
    assert gpt["model_metadata"]["inference_engine"] is None
    assert glm["model_metadata"]["inference_engine"] is None
