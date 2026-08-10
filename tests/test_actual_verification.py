from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from compass_ai.jsonl import read_jsonl, write_jsonl
from compass_ai.pilot import create_pilot_manifest, write_pilot_manifest
from compass_ai.student_data import StudentDatasetError, build_student_dataset
from compass_ai.teacher_verification import apply_verification_review, verify_teacher_runs, verification_readiness


def _question(question_id: str, answer: str, method: str = "answer_key") -> dict:
    return {
        "question_id": question_id,
        "split": "pilot",
        "domain": "stem",
        "task_type": "calculation",
        "question": f"{question_id} 질문",
        "reference_answer": answer,
        "verification_method": method,
        "evidence_source": "independent fixture key",
        "difficulty": "easy",
        "requires_current_info": False,
        "notes": "rubric=R-TEST; origin=original",
    }


def _raw(question_id: str, teacher: str, answer: str, checksum: str, run_id: str) -> dict:
    return {
        "question_id": question_id,
        "teacher_model": teacher,
        "dataset_version": "pilot-v1",
        "dataset_checksum": checksum,
        "run_id": run_id,
        "raw_response": {"choices": [{"message": {"reasoning_content": "비공개 사고", "content": answer}}]},
        "final_answer": answer,
        "generation_config_reference": "c" * 64,
        "response_started_at": "2026-08-10T00:00:00+00:00",
        "response_finished_at": "2026-08-10T00:00:01+00:00",
        "latency_seconds": 1.0,
        "ttft_seconds": None,
        "tokens_per_second": None,
        "input_tokens": None,
        "output_tokens": None,
        "response_status": "answered",
        "error": None,
        "metadata": {"full_config_checksum": "1" * 64, "model_specific": {"revision": teacher}},
    }


def test_actual_teacher_verification_handles_a_to_g_and_preserves_provenance(tmp_path: Path):
    questions = [
        _question("A", "정답A"),
        _question("B", "정답B"),
        _question("C", "정답C"),
        _question("D", "정답D"),
        _question("E", "사람 검토", method="human_rubric"),
        _question("F", "정답F"),
        _question("G", "정답G"),
    ]
    question_path = tmp_path / "questions.jsonl"
    left_path, right_path = tmp_path / "left" / "responses.jsonl", tmp_path / "right" / "responses.jsonl"
    write_jsonl(question_path, questions)
    checksum = hashlib.sha256(question_path.read_bytes()).hexdigest()
    write_jsonl(left_path, [
        _raw("A", "teacher-a", "정답A", checksum, "run-a"),
        _raw("B", "teacher-a", "정답B", checksum, "run-a"),
        _raw("C", "teacher-a", "오답", checksum, "run-a"),
        _raw("D", "teacher-a", "오답1", checksum, "run-a"),
        _raw("E", "teacher-a", "검토 후보 A", checksum, "run-a"),
        _raw("G", "teacher-a", "정답G", checksum, "run-a"),
    ])
    write_jsonl(right_path, [
        _raw("A", "teacher-b", "정답A", checksum, "run-b"),
        _raw("B", "teacher-b", "오답", checksum, "run-b"),
        _raw("C", "teacher-b", "정답C", checksum, "run-b"),
        _raw("D", "teacher-b", "오답2", checksum, "run-b"),
        _raw("E", "teacher-b", "검토 후보 B", checksum, "run-b"),
        _raw("F", "teacher-b", "정답F", checksum, "run-b"),
    ])
    write_jsonl(left_path.parent / "errors.jsonl", [{
        "run_id": "run-a", "question_id": "F", "timestamp": "2026-08-10T00:00:00+00:00",
        "error_type": "malformed_response", "error": "choices missing",
    }])
    write_jsonl(right_path.parent / "errors.jsonl", [{
        "run_id": "run-b", "question_id": "G", "timestamp": "2026-08-10T00:00:00+00:00",
        "error_type": "http_error", "error": "response missing",
    }])

    result = verify_teacher_runs(question_path, left_path, right_path, tmp_path / "verified", checksum)

    assert result["independently_verified"] == 5
    assert result["rejected"] == 1
    assert result["unresolved"] == 1
    assert result["conflicts"] == 4
    assert result["student_training_eligible"] == 5
    audit = {row["question_id"]: row for row in read_jsonl(Path(result["verification_records_path"]))}
    assert audit["A"]["verification_result"] == "verified"
    assert audit["B"]["selected_teacher_side"] == "A"
    assert audit["C"]["selected_teacher_side"] == "B"
    assert audit["D"]["verification_result"] == "rejected"
    assert audit["E"]["verification_result"] == "unresolved"
    assert audit["F"]["teacher_a"]["error_type"] == "malformed_response"
    assert audit["G"]["teacher_b"]["error_type"] == "http_error"
    assert audit["B"]["teacher_a"]["raw_response_reference"]["path"] == str(left_path)
    assert audit["B"]["source_run_ids"] == ["run-a", "run-b"]
    serialized = json.dumps(read_jsonl(Path(result["eligible_teacher_only_path"])), ensure_ascii=False)
    assert "비공개 사고" not in serialized
    assert "reference_answer" not in serialized
    assert "rubric" not in serialized


def test_verification_readiness_recomputes_checksums_and_pilot_is_blocked_from_student_training(tmp_path: Path):
    questions = [_question("P-001", "정답")]
    question_path = tmp_path / "questions.jsonl"
    left_path, right_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(question_path, questions)
    checksum = hashlib.sha256(question_path.read_bytes()).hexdigest()
    write_jsonl(left_path, [_raw("P-001", "a", "정답", checksum, "run-a")])
    write_jsonl(right_path, [_raw("P-001", "b", "정답", checksum, "run-b")])
    result = verify_teacher_runs(question_path, left_path, right_path, tmp_path / "verified", checksum)

    readiness = verification_readiness(
        Path(result["verification_records_path"]), Path(result["eligible_teacher_only_path"])
    )
    assert readiness["status"] == "ready"
    assert readiness["checksum"] == hashlib.sha256(Path(result["eligible_teacher_only_path"]).read_bytes()).hexdigest()
    cli = subprocess.run(
        [sys.executable, "-m", "compass_ai", "verification-readiness",
         "--verification-file", result["verification_records_path"],
         "--eligible-file", result["eligible_teacher_only_path"]],
        text=True, capture_output=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["status"] == "ready"

    with pytest.raises(StudentDatasetError, match="Pilot-v1"):
        build_student_dataset(
            Path(result["eligible_teacher_only_path"]), tmp_path / "student.jsonl", "S4_teacher_only"
        )


def test_verification_output_collision_and_source_checksum_mismatch_fail_closed(tmp_path: Path):
    question_path = tmp_path / "questions.jsonl"
    left_path, right_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(question_path, [_question("A", "정답")])
    checksum = hashlib.sha256(question_path.read_bytes()).hexdigest()
    write_jsonl(left_path, [_raw("A", "a", "정답", checksum, "run-a")])
    write_jsonl(right_path, [_raw("A", "b", "정답", "b" * 64, "run-b")])
    with pytest.raises(ValueError, match="checksum"):
        verify_teacher_runs(question_path, left_path, right_path, tmp_path / "verified", checksum)
    write_jsonl(right_path, [_raw("A", "b", "정답", checksum, "run-b")])
    verify_teacher_runs(question_path, left_path, right_path, tmp_path / "verified", checksum)
    with pytest.raises(FileExistsError, match="collision"):
        verify_teacher_runs(question_path, left_path, right_path, tmp_path / "verified", checksum)


def test_manual_review_creates_separate_reviewed_artifacts_and_resolves_gate(tmp_path: Path):
    question_path = tmp_path / "questions.jsonl"
    write_jsonl(question_path, [_question("E", "사람 검토", "human_rubric")])
    checksum = hashlib.sha256(question_path.read_bytes()).hexdigest()
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    write_jsonl(left, [_raw("E", "a", "후보 A", checksum, "run-a")])
    write_jsonl(right, [_raw("E", "b", "후보 B", checksum, "run-b")])
    initial = verify_teacher_runs(question_path, left, right, tmp_path / "initial", checksum)
    review_path = tmp_path / "manual_review.jsonl"
    write_jsonl(review_path, [{
        "question_id": "E", "decision": "accept_a", "reviewer": "independent-human-1",
        "evidence": "rubric R-TEST를 독립 적용하여 후보 A 확인",
    }])

    reviewed = apply_verification_review(
        question_path, Path(initial["verification_records_path"]), Path(initial["eligible_teacher_only_path"]),
        review_path, tmp_path / "reviewed",
    )

    assert reviewed["status"] == "ready"
    assert reviewed["unresolved"] == 0
    assert reviewed["student_training_eligible"] == 1
    assert Path(initial["verification_records_path"]).read_text(encoding="utf-8") != Path(reviewed["verification_records_path"]).read_text(encoding="utf-8")
    record = read_jsonl(Path(reviewed["verification_records_path"]))[0]
    assert record["verifier_metadata"]["manual_review"]["reviewer"] == "independent-human-1"

@contextmanager
def _rehearsal_server():
    answers = {
        "model-a": {"A": "정답A", "B": "정답B", "C": "오답", "D": "오답1", "E": "검토 후보 A", "G": "정답G"},
        "model-b": {"A": "정답A", "B": "오답", "C": "정답C", "D": "오답2", "E": "검토 후보 B", "F": "정답F"},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, status: int, value: dict):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            model = request["model"]
            question_id = request["messages"][-1]["content"].split()[0]
            if model == "model-a" and question_id == "F":
                self.send_json(200, {"choices": []})
                return
            if model == "model-b" and question_id == "G":
                self.send_json(500, {"error": "intentional missing response"})
                return
            answer = answers[model][question_id]
            self.send_json(200, {
                "choices": [{"message": {"reasoning_content": "학습 금지 사고", "content": answer}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            })

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _teacher_config(path: Path, teacher: str, model: str, base_url: str, checksum: str) -> Path:
    value = {
        "teacher_name": teacher,
        "experiment_id": f"rehearsal-{teacher}",
        "dataset_name": "compass-ai-pilot",
        "dataset_version": "pilot-v1",
        "dataset_checksum": checksum,
        "base_url": base_url,
        "model": model,
        "timeout_seconds": 2,
        "system_prompt": "최종 답만 출력",
        "generation": {"temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 7},
        "extra_body": {},
        "model_metadata": {"checkpoint": "fake", "revision": "fixture", "quantization": None, "inference_engine": "fake", "server_config": {}},
    }
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_run_pilot_cli_output_feeds_independent_verifier(tmp_path: Path):
    questions = [
        _question("A", "정답A"), _question("B", "정답B"), _question("C", "정답C"),
        _question("D", "정답D"), _question("E", "사람 검토", "human_rubric"),
        _question("F", "정답F"), _question("G", "정답G"),
    ]
    pilot_dir = tmp_path / "data" / "questions" / "pilot"
    question_path = pilot_dir / "pilot_questions.jsonl"
    write_jsonl(question_path, questions)
    manifest = create_pilot_manifest(question_path, questions)
    write_pilot_manifest(pilot_dir / "pilot_manifest.json", manifest)
    with _rehearsal_server() as base_url:
        config_a = _teacher_config(tmp_path / "a.json", "teacher_a", "model-a", base_url, manifest["source_file_sha256"])
        config_b = _teacher_config(tmp_path / "b.json", "teacher_b", "model-b", base_url, manifest["source_file_sha256"])
        for teacher, config in [("teacher_a", config_a), ("teacher_b", config_b)]:
            completed = subprocess.run(
                [sys.executable, "-m", "compass_ai", "run-pilot", "--root", str(tmp_path), "--teacher", teacher,
                 "--config", str(config), "--output-root", str(tmp_path / "teacher-runs")],
                text=True, capture_output=True,
            )
            assert completed.returncode == 0, completed.stderr
    left = tmp_path / "teacher-runs" / "teacher_a" / "rehearsal-teacher_a" / "responses.jsonl"
    right = tmp_path / "teacher-runs" / "teacher_b" / "rehearsal-teacher_b" / "responses.jsonl"
    result = verify_teacher_runs(
        question_path, left, right, tmp_path / "actual-verification", manifest["source_file_sha256"]
    )
    assert result["teacher_a_completed"] == 6
    assert result["teacher_b_completed"] == 6
    assert result["independently_verified"] == 5
    assert result["rejected"] == 1
    assert result["unresolved"] == 1
    assert "reasoning_content" in read_jsonl(left)[0]["raw_response"]["choices"][0]["message"]
    assert "학습 금지 사고" not in json.dumps(read_jsonl(Path(result["eligible_teacher_only_path"])), ensure_ascii=False)


def test_full_rehearsal_runs_real_contract_and_tiny_training(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from compass_ai.rehearsal import run_full_rehearsal

    repo_root = Path(__file__).resolve().parents[1]
    result = run_full_rehearsal(repo_root, tmp_path / "full-rehearsal")

    assert result["status"] == "complete"
    assert result["verification"]["independently_verified"] == 5
    assert result["verification"]["rejected"] == 1
    assert result["verification"]["unresolved"] == 1
    assert result["pilot_training_gate"].startswith("blocked:Pilot-v1")
    assert result["tiny_student"]["adapter_reload_verified"] is True
