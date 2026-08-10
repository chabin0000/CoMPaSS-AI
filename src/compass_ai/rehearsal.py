"""대형 모델 없이 실제 teacher JSONL 계약과 ARGOS tiny 경계를 리허설한다."""

from __future__ import annotations

import json
import shutil
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dgx import build_dgx_plan, collect_dgx_system_report
from .jsonl import write_jsonl
from .pilot import create_pilot_manifest, write_pilot_manifest
from .student_data import StudentDatasetError, build_student_dataset
from .student_tiny import run_tiny_student_dry_run
from .teacher_pilot import run_teacher_pilot
from .teacher_run import comparison_readiness
from .teacher_verification import verify_teacher_runs


def _question(question_id: str, answer: str, method: str = "answer_key") -> dict:
    return {
        "question_id": question_id, "split": "pilot", "domain": "stem", "task_type": "calculation",
        "question": f"{question_id} 질문", "reference_answer": answer, "verification_method": method,
        "evidence_source": "full rehearsal fixture", "difficulty": "easy", "requires_current_info": False,
        "notes": "rubric=R-REHEARSAL; origin=original",
    }


@contextmanager
def _fake_server():
    answers = {
        "model-a": {"A": "정답A", "B": "정답B", "C": "오답", "D": "오답1", "E": "검토 후보 A", "G": "정답G"},
        "model-b": {"A": "정답A", "B": "오답", "C": "정답C", "D": "오답2", "E": "검토 후보 B", "F": "정답F"},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, status: int, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self.send_json(200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})

        def do_POST(self) -> None:
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
            self.send_json(200, {
                "choices": [{"message": {"reasoning_content": "training target에서 제외", "content": answers[model][question_id]}}],
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


def _write_teacher_config(path: Path, teacher: str, model: str, base_url: str, checksum: str) -> Path:
    value = {
        "teacher_name": teacher, "experiment_id": f"rehearsal-{teacher}",
        "dataset_name": "compass-ai-pilot", "dataset_version": "pilot-v1", "dataset_checksum": checksum,
        "base_url": base_url, "model": model, "timeout_seconds": 2, "system_prompt": "최종 답만 출력",
        "generation": {"temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 7}, "extra_body": {},
        "model_metadata": {"checkpoint": "fake", "revision": "fixture", "quantization": None, "inference_engine": "fake", "server_config": {}},
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_full_rehearsal(repo_root: Path, output_root: Path) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"full rehearsal output collision: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    system_report = collect_dgx_system_report(repo_root, output_root / "system-report")
    plan = build_dgx_plan(repo_root, repo_root / "configs" / "dgx" / "argos_v1.yaml")

    questions = [
        _question("A", "정답A"), _question("B", "정답B"), _question("C", "정답C"),
        _question("D", "정답D"), _question("E", "사람 검토", "human_rubric"),
        _question("F", "정답F"), _question("G", "정답G"),
    ]
    pilot_dir = output_root / "fixture-root" / "data" / "questions" / "pilot"
    questions_path = pilot_dir / "pilot_questions.jsonl"
    write_jsonl(questions_path, questions)
    manifest = create_pilot_manifest(questions_path, questions)
    write_pilot_manifest(pilot_dir / "pilot_manifest.json", manifest)
    with _fake_server() as base_url:
        config_a = _write_teacher_config(output_root / "teacher-a.json", "teacher_a", "model-a", base_url, manifest["source_file_sha256"])
        config_b = _write_teacher_config(output_root / "teacher-b.json", "teacher_b", "model-b", base_url, manifest["source_file_sha256"])
        teacher_root = output_root / "teacher-runs"
        run_a = run_teacher_pilot(output_root / "fixture-root", "teacher_a", config_a, teacher_root)
        run_b = run_teacher_pilot(output_root / "fixture-root", "teacher_b", config_b, teacher_root)
    left, right = Path(run_a["responses_path"]), Path(run_b["responses_path"])
    comparison = comparison_readiness(left, right, manifest["source_file_sha256"])
    verification = verify_teacher_runs(
        questions_path, left, right, output_root / "verification", manifest["source_file_sha256"]
    )

    pilot_training_gate = "unexpectedly_allowed"
    try:
        build_student_dataset(
            Path(verification["eligible_teacher_only_path"]), output_root / "forbidden-student.jsonl", "S4_teacher_only"
        )
    except StudentDatasetError as error:
        pilot_training_gate = f"blocked:{error}"
    if not pilot_training_gate.startswith("blocked:Pilot-v1"):
        raise AssertionError("Pilot-v1 leakage gate did not block student training")

    # 학생 backend 자체는 별도의 non-Pilot fixture로 실제 LoRA 저장/reload/evaluation까지 검사한다.
    tiny_root = output_root / "tiny-student"
    (tiny_root / "tests" / "fixtures").mkdir(parents=True)
    (tiny_root / "evaluation" / "student").mkdir(parents=True)
    shutil.copy2(repo_root / "tests" / "fixtures" / "student_teacher_verified.jsonl", tiny_root / "tests" / "fixtures" / "student_teacher_verified.jsonl")
    shutil.copy2(repo_root / "evaluation" / "student" / "toy_eval.jsonl", tiny_root / "evaluation" / "student" / "toy_eval.jsonl")
    tiny = run_tiny_student_dry_run(repo_root / "configs" / "student" / "dry_run_tiny.yaml", tiny_root)
    result = {
        "status": "complete",
        "system_report": {"json_path": system_report["json_path"], "text_path": system_report["text_path"]},
        "plan_step_count": len(plan["steps"]),
        "teacher_a": {"status": run_a["status"], "completed": run_a["completed_count"]},
        "teacher_b": {"status": run_b["status"], "completed": run_b["completed_count"]},
        "comparison": comparison,
        "verification": verification,
        "pilot_training_gate": pilot_training_gate,
        "tiny_student": {"status": tiny["status"], "adapter_reload_verified": tiny["adapter_reload_verified"], "run_dir": tiny["run_dir"]},
        "notice": "Pilot-v1 verification and non-Pilot tiny training are deliberately separated; no performance claim is made.",
    }
    (output_root / "full_rehearsal.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

