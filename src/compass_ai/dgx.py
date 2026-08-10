"""DGX 환경 기록, ARGOS 실행 계획, 재개 가능한 본실험 오케스트레이션."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from .pre_dgx import pre_dgx_check
from .student_preflight import ARGOS_V1_MODEL, student_preflight
from .student_training import _load_yaml_or_json, run_student
from .teacher_api import teacher_check
from .teacher_pilot import run_teacher_pilot
from .teacher_run import comparison_readiness
from .teacher_verification import apply_verification_review, verify_teacher_runs, verification_readiness


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _system_memory_bytes() -> int | None:
    proc = Path("/proc/meminfo")
    if proc.exists():
        for line in proc.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    if platform.system() == "Windows":
        try:
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ["total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended"]
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
        except Exception:
            return None
    return None


def collect_dgx_system_report(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dgx_system_report.json"
    text_path = output_dir / "dgx_system_report.txt"
    if json_path.exists() or text_path.exists():
        raise FileExistsError(f"system report output collision: {output_dir}")
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        gpu = []
        for index in range(torch.cuda.device_count() if cuda_available else 0):
            properties = torch.cuda.get_device_properties(index)
            gpu.append({"index": index, "name": properties.name, "memory_bytes": properties.total_memory})
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        bf16 = bool(torch.cuda.is_bf16_supported()) if cuda_available else None
    except ImportError:
        cuda_available, gpu, torch_version, torch_cuda, bf16 = None, [], None, None, None
    usage = shutil.disk_usage(root)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "os": {"platform": platform.platform(), "system": platform.system(), "release": platform.release()},
        "python": platform.python_version(),
        "pytorch": torch_version,
        "cuda": {"available": cuda_available, "torch_cuda_version": torch_cuda, "bf16_supported": bf16},
        "gpu": gpu,
        "system_memory_bytes": _system_memory_bytes(),
        "packages": {name: _version(name) for name in ["transformers", "peft", "accelerate", "bitsandbytes", "vllm", "sglang"]},
        "inference_engines": {name: importlib.util.find_spec(name) is not None for name in ["vllm", "sglang"]},
        "git_commit": _git_commit(root),
        "disk": {"path": str(root.resolve()), "free_bytes": usage.free, "total_bytes": usage.total},
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"Created: {report['created_at']}", f"OS: {report['os']['platform']}", f"Python: {report['python']}",
        f"PyTorch: {torch_version}", f"CUDA: {torch_cuda} (available={cuda_available})", f"BF16: {bf16}",
        f"GPU: {json.dumps(gpu, ensure_ascii=False)}", f"Packages: {json.dumps(report['packages'], ensure_ascii=False)}",
        f"Git: {report['git_commit']}", f"Disk free bytes: {usage.free}",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**report, "status": "recorded", "json_path": str(json_path), "text_path": str(text_path)}


def load_dgx_config(path: Path) -> dict:
    config = _load_yaml_or_json(path)
    required = {"teacher_configs", "teacher_runs", "student_configs", "verified_student_datasets", "dataset_checksum", "minimum_disk_free_gb"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"DGX config missing: {', '.join(sorted(missing))}")
    return config


def build_dgx_plan(root: Path, config_path: Path) -> dict:
    config = load_dgx_config(config_path)
    students = [str(root / path) for path in config["student_configs"]]
    steps = [
        {"number": 1, "description": "DGX system report 기록", "inputs": [], "outputs": ["artifacts/dgx_runs/<run_id>/system"]},
        {"number": 2, "description": "Pilot-v1/Core-v2/protocol/leakage 최종 무결성 검사", "inputs": ["data/questions/pilot", "data/questions/core_v2"], "outputs": []},
        {"number": 3, "description": "운영자가 선택한 엔진으로 GPT-OSS localhost server 준비", "inputs": [], "outputs": []},
        {"number": 4, "description": "GPT-OSS endpoint check와 Pilot-v1/Core-v2 train 실행 또는 resume", "inputs": [config["teacher_configs"]["gpt_oss"], config["teacher_configs"]["gpt_oss_core_v2"]], "outputs": ["artifacts/dgx_runs/<run_id>/teachers"]},
        {"number": 5, "description": "GPT 서버 종료 후 운영자가 GLM localhost server 준비", "inputs": [], "outputs": []},
        {"number": 6, "description": "GLM endpoint check와 Pilot-v1/Core-v2 train 실행 또는 resume", "inputs": [config["teacher_configs"]["glm45_air"], config["teacher_configs"]["glm45_air_core_v2"]], "outputs": ["artifacts/dgx_runs/<run_id>/teachers"]},
        {"number": 7, "description": "Core-v2 teacher A/B comparison readiness", "inputs": ["Core-v2 train teacher responses"], "outputs": []},
        {"number": 8, "description": "독립 검증 및 unresolved 수동 검토 gate", "inputs": ["Core-v2 train teacher responses"], "outputs": ["artifacts/dgx_runs/<run_id>/verification"]},
        {"number": 9, "description": "1차 본실험 S1~S4 condition dataset build와 S3_matched/S4_teacher_only matching(S5 후속 보류)", "inputs": ["verified teacher/human artifacts"], "outputs": ["artifacts/dgx_runs/<run_id>/student_datasets"]},
        {"number": 10, "description": "ARGOS student runtime preflight", "inputs": students, "outputs": []},
        {"number": 11, "description": "S0 baseline와 1차 본실험 S1~S4 training", "inputs": students, "outputs": ["artifacts/dgx_runs/<run_id>/training"]},
        {"number": 12, "description": "동결 Core-v2 private_eval 동일 decoding 평가", "inputs": ["private key transferred separately"], "outputs": ["artifacts/dgx_runs/<run_id>/evaluation"]},
        {"number": 13, "description": "rubric unresolved blind manual scoring gate", "inputs": ["evaluation predictions"], "outputs": ["manual scoring artifact"]},
        {"number": 14, "description": "accuracy/domain/McNemar 결과 분석", "inputs": ["scored item-level results"], "outputs": ["artifacts/dgx_runs/<run_id>/summary"]},
        {"number": 15, "description": "run artifact backup", "inputs": ["artifacts/dgx_runs/<run_id>"], "outputs": ["external backup"]},
    ]
    return {
        "dry_run": True,
        "config": str(config_path),
        "student_model": ARGOS_V1_MODEL,
        "dataset_checksum": config["dataset_checksum"],
        "environment_setup": {
            "base": "uv sync --python 3.11 --extra student",
            "qlora_probe": "uv sync --python 3.11 --extra student --extra student-qlora",
            "repair_after_qlora_failure": "uv sync --python 3.11 --extra student",
            "note": "use a dedicated UV_PROJECT_ENVIRONMENT; QLoRA install failure must not trigger training",
        },
        "steps": steps,
    }


def format_dgx_plan(plan: dict) -> str:
    lines = [f"ARGOS-v1 model: {plan['student_model']}", f"DGX config: {plan['config']}"]
    environment = plan.get("environment_setup", {})
    if environment:
        lines.extend([
            f"ENV BASE: {environment['base']}",
            f"ENV OPTIONAL QLORA: {environment['qlora_probe']}",
            f"ENV REPAIR IF QLORA FAILS: {environment['repair_after_qlora_failure']}",
        ])
    for step in plan["steps"]:
        lines.append(f"STEP {step['number']} - {step['description']}")
        if step["inputs"]:
            lines.append(f"  inputs: {', '.join(step['inputs'])}")
        if step["outputs"]:
            lines.append(f"  outputs: {', '.join(step['outputs'])}")
    lines.append("DRY PLAN ONLY: no API call, model download, or training was performed.")
    return "\n".join(lines)


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run_dgx_launcher(root: Path, config_path: Path, run_id: str, state_root: Path | None = None) -> dict:
    config = load_dgx_config(config_path)
    run_root = (state_root or root / "artifacts" / "dgx_runs") / run_id
    artifact_dirs = {
        name: run_root / name for name in
        ["system", "teachers", "verification", "student_datasets", "training", "evaluation", "summary"]
    }
    config_checksum = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for name, path in artifact_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        metadata = path / "artifact_metadata.json"
        if not metadata.exists():
            metadata.write_text(json.dumps({
                "run_id": run_id, "artifact_type": name,
                "created_at": datetime.now(UTC).isoformat(),
                "config_path": str(config_path), "config_checksum": config_checksum,
                "dataset_checksum": config["dataset_checksum"],
            }, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    state_path = run_root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != str(config_path):
            raise ValueError("launcher resume config mismatch; choose another run_id")
    else:
        state = {"run_id": run_id, "config": str(config_path), "started_at": datetime.now(UTC).isoformat(), "completed_stages": [], "status": "running", "history": []}
        _write_state(state_path, state)

    def complete(stage: str, detail: object) -> None:
        if stage not in state["completed_stages"]:
            state["completed_stages"].append(stage)
        state["history"].append({"stage": stage, "at": datetime.now(UTC).isoformat(), "detail": detail})
        state["status"] = "running"
        state.pop("blocked_stage", None)
        state.pop("reason", None)
        _write_state(state_path, state)

    def blocked(stage: str, error: object) -> dict:
        state.update({"status": "blocked", "blocked_stage": stage, "reason": str(error), "updated_at": datetime.now(UTC).isoformat()})
        _write_state(state_path, state)
        return {**state, "state_path": str(state_path)}

    try:
        if "system_report" not in state["completed_stages"]:
            report_dir = artifact_dirs["system"] / "report"
            report_json = report_dir / "dgx_system_report.json"
            report_text = report_dir / "dgx_system_report.txt"
            if report_json.exists() and report_text.exists():
                existing_report = json.loads(report_json.read_text(encoding="utf-8"))
                existing_report.update({"status": "recorded", "json_path": str(report_json), "text_path": str(report_text)})
                complete("system_report", existing_report)
            else:
                try:
                    report = collect_dgx_system_report(root, report_dir)
                except Exception as error:
                    return blocked("system_report", error)
                complete("system_report", report)
        if "integrity" not in state["completed_stages"]:
            integrity = pre_dgx_check(root, run_tests=False)
            if integrity["status"] != "ready":
                return blocked("integrity", integrity)
            usage = shutil.disk_usage(root)
            if usage.free / 1024**3 < float(config["minimum_disk_free_gb"]):
                return blocked("disk_space", f"free disk is below {config['minimum_disk_free_gb']} GiB")
            complete("integrity", integrity)

        for teacher, stage in [("gpt_oss", "gpt_oss"), ("glm45_air", "glm45_air")]:
            cfg = root / config["teacher_configs"][teacher]
            check_stage = f"{stage}_teacher_check"
            if check_stage not in state["completed_stages"]:
                try:
                    report = teacher_check(root, cfg)
                except Exception as error:
                    return blocked(check_stage, error)
                if report.get("status") != "ready":
                    return blocked(check_stage, report)
                complete(check_stage, report)
            pilot_stage = f"{stage}_pilot"
            if pilot_stage not in state["completed_stages"]:
                try:
                    result = run_teacher_pilot(root, teacher, cfg)
                except Exception as error:
                    return blocked(pilot_stage, error)
                if result.get("status") != "complete":
                    return blocked(pilot_stage, "Pilot incomplete; rerun the same launcher after fixing the endpoint")
                complete(pilot_stage, result)

        if "teacher_comparison" not in state["completed_stages"]:
            try:
                comparison = comparison_readiness(
                    root / config["teacher_runs"]["gpt_oss"],
                    root / config["teacher_runs"]["glm45_air"],
                    config["dataset_checksum"],
                )
            except Exception as error:
                return blocked("teacher_comparison", error)
            if comparison.get("status") != "ready":
                return blocked("teacher_comparison", comparison)
            complete("teacher_comparison", comparison)

        if "independent_verification" not in state["completed_stages"]:
            verification_dir = run_root / "verification" / "initial"
            records_path = verification_dir / "verification_records.jsonl"
            eligible_path = verification_dir / "eligible_teacher_only.jsonl"
            if records_path.exists() and eligible_path.exists():
                verification = verification_readiness(records_path, eligible_path)
                verification.update({
                    "verification_records_path": str(records_path),
                    "eligible_teacher_only_path": str(eligible_path),
                })
            else:
                verification = verify_teacher_runs(
                    root / "data" / "questions" / "pilot" / "pilot_questions.jsonl",
                    root / config["teacher_runs"]["gpt_oss"],
                    root / config["teacher_runs"]["glm45_air"],
                    verification_dir,
                    config["dataset_checksum"],
                )
            if verification.get("status") == "needs_review":
                review_path = run_root / "manual_review.jsonl"
                reviewed_dir = run_root / "verification" / "reviewed"
                reviewed_records = reviewed_dir / "verification_records.jsonl"
                reviewed_eligible = reviewed_dir / "eligible_with_human_correction.jsonl"
                if not review_path.exists():
                    return blocked(
                        "independent_review",
                        {
                            **verification,
                            "required_action": "create manual_review.jsonl for unresolved IDs; do not infer correctness from teacher agreement",
                            "review_file": str(review_path),
                        },
                    )
                if reviewed_records.exists() and reviewed_eligible.exists():
                    verification = verification_readiness(reviewed_records, reviewed_eligible)
                    verification.update({
                        "verification_records_path": str(reviewed_records),
                        "eligible_with_human_correction_path": str(reviewed_eligible),
                    })
                else:
                    verification = apply_verification_review(
                        root / "data" / "questions" / "pilot" / "pilot_questions.jsonl",
                        records_path, eligible_path, review_path, reviewed_dir,
                    )
                if verification.get("status") == "needs_review":
                    return blocked("independent_review", verification)
            if verification.get("status") != "ready":
                return blocked("independent_verification", verification)
            complete("independent_verification", verification)

        missing_verified = [path for path in config["verified_student_datasets"] if not (root / path).exists()]
        if missing_verified:
            return blocked("verified_student_dataset_gate", f"missing verified student datasets: {', '.join(missing_verified)}")
        if "verified_student_dataset_gate" not in state["completed_stages"]:
            complete("verified_student_dataset_gate", {"datasets": config["verified_student_datasets"]})

        for relative in config["student_configs"]:
            student_config = root / relative
            stage = f"student_{Path(relative).stem}"
            if stage in state["completed_stages"]:
                continue
            preflight = student_preflight(root, student_config, runtime_model_check=True)
            if preflight["status"] != "PASS":
                return blocked(f"{stage}_preflight", preflight)
            try:
                result = run_student(student_config, root)
            except Exception as error:
                return blocked(stage, error)
            complete(stage, result)
        state.update({"status": "complete", "finished_at": datetime.now(UTC).isoformat()})
        _write_state(state_path, state)
        return {**state, "state_path": str(state_path)}
    except Exception as error:
        return blocked(state.get("blocked_stage", "unexpected_failure"), error)
