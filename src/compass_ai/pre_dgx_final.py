"""대형 모델을 열지 않고 ARGOS-v1의 마지막 정적 준비 상태를 판정한다."""

from __future__ import annotations

import json
from pathlib import Path

from .core_dataset import check_core_dataset
from .core_v2_dataset import check_core_v2_dataset
from .dgx import load_dgx_config
from .leakage import assert_private_final_is_safe
from .pre_dgx import pre_dgx_check
from .student_preflight import ARGOS_V1_MODEL
from .student_training import load_student_config
from .teacher_api import load_teacher_api_config


def _check(name: str, status: str, detail: object) -> dict:
    return {"name": name, "status": status, "detail": detail}


def pre_dgx_final(root: Path) -> dict:
    checks: list[dict] = []

    try:
        core_v1 = check_core_dataset(root)
        checks.append(_check("Core-v1 integrity", "PASS", core_v1["dataset_checksum"]))
    except Exception as error:
        checks.append(_check("Core-v1 integrity", "FAIL", str(error)))
    try:
        core_v2 = check_core_v2_dataset(root)
        checks.extend([
            _check("Core-v2 integrity", "PASS", core_v2["dataset_checksum"]),
            _check("Core-v2 evaluation protocol", "PASS", core_v2["evaluation_protocol_checksum"]),
            _check("Core-v2 REVIEW acceptance", "PASS", core_v2["review_acceptance"]),
        ])
    except Exception as error:
        core_v2 = {}
        checks.append(_check("Core-v2 integrity", "FAIL", str(error)))

    legacy = pre_dgx_check(root, run_tests=False)
    checks.append(_check("Pilot-v1 integrity", "PASS" if legacy["status"] == "ready" else "FAIL", legacy["status"]))
    try:
        assert_private_final_is_safe(root / "evaluation" / "private_final")
        checks.append(_check("training/evaluation leakage guard", "PASS", "private_final contains no evaluation items"))
    except Exception as error:
        checks.append(_check("training/evaluation leakage guard", "FAIL", str(error)))

    teacher_paths = sorted((root / "configs" / "teachers").glob("*_dgx.yaml"))
    try:
        teacher_configs = [load_teacher_api_config(path) for path in teacher_paths]
        models = {config["teacher_name"]: config["model"] for config in teacher_configs}
        checks.append(_check("teacher configs", "PASS" if len(teacher_configs) >= 2 else "FAIL", models))
    except Exception as error:
        checks.append(_check("teacher configs", "FAIL", str(error)))

    expected_conditions = {
        "S0", "S1", "S2", "S3a", "S3b", "S3_matched", "S4",
        "S4_teacher_only", "S4_with_human_correction", "S5",
    }
    try:
        student_paths = sorted((root / "configs" / "student").glob("s*.yaml"))
        real_configs = [load_student_config(path) for path in student_paths if path.stem != "dry_run_tiny"]
        by_condition = {config["condition"]: config for config in real_configs}
        missing = sorted(expected_conditions - by_condition.keys())
        wrong_models = sorted(condition for condition, config in by_condition.items() if config["base_model"] != ARGOS_V1_MODEL)
        status = "PASS" if not missing and not wrong_models else "FAIL"
        checks.append(_check("ARGOS S0-S5 configs", status, {"missing": missing, "wrong_models": wrong_models, "model": ARGOS_V1_MODEL}))
    except Exception as error:
        checks.append(_check("ARGOS S0-S5 configs", "FAIL", str(error)))

    try:
        dgx = load_dgx_config(root / "configs" / "dgx" / "argos_v1.yaml")
        checks.append(_check("DGX config", "PASS", {"student_configs": len(dgx["student_configs"]), "teacher_configs": len(dgx["teacher_configs"])}))
        s5_deferred = (
            "configs/student/s5.yaml" not in dgx["student_configs"]
            and "configs/student/s5.yaml" in dgx.get("deferred_student_configs", [])
        )
        checks.append(_check("S5 first-study deferral", "PASS" if s5_deferred else "FAIL", "config/schema retained for follow-up experiment"))
    except Exception as error:
        checks.append(_check("DGX config", "FAIL", str(error)))

    runbook = root / "DGX_RUNBOOK.md"
    checks.append(_check("DGX runbook", "PASS" if runbook.is_file() else "FAIL", str(runbook)))
    rehearsal_path = root / "reports" / "core_v2_synthetic_rehearsal_status.json"
    if rehearsal_path.is_file():
        try:
            rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
            valid = (
                rehearsal.get("status") == "complete"
                and rehearsal.get("matched_comparison") == "PASS"
                and rehearsal.get("private_eval_leakage") == "PASS"
                and rehearsal.get("tiny_training", {}).get("adapter_reload_verified") is True
                and rehearsal.get("dataset_checksum") == core_v2.get("dataset_checksum")
                and rehearsal.get("protocol_checksum") == core_v2.get("evaluation_protocol_checksum")
            )
            checks.append(_check("synthetic S0-S5 rehearsal", "PASS" if valid else "FAIL", rehearsal_path.as_posix()))
        except Exception as error:
            checks.append(_check("synthetic S0-S5 rehearsal", "FAIL", str(error)))
    else:
        checks.append(_check("synthetic S0-S5 rehearsal", "UNKNOWN", "run core-v2-rehearsal once"))

    for name, detail in [
        ("DGX CUDA/GPU/bf16", "runtime hardware check required"),
        ("bitsandbytes 4bit CUDA support", "runtime import and kernel check required"),
        ("EXAONE tokenizer/chat template/LoRA modules", "local model snapshot runtime check required"),
        ("teacher endpoints", "OpenAI-compatible servers must be started on DGX"),
    ]:
        checks.append(_check(name, "UNKNOWN", detail))

    failures = [check for check in checks if check["status"] == "FAIL"]
    unknown = [check["name"] for check in checks if check["status"] == "UNKNOWN"]
    return {
        "status": "FAIL" if failures else "PASS",
        "core_v2_checksum": core_v2.get("dataset_checksum"),
        "evaluation_protocol_checksum": core_v2.get("evaluation_protocol_checksum"),
        "checks": checks,
        "unresolved_static_blockers": [check["name"] for check in failures],
        "runtime_unknown": unknown,
    }
