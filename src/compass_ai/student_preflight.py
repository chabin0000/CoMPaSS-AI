"""ARGOS 학생 학습 설정의 정적 검사와 DGX 런타임 사전 검사."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from .jsonl import read_jsonl
from .student_data import validate_student_dataset
from .student_training import bitsandbytes_4bit_available, load_student_config, resolve_quantization_mode


ARGOS_V1_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"


def _check(name: str, status: str, detail: object = None, **extra: object) -> dict:
    return {"name": name, "status": status, "detail": detail, **extra}


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_target_modules(module_names: list[str], targets: list[str]) -> dict:
    """PEFT suffix 매칭과 같은 규칙으로 모델에 LoRA 대상이 있는지 확인한다."""
    missing = [target for target in targets if not any(name == target or name.endswith(f".{target}") for name in module_names)]
    return _check(
        "target_modules_runtime",
        "FAIL" if missing else "PASS",
        "missing target modules" if missing else "all configured target modules exist",
        missing=missing,
        targets=targets,
    )


def _hardware_checks() -> list[dict]:
    try:
        import torch
    except ImportError:
        return [
            _check("cuda", "UNKNOWN", "PyTorch is not installed"),
            _check("gpu", "UNKNOWN", None),
            _check("torch_cuda_version", "UNKNOWN", None),
            _check("bf16_support", "UNKNOWN", None),
            _check("bitsandbytes", "UNKNOWN", "PyTorch is not installed"),
        ]
    cuda = bool(torch.cuda.is_available())
    checks = [
        _check("cuda", "PASS" if cuda else "FAIL", cuda),
        _check("gpu", "PASS" if cuda else "UNKNOWN", torch.cuda.get_device_name(0) if cuda else None),
        _check("torch_cuda_version", "PASS" if torch.version.cuda else "UNKNOWN", torch.version.cuda),
        _check(
            "bf16_support",
            "PASS" if cuda and torch.cuda.is_bf16_supported() else ("FAIL" if cuda else "UNKNOWN"),
            bool(torch.cuda.is_bf16_supported()) if cuda else None,
        ),
    ]
    bnb_ok, bnb_reason = bitsandbytes_4bit_available(cuda)
    checks.append(_check(
        "bitsandbytes",
        "PASS" if bnb_ok else ("FAIL" if cuda else "UNKNOWN"),
        bnb_reason,
    ))
    return checks


def _runtime_model_checks(config: dict) -> list[dict]:
    """DGX에서 캐시에 준비된 모델만 로드한다. 이 함수 자체는 다운로드를 허용하지 않는다."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {
            "revision": config.get("model_revision"),
            "trust_remote_code": bool(config.get("trust_remote_code")),
            "local_files_only": True,
        }
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], **kwargs)
        model = AutoModelForCausalLM.from_pretrained(config["base_model"], **kwargs)
    except Exception as error:
        return [
            _check("target_modules_runtime", "FAIL", f"cached model load failed: {error}"),
            _check("tokenizer_chat_template_runtime", "FAIL", f"cached tokenizer/model load failed: {error}"),
        ]
    target_check = check_target_modules([name for name, _module in model.named_modules()], config["target_modules"])
    template_required = bool(config.get("chat_template_required")) or config.get("prompt_format") == "chat_template"
    has_template = bool(getattr(tokenizer, "chat_template", None)) and callable(getattr(tokenizer, "apply_chat_template", None))
    template_check = _check(
        "tokenizer_chat_template_runtime",
        "PASS" if has_template or not template_required else "FAIL",
        "chat template is usable" if has_template else ("not required by this config" if not template_required else "chat template is missing"),
    )
    return [target_check, template_check]


def _blocking_failures(checks: list[dict], config: dict) -> list[dict]:
    """auto/none에서 bitsandbytes 부재는 bf16 LoRA 진행을 막지 않는다."""
    return [
        item for item in checks
        if item["status"] == "FAIL"
        and not (item["name"] == "bitsandbytes" and config["quantization"] in {"auto", "none"})
    ]


def student_preflight(root: Path, config_path: Path, runtime_model_check: bool = False) -> dict:
    checks: list[dict] = []
    try:
        config = load_student_config(config_path)
        checks.append(_check("config_schema", "PASS", str(config_path)))
    except Exception as error:
        return {"status": "FAIL", "runtime_model_check_requested": runtime_model_check, "checks": [_check("config_schema", "FAIL", str(error))]}

    is_real = not str(config["base_model"]).endswith("tiny-base")
    model_ok = (config["base_model"] == ARGOS_V1_MODEL) if is_real else True
    checks.append(_check("base_model_id", "PASS" if model_ok else "FAIL", config["base_model"]))

    dataset_path = _resolve(root, config.get("dataset"))
    records: list[dict] = []
    if config["condition"] == "S0":
        checks.extend([
            _check("dataset", "PASS", "S0 has no training dataset"),
            _check("dataset_checksum", "PASS", None),
            _check("condition_consistency", "PASS", "S0 training disabled"),
            _check("pilot_leakage", "PASS", "S0 has no training samples"),
            _check("expected_sample_count", "PASS", 0),
        ])
    elif dataset_path is None or not dataset_path.exists():
        checks.extend([
            _check("dataset", "FAIL", str(dataset_path)),
            _check("dataset_checksum", "UNKNOWN", "dataset is not available"),
            _check("condition_consistency", "UNKNOWN", "dataset is not available"),
            _check("pilot_leakage", "UNKNOWN", "dataset is not available"),
            _check("expected_sample_count", "UNKNOWN", config.get("expected_sample_count")),
        ])
    else:
        try:
            records = validate_student_dataset(read_jsonl(dataset_path))
            actual_checksum = _sha256(dataset_path)
            expected = config.get("dataset_checksum")
            checks.append(_check("dataset", "PASS", str(dataset_path)))
            checks.append(_check(
                "dataset_checksum",
                "PASS" if expected is None or expected == actual_checksum else "FAIL",
                {"expected": expected, "actual": actual_checksum},
            ))
            conditions_ok = all(row["condition"] == config["condition"] for row in records)
            checks.append(_check("condition_consistency", "PASS" if conditions_ok else "FAIL", config["condition"]))
            pilot_path = root / "data" / "questions" / "pilot" / "pilot_questions.jsonl"
            pilot_ids = {row["question_id"] for row in read_jsonl(pilot_path)} if pilot_path.exists() else set()
            leaked = [
                row["source_question_id"] for row in records
                if row["source_question_id"] in pilot_ids
                or "pilot" in str(row.get("provenance", {}).get("source_file", "")).lower()
                or row.get("provenance", {}).get("dataset_version") == "pilot-v1"
            ]
            checks.append(_check("pilot_leakage", "FAIL" if leaked else "PASS", leaked or "no Pilot-v1 samples"))
            expected_count = config.get("expected_sample_count")
            count_ok = expected_count is None or expected_count == len(records)
            checks.append(_check("expected_sample_count", "PASS" if count_ok else "FAIL", {"expected": expected_count, "actual": len(records)}))
        except Exception as error:
            checks.append(_check("dataset", "FAIL", str(error)))

    lora_ok = all(float(config[key]) > 0 for key in ["lora_rank", "lora_alpha"]) and 0 <= float(config["lora_dropout"]) < 1
    checks.append(_check("lora_parameters", "PASS" if lora_ok else "FAIL", {key: config[key] for key in ["lora_rank", "lora_alpha", "lora_dropout", "target_modules"]}))
    checks.append(_check("quantization", "PASS", config["quantization"]))

    output = _resolve(root, config["output_dir"])
    collision = bool(output and output.exists() and any(output.iterdir()))
    resumable = False
    if collision and output is not None and (output / "run_metadata.json").exists():
        try:
            previous = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
            config_checksum = hashlib.sha256(
                json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            resumable = (
                config.get("resume_from_checkpoint") == "auto"
                and previous.get("status") in {"running", "failed"}
                and previous.get("config_checksum") == config_checksum
            )
        except Exception:
            resumable = False
    checks.append(_check(
        "output_path",
        "PASS" if not collision or resumable else "FAIL",
        {"path": str(output), "resumable": resumable},
        collision=collision,
    ))
    if output:
        usage = shutil.disk_usage(output.parent if output.parent.exists() else root)
        minimum = float(config.get("minimum_disk_free_gb", 0))
        free_gb = usage.free / 1024**3
        checks.append(_check("disk_free", "PASS" if free_gb >= minimum else "FAIL", {"free_gb": round(free_gb, 2), "minimum_gb": minimum}))

    hardware = _hardware_checks()
    checks.extend(hardware)
    cuda_available = next((item["detail"] for item in hardware if item["name"] == "cuda"), False) is True
    bnb_ok, _bnb_reason = bitsandbytes_4bit_available(cuda_available)
    try:
        selected, selection_reason = resolve_quantization_mode(
            config["quantization"],
            bool(config["allow_quantization_fallback"]),
            cuda_available,
            bnb_ok,
        )
        checks.append(_check("quantization_selection", "PASS", {"selected": selected, "reason": selection_reason}))
    except Exception as error:
        checks.append(_check("quantization_selection", "FAIL", str(error)))
    if runtime_model_check and not _blocking_failures(checks, config):
        checks.extend(_runtime_model_checks(config))
    elif runtime_model_check:
        checks.extend([
            _check("target_modules_runtime", "UNKNOWN", "runtime model load skipped because static/hardware checks failed"),
            _check("tokenizer_chat_template_runtime", "UNKNOWN", "runtime model load skipped because static/hardware checks failed"),
        ])
    else:
        checks.extend([
            _check("target_modules_runtime", "UNKNOWN", "use --runtime-model-check after the pinned model is cached on DGX"),
            _check("tokenizer_chat_template_runtime", "UNKNOWN", "use --runtime-model-check after the pinned tokenizer is cached on DGX"),
        ])
    return {
        "status": "FAIL" if _blocking_failures(checks, config) else ("UNKNOWN" if any(item["status"] == "UNKNOWN" for item in checks) else "PASS"),
        "config": str(config_path),
        "condition": config["condition"],
        "runtime_model_check_requested": runtime_model_check,
        "sample_count": len(records) if records else (0 if config["condition"] == "S0" else None),
        "checks": checks,
    }
