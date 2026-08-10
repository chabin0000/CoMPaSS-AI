"""ARGOS 학생 모델의 Transformers+PEFT LoRA 학습과 공통 평가."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .jsonl import read_jsonl
from .leakage import assert_no_evaluation_leakage
from .student_data import CONDITIONS, validate_student_dataset


class StudentConfigError(ValueError):
    """학생 학습 config 또는 실행 전 검사 오류."""


CONFIG_FIELDS = {
    "run_name", "condition", "base_model", "dataset", "evaluation_dataset", "output_dir",
    "training_enabled", "seed", "epochs", "learning_rate", "batch_size",
    "gradient_accumulation_steps", "max_sequence_length", "lora_rank", "lora_alpha",
    "lora_dropout", "target_modules", "fp16", "bf16", "max_steps",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def bitsandbytes_4bit_available(cuda_available: bool) -> tuple[bool, str]:
    if not cuda_available:
        return False, "CUDA is unavailable"
    if importlib.util.find_spec("bitsandbytes") is None:
        return False, "bitsandbytes is not installed"
    try:
        import bitsandbytes as bnb
        compiled = bool(getattr(getattr(bnb, "cextension", None), "lib", None)) and bool(
            getattr(bnb.cextension.lib, "compiled_with_cuda", False)
        )
        return (compiled, "bitsandbytes CUDA backend verified" if compiled else "bitsandbytes CUDA backend not verified")
    except Exception as error:
        return False, f"bitsandbytes import failed: {error}"


def _load_yaml_or_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise StudentConfigError("student config must be an object")
    return value


def load_student_config(path: Path) -> dict:
    config = _load_yaml_or_json(path)
    missing = CONFIG_FIELDS - config.keys()
    if missing:
        raise StudentConfigError(f"student config missing: {', '.join(sorted(missing))}")
    if "quantization" not in config and "quantization_mode" not in config:
        raise StudentConfigError("student config missing: quantization")
    if config["condition"] not in {"S0", *CONDITIONS}:
        raise StudentConfigError("unsupported student condition")
    if config["condition"] == "S0":
        if config["training_enabled"] or config["dataset"] is not None:
            raise StudentConfigError("S0 must disable training and use dataset: null")
    elif not config["training_enabled"] or not config["dataset"]:
        raise StudentConfigError("S1~S5 conditions require training and a dataset")
    numeric_positive = {
        "epochs", "learning_rate", "batch_size", "gradient_accumulation_steps",
        "max_sequence_length", "lora_rank", "lora_alpha", "max_steps",
    }
    for field in numeric_positive:
        if not isinstance(config[field], (int, float)) or config[field] <= 0:
            raise StudentConfigError(f"{field} must be positive")
    if not isinstance(config["target_modules"], list) or not config["target_modules"]:
        raise StudentConfigError("target_modules must be a non-empty list")
    if config["fp16"] and config["bf16"]:
        raise StudentConfigError("fp16 and bf16 cannot both be true")
    # 기존 tiny config의 quantization_mode를 호환하되 실제 실행 metadata는 quantization으로 통일한다.
    if "quantization" not in config:
        config["quantization"] = config.get("quantization_mode", "none")
    if config["quantization"] not in {"auto", "none", "4bit"}:
        raise StudentConfigError("quantization must be auto, none, or 4bit")
    config.setdefault("allow_quantization_fallback", config["quantization"] == "auto")
    config.setdefault("model_revision", None)
    config.setdefault("trust_remote_code", False)
    config.setdefault("prompt_format", "plain")
    config.setdefault("chat_template_required", False)
    config.setdefault("local_files_only", True)
    config.setdefault("dataset_checksum", None)
    config.setdefault("expected_sample_count", None)
    config.setdefault("minimum_disk_free_gb", 0)
    config.setdefault("save_steps", 50)
    config.setdefault("resume_from_checkpoint", "auto")
    if config["prompt_format"] not in {"plain", "chat_template"}:
        raise StudentConfigError("prompt_format must be plain or chat_template")
    return config


def resolve_quantization_mode(
    requested: str,
    allow_fallback: bool,
    cuda_available: bool,
    bitsandbytes_available: bool,
) -> tuple[str, str | None]:
    if requested == "none":
        return "none", None
    if requested == "auto":
        if cuda_available and bitsandbytes_available:
            return "4bit", "CUDA and bitsandbytes 4bit support verified; selected QLoRA"
        return "none", "4bit support was not verified; selected bf16 LoRA"
    if cuda_available and bitsandbytes_available:
        return "4bit", None
    if allow_fallback:
        return "none", "4bit unavailable; fallback to LoRA without quantization"
    raise StudentConfigError("4bit QLoRA requires CUDA and bitsandbytes")


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_model(root: Path, value: str) -> Path | str:
    """Hugging Face model ID는 경로로 변환하지 않고 로컬 경로만 root 기준으로 해석한다."""
    path = Path(value)
    if path.is_absolute() or value.startswith(".") or value.startswith("artifacts/") or value.startswith("artifacts\\"):
        return path if path.is_absolute() else root / path
    return value


def _model_load_kwargs(config: dict) -> dict:
    kwargs = {
        "local_files_only": bool(config.get("local_files_only", True)),
        "revision": config.get("model_revision"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    if config.get("bf16"):
        try:
            import torch
            kwargs["torch_dtype"] = torch.bfloat16
        except ImportError:
            pass
    return kwargs


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _validate_evaluation(path: Path) -> list[dict]:
    records = read_jsonl(path)
    ids: set[str] = set()
    for index, row in enumerate(records, 1):
        missing = {"evaluation_id", "prompt", "reference_answer"} - row.keys()
        if missing:
            raise StudentConfigError(f"evaluation {index} missing: {', '.join(sorted(missing))}")
        if row["evaluation_id"] in ids:
            raise StudentConfigError(f"duplicate evaluation_id: {row['evaluation_id']}")
        ids.add(row["evaluation_id"])
    if not records:
        raise StudentConfigError("evaluation dataset is empty")
    return records


def _format_prompt(tokenizer, prompt: str, *, add_generation_prompt: bool, prompt_format: str) -> str:
    if prompt_format == "chat_template":
        if not callable(getattr(tokenizer, "apply_chat_template", None)) or not getattr(tokenizer, "chat_template", None):
            raise StudentConfigError("configured tokenizer chat template is unavailable")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt if add_generation_prompt else f"### 질문\n{prompt}\n### 답변\n"


class _TokenizedStudentDataset:
    def __init__(self, records: list[dict], tokenizer, max_length: int, prompt_format: str = "plain"):
        self.items = []
        for row in records:
            prefix = _format_prompt(
                tokenizer,
                row["prompt"],
                add_generation_prompt=prompt_format == "chat_template",
                prompt_format=prompt_format,
            )
            full = prefix + row["target"] + tokenizer.eos_token
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
            labels = list(encoded["input_ids"])
            for index in range(min(len(prefix_ids), len(labels))):
                labels[index] = -100
            self.items.append({**encoded, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _collator(tokenizer):
    import torch

    def collate(features: list[dict]) -> dict:
        max_length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = max_length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collate


def evaluate_student_model(
    base_model_path: Path | str,
    evaluation_path: Path,
    adapter_path: Path | None = None,
    max_new_tokens: int = 4,
    model_options: dict | None = None,
) -> dict:
    """S0와 학습 조건이 함께 사용하는 최소 생성 evaluator."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    options = model_options or {}
    load_kwargs = {
        "local_files_only": bool(options.get("local_files_only", True)),
        "revision": options.get("model_revision"),
        "trust_remote_code": bool(options.get("trust_remote_code", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    records = _validate_evaluation(evaluation_path)
    results = []
    for row in records:
        prompt = _format_prompt(
            tokenizer,
            row["prompt"],
            add_generation_prompt=True,
            prompt_format=options.get("prompt_format", "plain"),
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append({
            "evaluation_id": row["evaluation_id"],
            "prediction": prediction,
            "reference_answer": row["reference_answer"],
            "exact_match": prediction == str(row["reference_answer"]).strip(),
        })
    return {
        "evaluation_count": len(results),
        "exact_match_count": sum(item["exact_match"] for item in results),
        "results": results,
        "notice": "This evaluator verifies execution only; it does not establish model performance.",
    }


def _run_student_impl(config_path: Path, root: Path = Path(".")) -> dict:
    config = load_student_config(config_path)
    base_model = _resolve_model(root, config["base_model"])
    dataset_path = _resolve(root, config["dataset"])
    evaluation_path = _resolve(root, config["evaluation_dataset"])
    output_dir = _resolve(root, config["output_dir"])
    assert evaluation_path is not None and output_dir is not None
    if isinstance(base_model, Path) and not base_model.exists():
        raise StudentConfigError(f"base model does not exist: {base_model}")
    if not evaluation_path.exists():
        raise StudentConfigError(f"evaluation dataset does not exist: {evaluation_path}")
    evaluation_records = _validate_evaluation(evaluation_path)
    dataset_checksum = None
    train_records: list[dict] = []
    if config["training_enabled"]:
        if dataset_path is None or not dataset_path.exists():
            raise StudentConfigError(f"training dataset does not exist: {dataset_path}")
        train_records = validate_student_dataset(read_jsonl(dataset_path))
        if any(row["condition"] != config["condition"] for row in train_records):
            raise StudentConfigError("dataset condition does not match config")
        assert_no_evaluation_leakage(
            [row["source_question_id"] for row in train_records],
            [row["evaluation_id"] for row in evaluation_records],
        )
        dataset_checksum = _sha256_file(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    snapshot = dict(config)
    (output_dir / "config.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        raise StudentConfigError("install the student extra: uv sync --extra student") from error
    set_seed(int(config["seed"]))
    bnb_available, bnb_reason = bitsandbytes_4bit_available(bool(torch.cuda.is_available()))
    actual_quantization, fallback_reason = resolve_quantization_mode(
        config["quantization"],
        bool(config["allow_quantization_fallback"]),
        bool(torch.cuda.is_available()),
        bnb_available,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, **_model_load_kwargs(config))
    load_kwargs = _model_load_kwargs(config)
    if actual_quantization == "4bit":
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if config["bf16"] else torch.float16
        load_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            "device_map": "auto",
        })
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    # Transformers 4.57은 config가 아니라 model.loss_type을 직접 읽으므로 causal LM loss를 명시한다.
    model.loss_type = "ForCausalLM"
    adapter_path: Path | None = None
    training_logs: list[dict] = []
    adapter_reload_verified = False
    lora_fan_in_fan_out = getattr(model.config, "model_type", None) == "gpt2"
    if config["training_enabled"]:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if actual_quantization == "4bit":
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(config["lora_rank"]),
                lora_alpha=int(config["lora_alpha"]),
                lora_dropout=float(config["lora_dropout"]),
                target_modules=list(config["target_modules"]),
                bias="none",
                fan_in_fan_out=lora_fan_in_fan_out,
            ),
        )
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=float(config["epochs"]),
            learning_rate=float(config["learning_rate"]),
            per_device_train_batch_size=int(config["batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            max_steps=int(config["max_steps"]),
            seed=int(config["seed"]),
            data_seed=int(config["seed"]),
            fp16=bool(config["fp16"]),
            bf16=bool(config["bf16"]),
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=max(1, int(config["save_steps"])),
            save_total_limit=2,
            report_to="none",
            disable_tqdm=True,
            use_cpu=not torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=_TokenizedStudentDataset(
                train_records,
                tokenizer,
                int(config["max_sequence_length"]),
                prompt_format=config["prompt_format"],
            ),
            data_collator=_collator(tokenizer),
        )
        resume_checkpoint = None
        if config.get("resume_from_checkpoint") == "auto":
            from transformers.trainer_utils import get_last_checkpoint
            checkpoints = output_dir / "checkpoints"
            resume_checkpoint = get_last_checkpoint(str(checkpoints)) if checkpoints.exists() else None
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        training_logs = list(trainer.state.log_history)
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
        adapter_path = output_dir / "adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        del model
        reloaded_base = AutoModelForCausalLM.from_pretrained(base_model, **_model_load_kwargs(config))
        from peft import PeftModel
        reloaded = PeftModel.from_pretrained(reloaded_base, adapter_path, local_files_only=True)
        adapter_reload_verified = reloaded is not None
        del reloaded
    evaluation = evaluate_student_model(base_model, evaluation_path, adapter_path, model_options=config)
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_logs.json").write_text(
        json.dumps(training_logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "status": "complete",
        "run_name": config["run_name"],
        "condition": config["condition"],
        "training_performed": bool(config["training_enabled"]),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_reload_verified": adapter_reload_verified,
        "dataset_checksum": dataset_checksum,
        "evaluation_dataset_checksum": _sha256_file(evaluation_path),
        "requested_quantization_mode": config["quantization"],
        "actual_quantization_mode": actual_quantization,
        "quantization_fallback_reason": fallback_reason,
        "bitsandbytes_runtime_reason": bnb_reason,
        "resolved_lora_fan_in_fan_out": lora_fan_in_fan_out,
        "resolved_loss_type": "ForCausalLM",
        "seed": config["seed"],
        "git_commit": _git_commit(root),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bitwise_reproducibility_claimed": False,
        },
        "config": snapshot,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {**metadata, "run_dir": str(output_dir)}


def run_student(config_path: Path, root: Path = Path(".")) -> dict:
    """알려진 미완료 run만 재개하고 모든 실패를 run metadata에 남긴다."""
    config = load_student_config(config_path)
    output_dir = _resolve(root, config["output_dir"])
    assert output_dir is not None
    checksum = _sha256_json(config)
    metadata_path = output_dir / "run_metadata.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not metadata_path.exists():
            raise StudentConfigError(f"output dir collision: {output_dir}")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        resumable = (
            config.get("resume_from_checkpoint") == "auto"
            and previous.get("status") in {"running", "failed"}
            and previous.get("config_checksum") == checksum
        )
        if not resumable:
            raise StudentConfigError(f"output dir collision: {output_dir}")
        started_at = previous.get("started_at") or datetime.now(UTC).isoformat()
        resume_attempts = int(previous.get("resume_attempts", 0)) + 1
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(UTC).isoformat()
        resume_attempts = 0
    running = {
        "status": "running",
        "run_name": config["run_name"],
        "condition": config["condition"],
        "config_checksum": checksum,
        "config": config,
        "started_at": started_at,
        "last_attempt_started_at": datetime.now(UTC).isoformat(),
        "resume_attempts": resume_attempts,
    }
    metadata_path.write_text(json.dumps(running, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        result = _run_student_impl(config_path, root)
    except Exception as error:
        running.update({
            "status": "failed",
            "finished_at": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        })
        metadata_path.write_text(json.dumps(running, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise
    # 구현부가 작성한 상세 metadata에도 재개 검증용 checksum을 보존한다.
    complete = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete.update({
        "config_checksum": checksum,
        "resume_attempts": resume_attempts,
        "started_at": started_at,
        "last_attempt_started_at": running["last_attempt_started_at"],
    })
    metadata_path.write_text(json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "config_checksum": checksum, "resume_attempts": resume_attempts}


def evaluate_saved_student(config_path: Path, root: Path = Path(".")) -> dict:
    config = load_student_config(config_path)
    base_model = _resolve_model(root, config["base_model"])
    evaluation_path = _resolve(root, config["evaluation_dataset"])
    output_dir = _resolve(root, config["output_dir"])
    assert evaluation_path is not None and output_dir is not None
    adapter_path = output_dir / "adapter" if config["training_enabled"] else None
    if adapter_path is not None and not adapter_path.exists():
        raise StudentConfigError(f"adapter does not exist: {adapter_path}")
    result = evaluate_student_model(base_model, evaluation_path, adapter_path, model_options=config)
    (output_dir / "evaluation.rerun.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
