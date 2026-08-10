from __future__ import annotations

import json
from pathlib import Path

import pytest

from compass_ai.dgx import build_dgx_plan, collect_dgx_system_report, run_dgx_launcher
from compass_ai.student_preflight import check_target_modules, student_preflight
from compass_ai.student_training import (
    StudentConfigError,
    _TokenizedStudentDataset,
    load_student_config,
    resolve_quantization_mode,
    run_student,
)


ROOT = Path(__file__).resolve().parents[1]
EXAONE = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION = "553ea250b9a5317231459279d5847d6cf955b9aa"
REAL_CONFIGS = [
    "s0.yaml", "s1.yaml", "s2.yaml", "s3a.yaml", "s3b.yaml", "s3_matched.yaml",
    "s4.yaml", "s4_teacher_only.yaml", "s4_with_human_correction.yaml", "s5.yaml",
]


def test_all_real_student_configs_pin_exaone_without_changing_tiny_configs():
    for filename in REAL_CONFIGS:
        config = load_student_config(ROOT / "configs" / "student" / filename)
        assert config["base_model"] == EXAONE
        assert config["model_revision"] == REVISION
        assert config["trust_remote_code"] is True
        assert config["prompt_format"] == "chat_template"
        assert config["chat_template_required"] is True
        assert config["bf16"] is True
        assert config["quantization"] == "auto"
        assert config["target_modules"] == ["q_proj", "k_proj", "v_proj", "out_proj"]

    tiny = load_student_config(ROOT / "configs" / "student" / "dry_run_tiny.yaml")
    assert tiny["base_model"].endswith("tiny-base")
    assert tiny["target_modules"] == ["c_attn"]
    assert tiny["quantization"] == "none"


def test_quantization_auto_selects_qlora_only_when_runtime_support_is_verified():
    actual, reason = resolve_quantization_mode("auto", True, True, True)
    assert actual == "4bit"
    assert "verified" in reason

    actual, reason = resolve_quantization_mode("auto", True, True, False)
    assert actual == "none"
    assert "bf16 LoRA" in reason

    with pytest.raises(Exception, match="4bit"):
        resolve_quantization_mode("4bit", False, False, False)


def test_target_module_runtime_validation_is_explicit():
    names = [
        "transformer.h.0.attn.attention.q_proj",
        "transformer.h.0.attn.attention.k_proj",
        "transformer.h.0.attn.attention.v_proj",
        "transformer.h.0.attn.attention.out_proj",
    ]
    passed = check_target_modules(names, ["q_proj", "k_proj", "v_proj", "out_proj"])
    assert passed["status"] == "PASS"
    failed = check_target_modules(names, ["q_proj", "missing_proj"])
    assert failed["status"] == "FAIL"
    assert failed["missing"] == ["missing_proj"]


def test_chat_template_keeps_assistant_target_separate():
    class FakeTokenizer:
        eos_token = "<eos>"
        chat_template = "configured"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            return f"<user>{messages[0]['content']}<assistant>" if add_generation_prompt else "bad"

        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(len(text))), "attention_mask": [1] * len(text)}

    dataset = _TokenizedStudentDataset(
        [{"prompt": "질문", "target": "최종답"}], FakeTokenizer(), 100, prompt_format="chat_template"
    )
    item = dataset[0]
    assert item["labels"][: len("<user>질문<assistant>")] == [-100] * len("<user>질문<assistant>")


def test_failed_student_run_records_reason_and_allows_only_matching_resume(tmp_path: Path):
    config = {
        "run_name": "resume-failure", "condition": "S0", "base_model": str(tmp_path / "missing-model"),
        "dataset": None, "evaluation_dataset": str(tmp_path / "eval.jsonl"), "output_dir": str(tmp_path / "run"),
        "training_enabled": False, "seed": 1, "epochs": 1, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "max_sequence_length": 32, "lora_rank": 2, "lora_alpha": 4,
        "lora_dropout": 0.0, "target_modules": ["c_attn"], "fp16": False, "bf16": False,
        "quantization": "none", "max_steps": 1, "resume_from_checkpoint": "auto",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(StudentConfigError, match="base model does not exist"):
        run_student(config_path, tmp_path)
    metadata_path = tmp_path / "run" / "run_metadata.json"
    failed = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error_type"] == "StudentConfigError"
    with pytest.raises(StudentConfigError, match="base model does not exist"):
        run_student(config_path, tmp_path)
    retried = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert retried["resume_attempts"] == 1


def test_static_student_preflight_does_not_load_model(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("static preflight must not load a model")

    monkeypatch.setattr("compass_ai.student_preflight._runtime_model_checks", forbidden)
    report = student_preflight(ROOT, ROOT / "configs" / "student" / "s3_matched.yaml")
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["config_schema"]["status"] == "PASS"
    assert checks["base_model_id"]["status"] == "PASS"
    assert checks["dataset"]["status"] in {"PASS", "FAIL"}
    assert checks["target_modules_runtime"]["status"] == "UNKNOWN"
    assert checks["tokenizer_chat_template_runtime"]["status"] == "UNKNOWN"
    assert report["runtime_model_check_requested"] is False


def test_system_report_writes_text_and_json(tmp_path: Path):
    report = collect_dgx_system_report(ROOT, tmp_path)
    assert Path(report["json_path"]).exists()
    assert Path(report["text_path"]).exists()
    saved = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    for key in ["os", "python", "pytorch", "cuda", "gpu", "disk", "git_commit"]:
        assert key in saved


def test_dgx_plan_is_dry_and_lists_models_and_paths(monkeypatch):
    monkeypatch.setattr("compass_ai.dgx.teacher_check", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr("compass_ai.dgx.run_teacher_pilot", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    plan = build_dgx_plan(ROOT, ROOT / "configs" / "dgx" / "argos_v1.yaml")
    assert plan["dry_run"] is True
    assert plan["student_model"] == EXAONE
    assert plan["environment_setup"]["base"].endswith("--extra student")
    assert plan["environment_setup"]["repair_after_qlora_failure"] == plan["environment_setup"]["base"]
    assert [step["number"] for step in plan["steps"]] == list(range(1, len(plan["steps"]) + 1))
    text = "\n".join(step["description"] for step in plan["steps"])
    assert "GPT-OSS" in text and "GLM" in text and "S0" in text


def test_launcher_stops_safely_and_resumes_same_run(monkeypatch, tmp_path: Path):
    config = ROOT / "configs" / "dgx" / "argos_v1.yaml"
    monkeypatch.setattr("compass_ai.dgx.collect_dgx_system_report", lambda *_a, **_k: {"status": "recorded"})
    monkeypatch.setattr("compass_ai.dgx.pre_dgx_check", lambda *_a, **_k: {"status": "ready", "errors": []})
    monkeypatch.setattr("compass_ai.dgx.shutil.disk_usage", lambda *_a, **_k: type("U", (), {"free": 200 * 1024**3})())

    def endpoint_down(*_args, **_kwargs):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr("compass_ai.dgx.teacher_check", endpoint_down)
    first = run_dgx_launcher(ROOT, config, "resume-test", tmp_path)
    assert first["status"] == "blocked"
    assert first["blocked_stage"] == "gpt_oss_teacher_check"
    assert Path(first["state_path"]).exists()

    second = run_dgx_launcher(ROOT, config, "resume-test", tmp_path)
    assert second["status"] == "blocked"
    assert second["completed_stages"] == first["completed_stages"]


def test_launcher_stops_at_independent_review_before_student_training(monkeypatch, tmp_path: Path):
    config = ROOT / "configs" / "dgx" / "argos_v1.yaml"
    monkeypatch.setattr("compass_ai.dgx.collect_dgx_system_report", lambda *_a, **_k: {"status": "recorded"})
    monkeypatch.setattr("compass_ai.dgx.pre_dgx_check", lambda *_a, **_k: {"status": "ready", "errors": []})
    monkeypatch.setattr("compass_ai.dgx.shutil.disk_usage", lambda *_a, **_k: type("U", (), {"free": 200 * 1024**3})())
    monkeypatch.setattr("compass_ai.dgx.teacher_check", lambda *_a, **_k: {"status": "ready"})
    monkeypatch.setattr("compass_ai.dgx.run_teacher_pilot", lambda *_a, **_k: {"status": "complete"})
    monkeypatch.setattr("compass_ai.dgx.comparison_readiness", lambda *_a, **_k: {"status": "ready"})
    monkeypatch.setattr(
        "compass_ai.dgx.verify_teacher_runs",
        lambda *_a, **_k: {"status": "needs_review", "unresolved": 2, "student_training_eligible": 3},
    )
    monkeypatch.setattr("compass_ai.dgx.run_student", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not train")))

    result = run_dgx_launcher(ROOT, config, "review-gate", tmp_path)

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "independent_review"
    assert "independent_verification" not in result["completed_stages"]

    review_file = tmp_path / "review-gate" / "manual_review.jsonl"
    review_file.write_text('{"question_id":"P-001","decision":"reject","reviewer":"human","evidence":"checked"}\n', encoding="utf-8")
    monkeypatch.setattr("compass_ai.dgx.apply_verification_review", lambda *_a, **_k: {"status": "ready"})
    resumed = run_dgx_launcher(ROOT, config, "review-gate", tmp_path)
    assert resumed["blocked_stage"] == "verified_student_dataset_gate"
    assert "independent_verification" in resumed["completed_stages"]
