import json
import sys
from pathlib import Path

import pytest

from compass_ai.conditions import MatchedComparisonError, assert_matched_comparison, build_conditions, condition_stats
from compass_ai.leakage import assert_private_final_is_safe, find_id_overlap
from compass_ai.logging_utils import create_run_logger
from compass_ai.pipeline import run_mock_pipeline
from compass_ai.pilot import assess_pilot_questions, create_pilot_manifest, validate_pilot_questions
from compass_ai.pre_dgx import pre_dgx_check
from compass_ai.schemas import SchemaError, validate_questions, validate_responses
from compass_ai.teacher import GLMBackend, GPTOSSBackend, generate_responses, merge_responses
from compass_ai.verification import verify_responses
from compass_ai.teacher_run import comparison_readiness, resume_question_ids, validate_teacher_config, validate_teacher_raw_responses


def question(question_id="q1", answer="4", **extra):
    data = {
        "question_id": question_id, "split": "pilot", "domain": "math",
        "task_type": "calculation", "question": "2 + 2는?", "reference_answer": answer,
        "verification_method": "answer_key", "evidence_source": "mock answer key",
        "difficulty": "easy", "requires_current_info": False, "notes": "test",
    }
    data.update(extra)
    return data


def response(question_id, teacher, answer, **extra):
    data = {
        "question_id": question_id, "domain": "math", "task_type": "calculation",
        "question": "2 + 2는?", "teacher_model": teacher, "checkpoint": "mock-v1",
        "quantization": "none", "generation_config": {"temperature": 0},
        "raw_response": answer, "final_answer": answer, "verification_status": "pending",
        "verification_method": "answer_key", "evidence": "mock", "error_type": None,
        "tag": "teacher_raw", "reviewer": None, "verification_outcome": "pending", "timestamp": "2026-08-09T00:00:00+00:00",
        "training_approved": False,
    }
    data.update(extra)
    return data


def test_question_schema_rejects_missing_and_duplicate_ids():
    with pytest.raises(SchemaError, match="question_id"):
        validate_questions([{"domain": "math"}])
    with pytest.raises(SchemaError, match="duplicate"):
        validate_questions([question(), question()])
    assert validate_questions([question()])[0]["question_id"] == "q1"


def test_response_schema_rejects_invalid_record():
    with pytest.raises(SchemaError, match="teacher_model"):
        validate_responses([{"question_id": "q1"}])


def test_merge_and_conflict_categories_use_normalized_answers():
    records = [
        response("same", "MockTeacherA", " 4 "), response("same", "MockTeacherB", "4.0"),
        response("different", "MockTeacherA", "4"), response("different", "MockTeacherB", "5"),
        response("missing", "MockTeacherA", "4"),
        response("badformat", "MockTeacherA", "4"), response("badformat", "MockTeacherB", "4", error_type="format_error"),
        response("suspect", "MockTeacherA", "unknown"), response("suspect", "MockTeacherB", "unknown"),
    ]
    merged = {item["question_id"]: item for item in merge_responses(records)}
    assert merged["same"]["conflict_type"] == "agree"
    assert merged["different"]["conflict_type"] == "answer_conflict"
    assert merged["missing"]["conflict_type"] == "one_missing"
    assert merged["badformat"]["conflict_type"] == "format_conflict"
    assert merged["suspect"]["conflict_type"] == "both_suspect"


def test_answer_key_verification_updates_tag_and_approval():
    verified = verify_responses([response("q1", "MockTeacherA", "4"), response("q1", "MockTeacherB", "5")], [question()])
    assert verified[0]["verification_status"] == "passed"
    assert verified[0]["tag"] == "teacher_verified"
    assert verified[0]["training_approved"] is True
    assert verified[0]["verification_outcome"] == "teacher_a_selected"
    assert verified[1]["verification_status"] == "failed"
    assert verified[1]["tag"] == "excluded"


def test_condition_builds_expose_s3a_s3b_and_s4_variants():
    questions = [question("q1"), question("q2", answer="5")]
    records = [
        response("q1", "MockTeacherA", "4", tag="teacher_verified", training_approved=True, verification_outcome="both_agree_verified"),
        response("q1", "MockTeacherB", "4", tag="teacher_verified", training_approved=False, verification_outcome="both_agree_verified"),
        response("q2", "MockTeacherA", "5", tag="teacher_verified", training_approved=True, verification_outcome="teacher_a_selected"),
        response("q2", "MockTeacherB", "7", tag="excluded", training_approved=False),
    ]
    conditions = build_conditions(questions, records)
    assert len(conditions["S1"]) == 2
    assert len(conditions["S2"]) == 2
    assert len(conditions["S3a"]) == 4
    assert len(conditions["S3b"]) == 2
    assert [item["teacher_model"] for item in conditions["S3b"]] == ["MockTeacherA", "MockTeacherB"]
    assert len(conditions["S4_teacher_only"]) == 2
    assert len(conditions["S4_with_human_correction"]) == 2
    assert condition_stats(conditions["S3a"])["token_count_kind"] == "approximate"


def test_s3b_is_deterministic_and_records_predeclared_rule():
    questions = [question("q1"), question("q2", answer="5"), question("q3", answer="6")]
    records = [
        response(qid, "MockTeacherA", answer, tag="teacher_verified", training_approved=True, verification_outcome="teacher_a_selected")
        for qid, answer in [("q1", "4"), ("q2", "5"), ("q3", "6")]
    ] + [
        response(qid, "MockTeacherB", answer, tag="excluded", training_approved=False, verification_outcome="excluded")
        for qid, answer in [("q1", "4"), ("q2", "5"), ("q3", "6")]
    ]
    first = build_conditions(questions, records)
    second = build_conditions(questions, records)
    assert [row["teacher_model"] for row in first["S3b"]] == ["MockTeacherA", "MockTeacherB", "MockTeacherA"]
    assert first["selection_metadata"] == second["selection_metadata"]
    assert first["selection_metadata"]["rule"] == "alternating_sorted_question_id_no_quality_signal"


def test_s4_creates_human_correction_and_reports_ratio():
    questions = [question("q1")]
    verified = verify_responses([
        response("q1", "MockTeacherA", "wrong"),
        response("q1", "MockTeacherB", "also wrong"),
    ], questions)
    conditions = build_conditions(questions, verified)
    assert len(conditions["S4_teacher_only"]) == 0
    assert len(conditions["S4_with_human_correction"]) == 1
    assert conditions["S4_with_human_correction"][0]["verification_outcome"] == "human_corrected"
    assert condition_stats(conditions["S4_with_human_correction"])["human_corrected_ratio"] == 1.0


def test_s5_without_korean_source_is_placeholder_not_s4_copy():
    conditions = build_conditions([question()], [response("q1", "MockTeacherA", "4", training_approved=True, tag="teacher_verified", verification_outcome="teacher_a_selected")])
    assert conditions["S5"] == []
    assert conditions["condition_metadata"]["S5"]["status"] == "placeholder"


def test_s3_matched_has_same_ids_count_and_domains_as_s4_teacher_only():
    questions = [question("q1"), question("q2", answer="5"), question("q3", answer="6")]
    records = [
        response("q1", "MockTeacherA", "4", training_approved=True, tag="teacher_verified", verification_outcome="teacher_a_selected"),
        response("q1", "MockTeacherB", "4"),
        response("q2", "MockTeacherA", "5", training_approved=True, tag="teacher_verified", verification_outcome="teacher_a_selected"),
        response("q2", "MockTeacherB", "5"),
        response("q3", "MockTeacherA", "6"),
        response("q3", "MockTeacherB", "6", training_approved=True, tag="teacher_verified", verification_outcome="teacher_b_selected"),
    ]
    conditions = build_conditions(questions, records)
    matched, teacher_only = conditions["S3_matched"], conditions["S4_teacher_only"]
    assert_matched_comparison(matched, teacher_only)
    assert [row["teacher_model"] for row in matched] == ["MockTeacherA", "MockTeacherB", "MockTeacherA"]
    assert conditions["condition_metadata"]["S3_matched"]["validation"] == "passed"


def test_matched_comparison_rejects_domain_or_id_difference():
    left = [response("q1", "MockTeacherA", "4")]
    right = [response("q2", "MockTeacherA", "4")]
    with pytest.raises(MatchedComparisonError, match="question_id"):
        assert_matched_comparison(left, right)


def test_pilot_validation_rejects_mock_and_quality_report_flags_duplicates(tmp_path):
    records = [question("pilot-001", split="pilot", language="ko", source_type="official", source_uri_or_reference="https://example.edu", answer_format="short", verification_notes="answer key checked"), question("pilot-002", split="pilot", question="2+2는?", language="ko", source_type="official", source_uri_or_reference="https://example.edu", answer_format="short", verification_notes="answer key checked")]
    report = assess_pilot_questions(validate_pilot_questions(records), private_ids=[])
    assert report["errors"] == []
    assert report["warnings"][0]["type"] == "possible_text_duplicate"
    with pytest.raises(SchemaError, match="Mock"):
        validate_pilot_questions([question("mock-01", split="pilot")])


def test_pilot_manifest_is_deterministic_for_same_source(tmp_path):
    source = tmp_path / "pilot_questions.jsonl"
    source.write_text(json.dumps(question("pilot-001", split="pilot"), ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = create_pilot_manifest(source, validate_pilot_questions([question("pilot-001", split="pilot")]), dataset_version="pilot-v1")
    assert manifest["question_count"] == 1
    assert len(manifest["question_id_sha256"]) == 64
    assert len(manifest["source_file_sha256"]) == 64


def test_pilot_accepts_extended_methods_and_preserves_political_pairing():
    china = question(
        "P-CN-001", domain="politics_china", task_type="sensitive_history",
        verification_method="official_source_review", dataset_version="pilot-v1",
        rubric="R-POL", origin="original", paired_control_ids=["P-CTRL-001", "P-CTRL-002"],
    )
    controls = [
        question(
            question_id, domain="politics_control", task_type="sensitive_history",
            verification_method="official_archive_review", dataset_version="pilot-v1",
            rubric="R-POL", origin="original", paired_china_id="P-CN-001",
        )
        for question_id in ["P-CTRL-001", "P-CTRL-002"]
    ]
    report = assess_pilot_questions(validate_pilot_questions([china, *controls]), private_ids=[])
    assert report["political_pairing"] == {"china_questions": 1, "control_questions": 2, "valid_pairs": 2}


def test_response_status_is_optional_but_rejects_unknown_values():
    assert validate_responses([response("q1", "MockTeacherA", "4", response_status="evasive")])[0]["response_status"] == "evasive"
    with pytest.raises(SchemaError, match="response_status"):
        validate_responses([response("q1", "MockTeacherA", "4", response_status="invented")])


def test_pilot_notes_metadata_is_preserved_as_structured_fields():
    record = question(
        "P-CN-001", domain="politics_china",
        notes="rubric=R-POL; origin=original; paired_controls=P-CTRL-001,P-CTRL-002; record response_status",
    )
    normalized = validate_pilot_questions([record])[0]
    assert normalized["rubric"] == "R-POL"
    assert normalized["origin"] == "original"
    assert normalized["paired_control_ids"] == ["P-CTRL-001", "P-CTRL-002"]
    assert normalized["response_status_required"] is True


def test_pilot_report_separates_auto_human_and_hybrid_grading():
    records = validate_pilot_questions([
        question("P-001", verification_method="answer_key"),
        question("P-002", verification_method="official_source_review"),
        question("P-003", verification_method="rubric_and_constraints"),
    ])
    report = assess_pilot_questions(records, private_ids=[])
    assert report["grading_modes"] == {"auto": 1, "human": 1, "hybrid": 1}


def test_pre_dgx_check_reports_missing_actual_pilot_as_not_ready(tmp_path):
    (tmp_path / "evaluation" / "private_final").mkdir(parents=True)
    (tmp_path / "evaluation" / "private_final" / "README.md").write_text("placeholder", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "dgx_runbook.md").write_text("runbook", encoding="utf-8")
    result = pre_dgx_check(tmp_path, run_tests=False)
    assert result["status"] == "not_ready"
    assert "actual pilot dataset missing" in result["not_ready_reasons"]


def test_future_backends_fail_explicitly_instead_of_mock_fallback():
    with pytest.raises(NotImplementedError, match="not implemented"):
        GPTOSSBackend().generate(question())
    with pytest.raises(NotImplementedError, match="not implemented"):
        GLMBackend().generate(question())


def test_leakage_guard_blocks_private_files_and_id_overlap(tmp_path):
    private = tmp_path / "private_final"
    private.mkdir()
    (private / "README.md").write_text("placeholder", encoding="utf-8")
    assert_private_final_is_safe(private)
    (private / "secret.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="private_final"):
        assert_private_final_is_safe(private)
    assert find_id_overlap(["q1"], ["q1", "q2"]) == ["q1"]


def test_run_logger_creates_reproducibility_files(tmp_path):
    logger = create_run_logger(tmp_path, {"seed": 7}, ["compass-ai", "mock-e2e"])
    logger.event("started", {"ok": True})
    logger.write_metrics({"count": 1})
    assert (logger.path / "config.json").exists()
    assert (logger.path / "environment.json").exists()
    assert (logger.path / "metrics.json").exists()
    assert "started" in (logger.path / "events.jsonl").read_text(encoding="utf-8")


def test_mock_teachers_generate_expected_records():
    records = generate_responses([question("q1", mock_case="same")])
    assert {record["teacher_model"] for record in records} == {"MockTeacherA", "MockTeacherB"}
    assert len(records) == 2


def test_cli_exposes_individual_pipeline_commands(monkeypatch, capsys, tmp_path):
    from compass_ai.__main__ import main
    monkeypatch.setattr(sys, "argv", ["compass-ai", "generate", "--root", str(tmp_path)])
    main()
    assert "generated" in capsys.readouterr().out


def test_full_mock_e2e_creates_conditions_logs_and_valid_jsonl(tmp_path):
    result = run_mock_pipeline(tmp_path, seed=11)
    assert result["question_count"] >= 20
    assert {"S0", "S1", "S2", "S3a", "S3b", "S3_matched", "S4", "S4_teacher_only", "S4_with_human_correction", "S5"} <= set(result["condition_counts"])
    assert result["condition_counts"]["S0"] == 0
    assert result["condition_counts"]["S4"] > 0
    assert result["condition_metadata"]["S5"]["status"] == "placeholder"
    assert result["condition_stats"]["S4"]["human_corrected_ratio"] > 0
    assert result["condition_metadata"]["S3_matched"]["validation"] == "passed"
    for relative in ["data/merged/mock_merged.jsonl", "data/verified/mock_verified.jsonl", "reports/mock_summary.json"]:
        assert (tmp_path / relative).exists(), relative
    with (tmp_path / "data/verified/mock_verified.jsonl").open(encoding="utf-8") as handle:
        assert all(json.loads(line)["question_id"] for line in handle if line.strip())


def test_teacher_run_contract_rejects_mismatch_and_keeps_resume_ids(tmp_path):
    config = {"experiment_id": "pilot-v1-run", "dataset_name": "compass-ai-pilot", "dataset_version": "pilot-v1", "dataset_checksum": "a" * 64, "teacher_model": "GPT-OSS-120B", "checkpoint": None, "revision": None, "quantization": None, "inference_engine": None, "system_prompt": "same", "temperature": 0, "top_p": 1, "max_new_tokens": 512, "seed": 7, "num_repeats": 1, "timestamp": None, "environment_log_reference": None, "status": "proposed"}
    assert validate_teacher_config(config)["status"] == "proposed"
    row = {"question_id": "P-001", "teacher_model": "GPT-OSS-120B", "dataset_version": "pilot-v1", "dataset_checksum": "a" * 64, "run_id": "run-1", "raw_response": "answer", "final_answer": "answer", "generation_config_reference": "b" * 64, "response_started_at": None, "response_finished_at": None, "latency_seconds": None, "ttft_seconds": None, "input_tokens": None, "output_tokens": None, "response_status": "answered", "error": None, "metadata": {}}
    assert resume_question_ids(validate_teacher_raw_responses([row])) == {"P-001"}
    gpt, glm = tmp_path / "gpt.jsonl", tmp_path / "glm.jsonl"
    gpt.write_text(json.dumps(row) + "\n", encoding="utf-8")
    glm.write_text(json.dumps({**row, "teacher_model": "GLM-4.5-Air", "dataset_checksum": "c" * 64}) + "\n", encoding="utf-8")
    report = comparison_readiness(gpt, glm, expected_dataset_checksum="a" * 64)
    assert report["status"] == "error"
    assert "dataset_checksum_mismatch" in report["errors"]


def test_comparison_readiness_reports_missing_actual_results(tmp_path):
    report = comparison_readiness(tmp_path / "missing-gpt.jsonl", tmp_path / "missing-glm.jsonl", expected_dataset_checksum="a" * 64)
    assert report == {"status": "not_ready", "errors": [], "reason": "teacher results missing"}
