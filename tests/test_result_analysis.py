from __future__ import annotations

import csv
from pathlib import Path

import pytest

from compass_ai.jsonl import write_jsonl
from compass_ai.result_analysis import CONDITIONS, CORE_V2_CHECKSUM, DOMAINS, PROTOCOL_CHECKSUM, analyze_results
from compass_ai.teacher_api import load_teacher_api_config


ROOT = Path(__file__).resolve().parents[1]


def test_analysis_without_results_writes_non_performance_template(tmp_path: Path):
    report = analyze_results(tmp_path / "missing", tmp_path / "summary")
    assert report["status"] == "awaiting_results"
    assert report["performance_claim"] is False
    with (tmp_path / "summary" / "condition_results.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(CONDITIONS)


def test_synthetic_analysis_is_explicit_and_paired(tmp_path: Path):
    input_dir = tmp_path / "evaluation"
    rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        for index in range(48):
            rows.append({
                "run_id": "synthetic-test", "condition": condition,
                "evaluation_id": f"PE-{index:03d}", "domain": DOMAINS[index % len(DOMAINS)],
                "correct": index < condition_index, "invalid": index == 47 and condition == "S0",
                "unresolved": False, "mode": "synthetic_pipeline_validation",
                "dataset_checksum": CORE_V2_CHECKSUM, "protocol_checksum": PROTOCOL_CHECKSUM,
            })
    write_jsonl(input_dir / "scored_items.jsonl", rows)
    with pytest.raises(ValueError, match="synthetic"):
        analyze_results(input_dir, tmp_path / "blocked")
    report = analyze_results(input_dir, tmp_path / "summary", allow_synthetic=True)
    assert report["status"] == "complete"
    assert report["performance_claim"] is False
    assert report["primary_comparison"]["paired_n"] == 48
    assert set(report["condition_results"]["S0"]) >= {"correct", "incorrect", "invalid", "unresolved"}


def test_core_v2_teacher_configs_are_static_and_download_nothing():
    gpt = load_teacher_api_config(ROOT / "configs" / "teachers" / "gpt_oss_core_v2_dgx.yaml")
    glm = load_teacher_api_config(ROOT / "configs" / "teachers" / "glm45_air_core_v2_dgx.yaml")
    assert gpt["dataset_version"] == glm["dataset_version"] == "core-v2"
    assert gpt["dataset_checksum"] == glm["dataset_checksum"] == "22652b4f1e34e7f3ec9bbc0fa18c3114204e8dce4d6171b3a49967ead51f4715"
