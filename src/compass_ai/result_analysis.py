"""동결 protocol에 맞춘 condition별 평가 집계와 paired McNemar 입력 생성."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from math import comb
import json
from pathlib import Path

from .jsonl import read_jsonl


CONDITIONS = ["S0", "S1", "S2", "S3a", "S3b", "S3_matched", "S4_teacher_only", "S4_with_human_correction", "S5"]
PRIMARY_CONDITIONS = CONDITIONS[:-1]
DOMAINS = ["math", "coding", "science", "logic", "korean"]
CORE_V2_CHECKSUM = "49e72cdcca2425d75080bf95428cb9d1ff8a9f53b016b5e7bcc409422a203fe2"
PROTOCOL_CHECKSUM = "a8d67d3e706838ace6d37c3058d7790fa77912976c101eb761106eb0febedd55"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(left_only, right_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def _template(output_dir: Path) -> dict:
    rows = [{"condition": condition, "overall": "", **{domain: "" for domain in DOMAINS}} for condition in CONDITIONS]
    _write_csv(output_dir / "condition_results.csv", ["condition", "overall", *DOMAINS], rows)
    _write_csv(
        output_dir / "paired_items.csv",
        ["evaluation_id", "domain", "S3_matched_correct", "S4_teacher_only_correct"], [],
    )
    report = {
        "status": "awaiting_results", "performance_claim": False,
        "dataset_checksum": CORE_V2_CHECKSUM, "protocol_checksum": PROTOCOL_CHECKSUM,
        "condition_order": CONDITIONS, "domains": DOMAINS,
    }
    (output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def analyze_results(input_dir: Path, output_dir: Path, allow_synthetic: bool = False) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"analysis output collision: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.rglob("scored_items.jsonl")) if input_dir.exists() else []
    if not paths:
        return _template(output_dir)
    items = [row for path in paths for row in read_jsonl(path)]
    modes = {row.get("mode", "actual") for row in items}
    if modes != {"actual"} and not (allow_synthetic and modes == {"synthetic_pipeline_validation"}):
        raise ValueError(f"synthetic/mixed evaluation records are not allowed: {sorted(modes)}")
    required = {"run_id", "condition", "evaluation_id", "domain", "correct", "invalid", "unresolved", "mode", "dataset_checksum", "protocol_checksum"}
    for index, row in enumerate(items, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"scored item {index} missing: {sorted(missing)}")
        if row["condition"] not in CONDITIONS or row["domain"] not in DOMAINS:
            raise ValueError(f"scored item {index} has invalid condition/domain")
        if row["dataset_checksum"] != CORE_V2_CHECKSUM or row["protocol_checksum"] != PROTOCOL_CHECKSUM:
            raise ValueError("evaluation checksum/protocol mismatch")
        if row["invalid"] or row["unresolved"]:
            row["correct"] = False
    grouped = defaultdict(list)
    for row in items:
        grouped[row["condition"]].append(row)
    missing_primary = set(PRIMARY_CONDITIONS) - set(grouped)
    if missing_primary:
        raise ValueError(f"missing frozen primary conditions: {sorted(missing_primary)}")
    active_conditions = [condition for condition in CONDITIONS if grouped[condition]]
    expected_ids = None
    for condition in active_conditions:
        ids = {row["evaluation_id"] for row in grouped[condition]}
        if expected_ids is None:
            expected_ids = ids
        if ids != expected_ids or len(ids) != len(grouped[condition]):
            raise ValueError("conditions do not share the same unique private_eval item set")
    if len(expected_ids or set()) != 48:
        raise ValueError("Core-v2 private evaluation must contain 48 items")

    csv_rows, condition_summary = [], {}
    for condition in CONDITIONS:
        if not grouped[condition]:
            csv_rows.append({"condition": condition, "overall": "", **{domain: "" for domain in DOMAINS}})
            condition_summary[condition] = {"status": "deferred_follow_up_placeholder"}
            continue
        rows = grouped[condition]
        by_domain = defaultdict(list)
        for row in rows: by_domain[row["domain"]].append(row)
        overall = sum(bool(row["correct"]) for row in rows) / len(rows)
        domains = {domain: sum(bool(row["correct"]) for row in by_domain[domain]) / len(by_domain[domain]) for domain in DOMAINS}
        counts = Counter(
            "invalid" if row["invalid"] else "unresolved" if row["unresolved"] else "correct" if row["correct"] else "incorrect"
            for row in rows
        )
        csv_rows.append({"condition": condition, "overall": f"{overall:.6f}", **{domain: f"{domains[domain]:.6f}" for domain in DOMAINS}})
        condition_summary[condition] = {
            "overall": overall, "domains": domains,
            **{name: counts[name] for name in ["correct", "incorrect", "invalid", "unresolved"]},
        }
    _write_csv(output_dir / "condition_results.csv", ["condition", "overall", *DOMAINS], csv_rows)

    left = {row["evaluation_id"]: row for row in grouped["S4_teacher_only"]}
    right = {row["evaluation_id"]: row for row in grouped["S3_matched"]}
    paired_rows, s4_only, s3_only = [], 0, 0
    for evaluation_id in sorted(left):
        s4, s3 = bool(left[evaluation_id]["correct"]), bool(right[evaluation_id]["correct"])
        s4_only += s4 and not s3; s3_only += s3 and not s4
        paired_rows.append({
            "evaluation_id": evaluation_id, "domain": left[evaluation_id]["domain"],
            "S3_matched_correct": s3, "S4_teacher_only_correct": s4,
        })
    _write_csv(output_dir / "paired_items.csv", ["evaluation_id", "domain", "S3_matched_correct", "S4_teacher_only_correct"], paired_rows)
    delta = condition_summary["S4_teacher_only"]["overall"] - condition_summary["S3_matched"]["overall"]
    report = {
        "status": "complete", "mode": next(iter(modes)),
        "performance_claim": modes == {"actual"},
        "dataset_checksum": CORE_V2_CHECKSUM, "protocol_checksum": PROTOCOL_CHECKSUM,
        "condition_results": condition_summary,
        "primary_comparison": {
            "name": "S4_teacher_only - S3_matched", "difference": delta, "paired_n": 48,
            "discordant_s4_correct": s4_only, "discordant_s3_correct": s3_only,
            "mcnemar_exact_p": _mcnemar_exact(s4_only, s3_only),
        },
        "notice": "Synthetic mode validates aggregation only and is not an ARGOS performance result." if modes != {"actual"} else None,
    }
    (output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report
