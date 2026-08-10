"""Core-v1 원본 질문 풀을 한 번 생성하는 결정적 데이터 제작 스크립트."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from math import gcd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass_ai.core_dataset import CORE_FILES, dataset_checksum, file_sha256  # noqa: E402
from compass_ai.jsonl import write_jsonl  # noqa: E402


SPLIT_DOMAINS = {
    "train": {"math": 24, "coding": 24, "science": 24, "logic": 24, "korean": 24},
    "validation": {"math": 5, "coding": 5, "science": 5, "logic": 5, "korean": 4},
    "private_eval": {"math": 10, "coding": 10, "science": 10, "logic": 9, "korean": 9},
}
PREFIX = {"train": "TR", "validation": "VA", "private_eval": "PE"}
DOMAIN_PREFIX = {"math": "MATH", "coding": "CODE", "science": "SCI", "logic": "LOG", "korean": "KOR"}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def math_question(n: int) -> tuple[str, str, str, str]:
    k = n + 2
    percent = 20 + 5 * (n % 5)
    percent_tenths = k * percent
    percent_answer = str(percent_tenths // 10) if percent_tenths % 10 == 0 else f"{percent_tenths / 10:g}"
    cases = [
        (f"방정식 {k}x + {k + 3} = {k * (k + 1) + k + 3}을 만족하는 x를 구하시오.", str(k + 1), "linear_equation", "calculation"),
        (f"가로가 {k + 4} cm, 세로가 {k + 1} cm인 직사각형의 넓이를 구하시오.", f"{(k + 4) * (k + 1)} cm²", "geometry", "calculation"),
        (f"{k * 10}의 {percent}%는 얼마인가?", percent_answer, "percentage", "calculation"),
        (f"첫째항이 {k}, 공차가 {n % 4 + 2}인 등차수열의 제{n % 6 + 6}항을 구하시오.", str(k + (n % 6 + 5) * (n % 4 + 2)), "sequence", "calculation"),
        (f"{12 * k}와 {18 * k}의 최대공약수를 구하시오.", str(gcd(12 * k, 18 * k)), "number_theory", "calculation"),
        (f"주머니에 빨간 공 {k}개와 파란 공 {k + 2}개가 있다. 한 개를 무작위로 꺼낼 때 빨간 공일 확률을 기약분수로 쓰시오.", f"{k // gcd(k, 2 * k + 2)}/{(2 * k + 2) // gcd(k, 2 * k + 2)}", "probability", "exact_match"),
        (f"좌표평면의 두 점 ({n}, {n + 1})과 ({n + 3}, {n + 5}) 사이의 거리를 구하시오.", "5", "coordinate_geometry", "calculation"),
        (f"연속한 두 자연수의 합이 {2 * k + 1}이다. 두 수 중 큰 수를 구하시오.", str(k + 1), "word_problem", "calculation"),
    ]
    return cases[(n - 1) % len(cases)]


def coding_question(n: int) -> tuple[str, str, str, str]:
    k = n + 2
    cases = [
        (f"다음 Python 코드의 출력값을 쓰시오: `x = {k}; print(x * 2 + 1)`", str(k * 2 + 1), "python_trace", "exact_match"),
        (f"다음 Python 코드의 출력값을 쓰시오: `print(sum(range(1, {k + 1})))`", str(k * (k + 1) // 2), "loop_trace", "exact_match"),
        (f"정수 {k * 3}을 2진수로 나타내시오. 접두사 `0b`는 쓰지 않는다.", bin(k * 3)[2:], "representation", "exact_match"),
        (f"길이가 n인 배열을 처음부터 끝까지 한 번 순회하는 알고리즘의 시간 복잡도를 Big-O로 쓰시오. 문항 번호는 {n}이다.", "O(n)", "complexity", "exact_match"),
        (f"다음 Python 표현식의 결과를 쓰시오: `[x for x in range({k + 5}) if x % 2 == 0][-1]`", str((k + 4) // 2 * 2), "list_comprehension", "exact_match"),
        (f"다음 Python 코드 실행 후 `count`의 값은? `count=0\nfor i in range({k}):\n    if i % 3 == 0: count += 1`", str((k - 1) // 3 + 1), "branch_trace", "exact_match"),
        (f"0부터 시작하는 배열 `a = list(range({k + 4}))`에서 `a[-2]`의 값을 쓰시오.", str(k + 2), "indexing", "exact_match"),
        (f"불리언 식 `({k} > {k - 1}) and ({k} % 2 == 0)`의 Python 결과를 `True` 또는 `False`로 쓰시오.", "True" if k % 2 == 0 else "False", "boolean_logic", "exact_match"),
    ]
    return cases[(n - 1) % len(cases)]


def science_question(n: int) -> tuple[str, str, str, str]:
    k = n + 2
    cases = [
        (f"질량 {k} kg인 물체에 {k * 3} N의 알짜힘이 작용한다. 가속도를 구하시오.", "3 m/s²", "mechanics", "unit_calculation"),
        (f"저항 {k} Ω에 {k * 2} V를 걸었다. 흐르는 전류를 구하시오.", "2 A", "electricity", "unit_calculation"),
        (f"속력 {k + 3} m/s로 {k}초 동안 등속 운동한 물체의 이동 거리를 구하시오.", f"{(k + 3) * k} m", "kinematics", "unit_calculation"),
        (f"질량 {k * 10} g, 부피 {k * 2} cm³인 물질의 밀도를 구하시오.", "5 g/cm³", "density", "unit_calculation"),
        (f"전력 {k * 10} W인 장치를 {k}초 사용했을 때 소비한 에너지를 구하시오.", f"{k * k * 10} J", "energy", "unit_calculation"),
        (f"진동수가 {k} Hz이고 파장이 {k + 1} m인 파동의 속력을 구하시오.", f"{k * (k + 1)} m/s", "waves", "unit_calculation"),
        (f"{k} Ω과 {k + 3} Ω 저항을 직렬 연결했을 때 합성 저항을 구하시오.", f"{2 * k + 3} Ω", "circuits", "unit_calculation"),
        (f"비열이 4 J/(g·°C)인 물질 {k * 5} g의 온도를 2°C 올리는 데 필요한 열량을 구하시오.", f"{k * 40} J", "thermodynamics", "unit_calculation"),
    ]
    return cases[(n - 1) % len(cases)]


def logic_question(n: int) -> tuple[str, str, str, str]:
    k = n + 2
    cases = [
        (f"수열 {k}, {k + 3}, {k + 6}, {k + 9}, ...의 다음 항을 구하시오.", str(k + 12), "sequence_logic", "calculation"),
        (f"명제 P는 참이고 Q는 거짓이다. `(P and Q) or (not Q)`의 진릿값을 쓰시오. 문항 {n}.", "참", "propositional_logic", "exact_match"),
        (f"모든 A는 B이고, 어떤 C는 A이다. '어떤 C는 B이다'는 반드시 참인가? 예/아니오로 답하시오. 사례 번호 {n}.", "예", "syllogism", "exact_match"),
        (f"민수는 지수보다 {n % 3 + 1}살 많고 지수는 현우보다 2살 많다. 세 사람 중 가장 나이가 많은 사람은 누구인가? 사례 {n}.", "민수", "ordering", "exact_match"),
        (f"집합 A의 원소 수는 {k + 3}, 집합 B의 원소 수는 {k + 1}, 교집합의 원소 수는 {k - 1}이다. 합집합의 원소 수를 구하시오.", str(k + 5), "set_logic", "calculation"),
        (f"세 스위치 A, B, C 중 정확히 하나만 켜져 있다. A는 꺼져 있고 B도 꺼져 있다. 켜진 스위치를 쓰시오. 사례 {n}.", "C", "constraint_logic", "exact_match"),
        (f"'비가 오면 길이 젖는다'와 '길이 젖지 않았다'가 참이다. 비가 왔는가? 예/아니오로 답하시오. 사례 {n}.", "아니오", "modus_tollens", "exact_match"),
        (f"정수 {k}와 {k + 1}의 곱은 홀수인가 짝수인가?", "짝수", "parity", "exact_match"),
    ]
    return cases[(n - 1) % len(cases)]


def korean_question(n: int) -> tuple[str, str, str, str]:
    k = n + 2
    cases = [
        (f"문장 '봄바람이 들판을 천천히 지나간다 {k}'에서 숫자를 제외한 어절 수를 쓰시오.", "4", "word_count", "exact_match"),
        (f"다음 낱말을 가나다순으로 배열하시오: 바다{n}, 가람{n}, 나무{n}", f"가람{n}, 나무{n}, 바다{n}", "sorting", "exact_match"),
        (f"문장 '학생들이 운동장에서 달린다'에서 주어를 쓰시오. 사례 번호는 {n}이다.", "학생들이", "grammar", "exact_match"),
        (f"'기쁘다'의 반대말을 한 단어로 쓰시오. 문항 식별 번호 {n}.", "슬프다", "vocabulary", "exact_match"),
        (f"다음 문장을 높임 표현으로 고치시오: '선생님이 교실에 있다.' 문항 {n}.", "선생님께서 교실에 계신다.", "honorific", "rubric_review"),
        (f"문장 '철수는 책을 읽고 영희는 그림을 그린다'에 포함된 서술어를 순서대로 쓰시오. 사례 {n}.", "읽고, 그린다", "grammar", "exact_match"),
        (f"낱말 '맨손'에서 접두사를 쓰시오. 문항 식별 번호 {n}.", "맨-", "morphology", "exact_match"),
        (f"'가는 말이 고와야 오는 말이 곱다'가 강조하는 태도를 '상호 존중' 또는 '속도 경쟁' 중 고르시오. 사례 {n}.", "상호 존중", "meaning", "exact_match"),
    ]
    return cases[(n - 1) % len(cases)]


BUILDERS = {"math": math_question, "coding": coding_question, "science": science_question, "logic": logic_question, "korean": korean_question}


def make_record(split: str, domain: str, local_number: int, domain_number: int) -> tuple[dict, dict | None]:
    question, answer, task_type, method = BUILDERS[domain](domain_number)
    qid = f"CORE-{PREFIX[split]}-{DOMAIN_PREFIX[domain]}-{local_number:03d}"
    difficulty = ("easy", "medium", "hard")[(domain_number - 1) % 3]
    auto = method != "rubric_review"
    rubric_id = f"CORE-{DOMAIN_PREFIX[domain]}-{'AUTO' if auto else 'REVIEW'}-v1"
    common = {
        "question_id": qid,
        "dataset_version": "core-v1",
        "split": split,
        "domain": domain,
        "difficulty": difficulty,
        "task_type": task_type,
        "question": question,
        "verification_method": method,
        "provenance": {"origin": "original", "license": "project-authored", "generator": "core-v1"},
    }
    criteria = ["최종 답이 기준 답과 일치한다."] if auto else ["높임 주체와 서술어가 모두 적절하다.", "문장의 의미가 유지된다."]
    if split != "private_eval":
        common["reference"] = {"visibility": "verification_only", "answer": answer, "answer_sha256": sha(answer)}
        common["rubric"] = {"visibility": "verification_only", "rubric_id": rubric_id, "auto_verifiable": auto, "criteria": criteria}
        return common, None
    key_id = f"CORE-PRIVATE-{sha(qid)[:16]}"
    common["reference"] = {"visibility": "private", "key_id": key_id, "answer_sha256": sha(answer)}
    common["rubric"] = {"visibility": "private", "rubric_id": rubric_id, "auto_verifiable": auto}
    private_key = {"key_id": key_id, "question_id": qid, "answer": answer, "criteria": criteria, "verification_method": method}
    return common, private_key


def generate(root: Path, force: bool = False) -> dict:
    core_dir = root / "data" / "questions" / "core_v1"
    if core_dir.exists() and any(core_dir.iterdir()) and not force:
        raise FileExistsError(f"Core-v1 already exists: {core_dir}")
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    private_keys: list[dict] = []
    domain_serial = Counter()
    for split, distribution in SPLIT_DOMAINS.items():
        for domain, count in distribution.items():
            for local_number in range(1, count + 1):
                domain_serial[domain] += 1
                row, key = make_record(split, domain, local_number, domain_serial[domain])
                rows_by_split[split].append(row)
                if key:
                    private_keys.append(key)
    for split, filename in CORE_FILES.items():
        write_jsonl(core_dir / filename, rows_by_split[split])

    private_key_path = root / "artifacts" / "private" / "core_v1_private_key.jsonl"
    write_jsonl(private_key_path, private_keys)
    hashes = {filename: file_sha256(core_dir / filename) for filename in CORE_FILES.values()}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    summary = {
        "question_count": len(all_rows),
        "split_counts": {split: len(rows_by_split[split]) for split in CORE_FILES},
        "domain_counts": dict(sorted(Counter(row["domain"] for row in all_rows).items())),
        "auto_verifiable_count": sum(row["rubric"]["auto_verifiable"] for row in all_rows),
    }
    manifest = {
        "dataset_name": "CoMPaSS-AI Core",
        "dataset_version": "core-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "file_sha256": hashes,
        "dataset_checksum": dataset_checksum(hashes),
        "private_key_sha256": file_sha256(private_key_path),
        "private_key_location": "artifacts/private (gitignored; transfer separately)",
        "summary": summary,
    }
    (core_dir / "core_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.root, args.force), ensure_ascii=False, indent=2))
