"""본실험용 Core-v2 원본 문항을 최초 1회 구축한다.

Core-v1은 읽거나 수정하지 않는다. private_eval 정답은 gitignored artifacts에만 저장한다.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CODE = {"train": "TR", "validation": "VA", "private_eval": "PE"}
DOMAIN_CODE = {"math": "MATH", "coding": "CODE", "science": "SCI", "logic": "LOG", "korean": "KOR"}
EXPECTED = {"train": 120, "validation": 24, "private_eval": 48}
ROWS: list[dict] = []


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def add(
    split: str, domain: str, family: str, difficulty: str, task_type: str,
    question: str, answer: str, method: str = "exact_match", steps: int = 2,
    criteria: list[str] | None = None,
) -> None:
    local = sum(row["split"] == split and row["domain"] == domain for row in ROWS) + 1
    question_id = f"CORE2-{SPLIT_CODE[split]}-{DOMAIN_CODE[domain]}-{local:03d}"
    auto = method != "rubric_review"
    rubric_id = f"CORE2-{family.upper()}-v1"
    common = {
        "question_id": question_id,
        "dataset_version": "core-v2",
        "split": split,
        "domain": domain,
        "difficulty": difficulty,
        "task_type": task_type,
        "template_family": f"{split}:{domain}:{family}",
        "question": question,
        "verification_method": method,
        "rubric": {
            "visibility": "private" if split == "private_eval" else "verification_only",
            "rubric_id": rubric_id,
            "auto_verifiable": auto,
            "reasoning_steps": steps,
        },
        "provenance": {"origin": "project-authored", "version": "core-v2", "review_state": "draft"},
        "_answer": answer,
        "_criteria": criteria or (["최종 답이 기준 답과 일치한다."] if auto else ["핵심 근거가 모두 포함된다."]),
    }
    ROWS.append(common)


def math_bank() -> None:
    s = "train"
    add(s,"math","ratio-allocation","medium","ratio_reasoning","학생회 예산 84만 원을 홍보와 운영에 3:4로 나눈 뒤, 운영비의 25%를 안전 장비에 썼다. 남은 운영비는 몇 만 원인가?","36", "calculation",3)
    add(s,"math","system-bundle","medium","linear_system","공책 2권과 펜 3자루는 8,000원이고, 공책 3권과 펜 1자루는 7,800원이다. 공책 한 권과 펜 한 자루의 가격 합은 얼마인가?","3400원","calculation",3)
    add(s,"math","tiered-fare","medium","piecewise_cost","택시는 기본 2 km까지 4,000원이고 이후 500 m마다 600원이 추가된다. 5.2 km를 이동할 때 요금은 얼마인가? 단, 남은 거리는 500 m 단위로 올림한다.","8200원","calculation",3)
    add(s,"math","cutout-area","medium","composite_geometry","가로 12 cm, 세로 9 cm인 직사각형의 한 모서리에서 가로 4 cm, 세로 3 cm인 직사각형을 잘라 냈다. 남은 도형의 넓이는 얼마인가?","96 cm²","calculation",2)
    add(s,"math","successive-percent","medium","percent_change","가격을 20% 올린 뒤 새 가격에서 20% 할인했다. 처음 가격이 50,000원이면 최종 가격과 처음 가격의 차이는 얼마인가?","2000원 감소","calculation",3)
    add(s,"math","sequence-threshold","hard","arithmetic_sequence","첫째항이 7이고 공차가 5인 등차수열에서 처음으로 100보다 큰 항은 몇 번째 항이며 그 값은 얼마인가?","20번째 항, 102","calculation",4)
    add(s,"math","without-replacement","medium","probability","빨간 공 3개와 파란 공 2개가 든 주머니에서 공을 되돌려 넣지 않고 2개를 뽑는다. 서로 다른 색이 나올 확률을 기약분수로 쓰시오.","3/5","calculation",3)
    add(s,"math","missing-score","medium","mean_reasoning","다섯 번의 측정값 평균이 18이다. 앞의 네 값이 12, 17, 21, 19라면 마지막 값은 얼마인가?","21","calculation",2)
    add(s,"math","midpoint-reflection","medium","coordinate_reasoning","점 A(2,-1)와 B(8,5)의 중점을 M이라 하자. A를 M에 대하여 대칭이동한 점의 좌표를 구하시오.","(8, 5)","calculation",3)
    add(s,"math","quadratic-factor","medium","quadratic_equation","직사각형의 가로는 세로보다 3 cm 길고 넓이는 40 cm²이다. 세로의 길이를 구하시오.","5 cm","calculation",3)
    add(s,"math","divisibility-constraints","hard","number_constraints","두 자리 자연수는 각 자리 숫자의 합이 11이고, 십의 자리와 일의 자리를 바꾼 수보다 27 크다. 원래 수를 구하시오.","74","calculation",4)
    add(s,"math","combined-work","hard","work_rate","A는 혼자 6시간, B는 혼자 3시간에 일을 끝낸다. 둘이 함께 1시간 일한 뒤 B가 떠났다면, 남은 일을 A가 끝내는 데 몇 시간이 필요한가?","3시간","calculation",4)
    add(s,"math","round-trip-speed","hard","average_speed","같은 거리를 갈 때는 시속 60 km, 돌아올 때는 시속 40 km로 이동했다. 전체 왕복 평균 속력을 구하시오.","48 km/h","calculation",4)
    add(s,"math","fraction-balance","medium","fraction_equation","어떤 수의 3/5에서 4를 뺀 값이 그 수의 1/3과 같다. 그 수를 구하시오.","15","calculation",3)
    add(s,"math","perimeter-relation","medium","geometry_constraints","직사각형의 둘레는 46 cm이고 가로는 세로의 2배보다 1 cm 짧다. 가로와 세로를 구하시오.","가로 15 cm, 세로 8 cm","calculation",3)
    add(s,"math","exponent-comparison","medium","exponent_rules","2³×4²을 2의 거듭제곱으로 나타낸 뒤 그 값을 구하시오.","2⁷ = 128","calculation",2)
    add(s,"math","compound-units","medium","unit_conversion","분속 90 m로 8분 20초 동안 일정하게 이동했다. 이동 거리를 km 단위로 쓰시오.","0.75 km","calculation",3)
    add(s,"math","weighted-mean","medium","weighted_average","중간고사 40%, 기말고사 60%를 반영한다. 두 점수가 각각 75점과 90점이면 반영 점수는 얼마인가?","84점","calculation",2)
    add(s,"math","three-set-count","hard","inclusion_exclusion","학생 40명 중 축구 22명, 농구 18명, 두 종목 모두 9명이다. 두 종목 모두 하지 않는 학생 수를 구하시오.","9명","calculation",3)
    add(s,"math","parallel-angle","medium","angle_reasoning","평행한 두 직선을 한 직선이 가로지를 때 한 엇각이 68°이다. 그 각과 같은 쪽 내각의 합을 이루는 다른 내각의 크기는 얼마인가?","112°","calculation",2)
    add(s,"math","integer-max-product","hard","discrete_optimization","합이 17인 서로 다른 양의 정수 두 개의 곱을 최대로 하려 한다. 두 정수와 최대 곱을 구하시오.","8과 9, 최대 곱 72","calculation",3)
    add(s,"math","function-composition","medium","function_composition","f(x)=2x-3, g(x)=x²+1일 때 g(f(4))-f(g(2))를 구하시오.","19","calculation",3)
    add(s,"math","discount-tax","hard","successive_operations","정가 80,000원인 물건을 15% 할인하고, 할인 가격에 배송비 3,000원을 더한 뒤 전체 금액에 10% 부가비용을 적용했다. 최종 금액은 얼마인가?","78100원","calculation",4)
    add(s,"math","table-rule","hard","pattern_inference","입력 1,2,3,4에 대한 출력이 각각 4,9,16,25이다. 출력이 (입력+1)²이라는 규칙을 따른다면 출력 100에 대응하는 입력은 얼마인가?","9","calculation",3)
    s = "validation"
    add(s,"math","diagonal-perimeter","medium","pythagorean_geometry","대각선 길이가 13 cm이고 가로가 12 cm인 직사각형의 둘레를 구하시오.","34 cm","calculation",3)
    add(s,"math","inverse-workers","hard","inverse_proportion","같은 능률의 작업자 8명이 15일에 끝내는 일을 12명이 시작했다. 4일 뒤 3명이 빠졌다면 남은 일을 끝내는 데 추가로 며칠이 필요한가?","8일","calculation",4)
    add(s,"math","median-replacement","medium","statistics_reasoning","정렬된 다섯 수 3, 7, x, 14, 20의 평균이 11일 때 중앙값을 구하시오.","11","calculation",3)
    add(s,"math","remainder-system","hard","modular_reasoning","50보다 작은 자연수 중 5로 나누면 2가 남고 7로 나누면 4가 남는 가장 큰 수를 구하시오.","32","calculation",4)
    add(s,"math","conditional-draw","hard","conditional_probability","두 개의 공을 동시에 뽑았더니 적어도 하나가 빨간색이었다. 주머니에 빨간 공 2개와 파란 공 2개가 있을 때 두 공 모두 빨간색일 조건부확률을 구하시오.","1/5","calculation",4)
    s = "private_eval"
    add(s,"math","concentration-replacement","hard","mixture_reasoning","20% 소금물 500 g에서 100 g을 덜어 내고 같은 양의 물을 넣었다. 최종 소금물의 농도는 얼마인가?","16%","calculation",4)
    add(s,"math","catch-up-delay","hard","relative_speed","A가 시속 4 km로 출발한 지 30분 뒤 같은 길을 B가 시속 6 km로 따라간다. B가 출발한 뒤 몇 시간 만에 A를 따라잡는가?","1시간","calculation",3)
    add(s,"math","ticket-system","medium","linear_system_context","성인 표 3장과 청소년 표 2장은 34,000원, 성인 표 1장과 청소년 표 4장은 26,000원이다. 성인 표 한 장의 가격을 구하시오.","8400원","calculation",3)
    add(s,"math","polygon-exterior","medium","polygon_angles","정다각형의 한 외각이 24°이다. 이 정다각형의 대각선 수를 구하시오.","90개","calculation",3)
    add(s,"math","two-year-growth","medium","compound_growth","어떤 양이 첫해에 10% 증가하고 다음 해에 20% 증가하여 132가 되었다. 처음 양을 구하시오.","100","calculation",3)
    add(s,"math","restricted-arrangement","hard","combinatorics","A, B, C, D 네 사람을 한 줄로 세울 때 A와 B가 이웃하지 않는 배열의 수를 구하시오.","12","calculation",4)
    add(s,"math","piecewise-inverse","hard","piecewise_function","함수 f(x)는 x<3일 때 2x+1, x≥3일 때 x²-4이다. f(x)=5를 만족하는 모든 x의 합을 구하시오.","5","calculation",4)
    add(s,"math","map-scale-route","medium","scale_reasoning","축척 1:25,000인 지도에서 등산로 길이가 7.2 cm이다. 실제 거리를 km로 환산하시오.","1.8 km","calculation",3)
    add(s,"math","vertex-distance","hard","quadratic_vertex","y=x²-6x+13의 꼭짓점과 원점 사이의 거리를 구하시오.","5","calculation",4)
    add(s,"math","complement-events","hard","multi_stage_probability","동전을 세 번 던질 때 앞면이 적어도 두 번 나올 확률을 구하시오.","1/2","calculation",3)


def coding_bank() -> None:
    s = "train"
    add(s,"coding","nested-accumulator","medium","code_trace","다음 Python 코드의 출력값을 쓰시오. `total=0\nfor i in range(1,4):\n    for j in range(i):\n        total += i+j\nprint(total)`","18","code_execution",3)
    add(s,"coding","alias-mutation","medium","reference_semantics","다음 코드의 출력값을 쓰시오. `a=[1,2]; b=a; b.append(3); a[0]=9; print(b)`","[9, 2, 3]","code_execution",3)
    add(s,"coding","local-shadowing","medium","scope_trace","다음 코드의 출력 두 값을 순서대로 쓰시오. `x=5\ndef f(x):\n    x += 2\n    return x\nprint(f(x), x)`","7 5","code_execution",2)
    add(s,"coding","recursive-skip","hard","recursion_trace","함수 `f(n)`이 n≤1이면 1, 그 외에는 `f(n-1)+f(n-2)`를 반환한다. `f(5)`의 값을 구하시오.","8","code_execution",4)
    add(s,"coding","binary-search-path","medium","algorithm_trace","정렬된 배열 `[2,5,8,12,16,23,38]`에서 표준 이진 탐색으로 16을 찾는다. 확인하는 원소를 순서대로 쓰시오.","12, 23, 16","programmatic_constraints",3)
    add(s,"coding","bubble-first-pass","medium","sorting_trace","오름차순 버블 정렬을 `[5,1,4,2]`에 적용한다. 왼쪽부터 인접 쌍을 한 번 끝까지 비교한 직후 배열을 쓰시오.","[1, 4, 2, 5]","code_execution",3)
    add(s,"coding","frequency-dictionary","medium","data_structure_trace","문자열 `banana`의 문자를 왼쪽부터 세어 딕셔너리에 저장한다. Python 딕셔너리 표기에서 삽입 순서를 유지해 결과를 쓰시오.","{'b': 1, 'a': 3, 'n': 2}","code_execution",2)
    add(s,"coding","slice-reversal","medium","string_trace","다음 Python 표현식의 결과를 쓰시오. `'engineering'[1:9:2][::-1]`","rein","code_execution",3)
    add(s,"coding","try-finally-flow","hard","exception_trace","다음 코드의 출력 순서를 공백으로 이어 쓰시오. `try:\n print('A'); 1/0\nexcept ZeroDivisionError:\n print('B')\nfinally:\n print('C')`","A B C","code_execution",3)
    add(s,"coding","off-by-one-sum","medium","loop_boundary_repair","1부터 n까지 합하려고 `for i in range(1,n): total += i`를 작성했다. n도 포함하도록 바꾼 반복 범위를 정확히 쓰시오.","range(1, n+1)","executable_tests",2)
    add(s,"coding","short-circuit-side-effect","hard","boolean_trace","다음 코드의 출력값을 쓰시오. `calls=[]\ndef g(): calls.append(1); return True\nresult = False and g()\nprint(len(calls), result)`","0 False","code_execution",3)
    add(s,"coding","set-combination","medium","set_operations","Python 집합 A={1,2,3,4}, B={3,4,5}일 때 `(A-B) | (A&B)`를 오름차순 리스트로 바꾼 결과를 쓰시오.","[1, 2, 3, 4]","code_execution",3)
    add(s,"coding","queue-events","medium","queue_simulation","빈 큐에 4, 7을 차례로 넣고 하나를 꺼낸 뒤 9를 넣는다. 그 다음 꺼내지는 값과 마지막 큐 상태를 쓰시오.","7, [9]","programmatic_constraints",3)
    add(s,"coding","stack-brackets","hard","stack_validation","문자열 `([{}])(()`를 왼쪽부터 스택으로 검사할 때 최종적으로 올바른 괄호열인지 쓰고, 아니라면 남은 여는 괄호 수를 쓰시오.","올바르지 않음, 1개","programmatic_constraints",4)
    add(s,"coding","two-pointer-pairs","hard","algorithm_trace","정렬 배열 `[1,2,3,4,6,8]`에서 양끝 포인터로 합이 9인 서로 다른 쌍을 찾는다. 발견되는 쌍을 순서대로 쓰시오.","(1, 8), (3, 6)","programmatic_constraints",4)
    add(s,"coding","shallow-copy","medium","copy_semantics","다음 코드의 출력값을 쓰시오. `a=[[1],[2]]; b=a.copy(); b[0].append(9); b.append([3]); print(a)`","[[1, 9], [2]]","code_execution",3)
    add(s,"coding","mutable-default","hard","function_state","다음 코드의 두 출력값을 순서대로 쓰시오. `def add(x,L=[]): L.append(x); return L\nprint(add(1)); print(add(2))`","[1] / [1, 2]","code_execution",3)
    add(s,"coding","divmod-negative","hard","integer_arithmetic","Python에서 `-17 // 5`와 `-17 % 5`의 값을 순서대로 쓰시오.","-4, 3","code_execution",3)
    add(s,"coding","zip-filter","medium","iteration_trace","다음 코드의 출력값을 쓰시오. `a=[2,4,6]; b=[1,3,5]; print([x-y for x,y in zip(a,b) if x+y>5])`","[1, 1]","code_execution",3)
    add(s,"coding","nested-comprehension","hard","comprehension_trace","다음 표현식의 결과를 쓰시오. `[(i,j) for i in range(3) for j in range(i) if (i+j)%2==0]`","[(2, 0)]","code_execution",4)
    add(s,"coding","triangular-complexity","medium","complexity_analysis","바깥 반복이 n번 실행되고, i번째 바깥 반복에서 안쪽 반복이 i번 실행된다. 전체 시간 복잡도를 Big-O로 쓰시오.","O(n²)","exact_match",2)
    add(s,"coding","remove-while-iterating","hard","mutation_bug_repair","리스트를 순회하며 짝수를 `remove`하는 코드는 연속한 짝수를 건너뛸 수 있다. 원본 순서를 유지하며 홀수만 남기는 한 줄짜리 리스트 컴프리헨션을 쓰시오.","[x for x in values if x % 2 != 0]","executable_tests",3)
    add(s,"coding","boundary-test","medium","test_design","함수 `is_adult(age)`가 age≥18일 때 참을 반환한다. 경계 오류를 확인하기 위한 최소 두 입력값을 오름차순으로 쓰시오.","17, 18","programmatic_constraints",2)
    add(s,"coding","state-machine","hard","pseudocode_trace","상태가 0에서 시작한다. 입력 문자가 A이면 상태에 2를 더하고, B이면 상태를 2배 하며, C이면 1을 뺀다. 입력 `AABCB` 처리 후 상태는 얼마인가?","14","code_execution",4)
    s = "validation"
    add(s,"coding","digit-recursion","medium","recursion_trace","함수 `s(n)`이 n<10이면 n, 아니면 `n%10+s(n//10)`을 반환한다. `s(5072)`의 결과를 구하시오.","14","code_execution",3)
    add(s,"coding","stable-record-sort","medium","sorting_semantics","레코드 `[('A',2),('B',1),('C',2)]`를 두 번째 값만 기준으로 안정 정렬했을 때 순서를 쓰시오.","[('B', 1), ('A', 2), ('C', 2)]","code_execution",2)
    add(s,"coding","memo-call-count","hard","memoization_trace","`fib(0)=0`, `fib(1)=1`이고 메모이제이션된 `fib(5)`를 빈 캐시에서 호출한다. 서로 다른 n에 대해 실제 계산 본문이 실행되는 횟수는 몇 번인가?","6번","programmatic_constraints",4)
    add(s,"coding","validation-branch","hard","method_call_bug","정수 문자열만 허용해야 하는 함수가 `if text.isdigit: return int(text)`라고 썼다. 잘못된 부분과 수정식을 쓰시오.","메서드를 호출하지 않았다; if text.isdigit():","executable_tests",3)
    add(s,"coding","bfs-order","hard","graph_traversal","무방향 그래프의 인접 목록이 A:[B,C], B:[A,D], C:[A,D], D:[B,C]이다. 이웃을 알파벳순으로 큐에 넣는 BFS를 A에서 시작할 때 방문 순서를 쓰시오.","A, B, C, D","programmatic_constraints",4)
    s = "private_eval"
    add(s,"coding","closure-binding","hard","closure_trace","다음 코드의 출력값을 쓰시오. `fs=[]\nfor i in range(3): fs.append(lambda: i)\nprint([f() for f in fs])`","[2, 2, 2]","code_execution",4)
    add(s,"coding","generator-exhaustion","medium","iterator_semantics","다음 코드의 출력값을 쓰시오. `g=(x*x for x in range(4)); a=sum(g); b=list(g); print(a,b)`","14 []","code_execution",3)
    add(s,"coding","matrix-rotation","hard","array_transformation","행렬 `[[1,2,3],[4,5,6]]`을 시계 방향으로 90° 회전한 결과를 중첩 리스트로 쓰시오.","[[4, 1], [5, 2], [6, 3]]","programmatic_constraints",4)
    add(s,"coding","merge-intervals","hard","algorithm_trace","구간을 시작점 기준으로 정렬한 `[(1,3),(2,6),(8,10),(9,12)]`를 겹치는 구간끼리 합친 결과를 쓰시오.","[(1, 6), (8, 12)]","programmatic_constraints",4)
    add(s,"coding","binary-loop-bug","hard","representation_bug","양의 정수를 이진수로 바꾸는 반복문이 `while n>0: bits += str(n%2); n//=2`만 수행한다. n=6에서 만들어지는 문자열과 올바른 결과를 얻기 위한 마지막 연산을 쓰시오.","011, 문자열 뒤집기","executable_tests",4)
    add(s,"coding","inheritance-super","hard","object_trace","다음 코드의 출력값을 쓰시오. `class A:\n def f(self): return 'A'\nclass B(A):\n def f(self): return super().f()+'B'\nprint(B().f())`","AB","code_execution",3)
    add(s,"coding","coin-dp","hard","dynamic_programming","동전 1원과 3원만 무제한 사용할 때 7원을 만드는 최소 동전 수를 구하시오.","3개","programmatic_constraints",4)
    add(s,"coding","circular-buffer","hard","queue_simulation","크기 4인 원형 큐가 비어 있다. 10,20,30을 넣고 두 개를 꺼낸 뒤 40,50,60을 차례로 넣으려 한다. 마지막 삽입의 성공 여부와 앞에서부터 저장된 값을 쓰시오.","성공, 30, 40, 50, 60","programmatic_constraints",4)
    add(s,"coding","token-parser","medium","string_parsing","문자열 `temp=21;mode=eco;fan=3`을 세미콜론과 등호로 나누어 값만 원래 순서대로 쓰시오.","21, eco, 3","programmatic_constraints",3)
    add(s,"coding","loop-invariant","hard","initialization_bug","최댓값을 찾는 코드가 `best=0`으로 시작해 음수만 있는 리스트에서 실패한다. 리스트가 비어 있지 않다는 조건에서 안전한 초기화 식을 쓰시오.","best = values[0]","executable_tests",3)


def science_bank() -> None:
    s = "train"
    add(s,"science","series-parallel-current","hard","circuit_analysis","6 Ω과 3 Ω 저항을 병렬로 연결하고 그 묶음을 4 Ω 저항과 직렬로 연결했다. 전체에 12 V를 걸었을 때 전원에서 흐르는 전류를 구하시오.","2 A","deterministic_calculation",4)
    add(s,"science","accelerate-cruise","hard","motion_analysis","정지한 물체가 2 m/s²으로 5초간 가속한 뒤 그 속력으로 3초간 이동했다. 전체 이동 거리를 구하시오.","55 m","deterministic_calculation",4)
    add(s,"science","machine-efficiency","medium","energy_efficiency","장치에 500 J을 공급했더니 60%가 유용한 에너지로 전환되었다. 유용한 에너지와 손실 에너지를 각각 구하시오.","300 J, 200 J","deterministic_calculation",2)
    add(s,"science","floating-fraction","hard","density_buoyancy","질량 540 g, 부피 600 cm³인 균일한 물체를 밀도 1.0 g/cm³인 물에 띄웠다. 물체 부피의 몇 %가 잠기는가?","90%","deterministic_calculation",3)
    add(s,"science","equal-mass-mixing","medium","thermal_equilibrium","같은 물질 200 g씩을 각각 80°C와 20°C에서 섞었다. 열손실이 없고 상태 변화가 없다면 평형 온도는 얼마인가?","50°C","deterministic_calculation",2)
    add(s,"science","wave-speed","medium","wave_relation","진동수가 5 Hz이고 파장이 4 m인 파동이 30초 동안 진행한 거리를 구하시오.","600 m","deterministic_calculation",3)
    add(s,"science","lever-balance","medium","lever_balance","받침점에서 왼쪽 0.4 m 지점에 30 N의 힘이 작용한다. 오른쪽 0.6 m 지점에서 평형을 이루는 힘을 구하시오.","20 N","deterministic_calculation",2)
    add(s,"science","inelastic-collision","hard","momentum_conservation","질량 2 kg인 수레가 3 m/s로 움직이다가 정지한 1 kg 수레와 충돌해 붙었다. 마찰을 무시할 때 충돌 직후 속력을 구하시오.","2 m/s","deterministic_calculation",3)
    add(s,"science","power-to-energy","medium","electric_energy","소비 전력 1,200 W인 전열기를 15분 사용했다. 소비한 전기에너지를 kWh로 구하시오.","0.3 kWh","deterministic_calculation",3)
    add(s,"science","shoe-pressure","medium","pressure_reasoning","무게가 600 N인 사람이 두 발로 서 있다. 바닥과 닿는 한쪽 신발 면적이 150 cm²라면 바닥에 가하는 압력은 얼마인가?","20000 Pa","deterministic_calculation",3)
    add(s,"science","mass-conservation","easy","chemical_conservation","밀폐 용기에서 탄소 12 g이 산소 32 g과 완전히 반응했다. 생성물의 총질량을 구하시오.","44 g","deterministic_calculation",1)
    add(s,"science","dilution-volume","medium","solution_dilution","2.0 mol/L 용액 100 mL를 0.50 mol/L로 희석하려 한다. 최종 부피와 추가할 물의 부피를 구하시오.","400 mL, 물 300 mL","deterministic_calculation",3)
    add(s,"science","reaction-mole-ratio","medium","stoichiometry","반응식 2H₂+O₂→2H₂O에서 수소가 충분할 때 산소 3 mol이 완전히 반응하면 물은 몇 mol 생성되는가?","6 mol","deterministic_calculation",2)
    add(s,"science","cooling-crystals","hard","solubility_reasoning","어떤 물질의 물 100 g당 용해도는 60°C에서 50 g, 20°C에서 20 g이다. 60°C 포화용액을 20°C로 식힐 때 석출되는 질량을 구하시오.","30 g","deterministic_calculation",3)
    add(s,"science","monohybrid-recessive","medium","genetics_probability","완전 우성인 형질에서 Aa와 Aa를 교배했다. 열성 표현형 자손이 태어날 확률을 구하시오.","1/4","deterministic_calculation",2)
    add(s,"science","food-chain-transfer","medium","ecosystem_energy","생산자가 10,000 kJ를 저장했고 영양 단계마다 10%가 다음 단계로 전달된다. 생산자→초식동물→육식동물일 때 육식동물이 얻는 에너지는 얼마인가?","100 kJ","deterministic_calculation",3)
    add(s,"science","respiration-data","medium","data_interpretation","밀폐 용기 속 발아 종자 실험에서 산소가 21%에서 18%로 줄고 이산화탄소가 0.04%에서 3.04%로 늘었다. 두 기체 변화량의 비 O₂ 감소:CO₂ 증가는 얼마인가?","1:1","deterministic_calculation",2)
    add(s,"science","osmosis-mass","medium","cell_transport","같은 크기의 감자 조각을 증류수와 진한 소금물에 넣었다. 증류수 조각의 질량은 증가하고 소금물 조각은 감소했다. 물이 순이동한 방향을 각각 쓰시오.","증류수에서 감자 안으로, 감자에서 소금물로","exact_match",2)
    add(s,"science","seismic-distance","hard","earthquake_waves","P파 속력은 8 km/s, S파 속력은 4 km/s이고 두 파의 도착 시각 차가 10초이다. 진원까지의 거리를 구하시오.","80 km","deterministic_calculation",4)
    add(s,"science","air-parcel-rise","medium","weather_reasoning","지표의 공기 온도가 24°C이고 공기 덩어리가 상승하며 1 km마다 10°C씩 낮아진다고 단순화한다. 1.5 km 높이의 온도를 구하시오.","9°C","deterministic_calculation",2)
    add(s,"science","moon-sequence","medium","astronomy_sequence","보름달이 관측된 뒤 약 7일 후와 다시 약 7일 후에 보이는 대표 달의 위상을 순서대로 쓰시오.","하현달, 삭","exact_match",3)
    add(s,"science","control-variable","medium","experimental_design","비료 양이 식물 성장에 미치는 영향을 조사하면서 화분마다 빛의 세기가 달랐다. 독립변인과 반드시 통제해야 할 변인을 각각 쓰시오.","독립변인: 비료 양, 통제변인: 빛의 세기","rubric_review",3,["비료 양을 독립변인으로 식별한다.","빛의 세기를 통제변인으로 식별한다."])
    add(s,"science","repeated-measurement","medium","measurement_uncertainty","길이를 10.2, 10.4, 10.3, 12.8 cm로 측정했다. 한 값이 명백한 이상값이라면 이를 제외한 평균을 구하시오.","10.3 cm","deterministic_calculation",3)
    add(s,"science","linear-sensor","hard","sensor_calibration","온도 센서 출력이 0°C에서 0.50 V, 100°C에서 2.50 V로 선형이다. 출력이 1.70 V일 때 온도를 구하시오.","60°C","deterministic_calculation",4)
    s = "validation"
    add(s,"science","parallel-branch-power","hard","circuit_power","6 Ω과 3 Ω 저항을 병렬 연결해 12 V 전원에 직접 연결했다. 전체 소비 전력을 구하시오.","72 W","deterministic_calculation",4)
    add(s,"science","neutralization-moles","medium","acid_base_reasoning","0.10 mol/L 염산 200 mL를 같은 농도의 수산화나트륨으로 완전히 중화하려 한다. 필요한 수산화나트륨 용액의 부피를 구하시오.","200 mL","deterministic_calculation",2)
    add(s,"science","photosynthesis-variable","medium","graph_interpretation","빛의 세기를 높일수록 광합성 속도가 증가하다가 일정해졌다. 포화 구간에서 빛 외에 제한 요인이 될 수 있는 것을 이산화탄소 농도와 빛의 세기 중 고르시오.","이산화탄소 농도","exact_match",2)
    add(s,"science","albedo-feedback","hard","climate_reasoning","눈이 녹아 어두운 지표가 드러나면 반사율이 낮아지고 흡수 에너지가 늘어난다. 이것이 추가 융해를 일으키는 되먹임의 종류를 쓰시오.","양의 되먹임","exact_match",3)
    add(s,"science","half-life-two","medium","radioactive_decay","반감기가 6시간인 물질이 160 mg 있다. 18시간 뒤 남는 양을 구하시오.","20 mg","deterministic_calculation",3)
    s = "private_eval"
    add(s,"science","reaction-braking","hard","motion_safety","자동차가 20 m/s로 달리다가 운전자가 0.75초 후 제동을 시작하고, 이후 5 m/s²의 일정한 감속도로 멈춘다. 반응 거리와 제동 거리를 합한 정지 거리를 구하시오.","55 m","deterministic_calculation",4)
    add(s,"science","metal-calorimetry","hard","calorimetry","비열 0.5 J/(g·°C)인 금속 100 g이 80°C에서 30°C로 식었다. 방출한 열량을 구하시오.","2500 J","deterministic_calculation",3)
    add(s,"science","gear-speed-torque","hard","mechanical_system","서로 맞물린 12톱니 기어가 36톱니 기어를 분당 120회전으로 구동한다. 큰 기어의 회전수와 회전 방향을 쓰시오.","분당 40회전, 반대 방향","deterministic_calculation",3)
    add(s,"science","echo-depth","medium","sound_reflection","물속 음파 속력이 1,500 m/s이고 배에서 보낸 신호가 0.8초 뒤 돌아왔다. 수심을 구하시오.","600 m","deterministic_calculation",3)
    add(s,"science","gas-expansion","medium","gas_law","압력이 일정할 때 기체의 절대온도가 300 K에서 450 K로 변했다. 처음 부피가 2.0 L라면 최종 부피는 얼마인가?","3.0 L","deterministic_calculation",2)
    add(s,"science","limiting-reactant","hard","stoichiometry","반응식 2H₂+O₂→2H₂O에서 수소 5 mol과 산소 2 mol을 반응시켰다. 제한 반응물과 생성되는 물의 양을 쓰시오.","산소, 4 mol","deterministic_calculation",4)
    add(s,"science","xlinked-probability","hard","genetics_probability","어머니가 X연관 열성 형질의 보인자이고 아버지는 정상이다. 태어날 아이가 아들이면서 해당 형질을 나타낼 확률을 구하시오.","1/4","deterministic_calculation",4)
    add(s,"science","mark-recapture","medium","population_estimation","호수에서 물고기 40마리를 표시해 놓았다. 이후 50마리를 잡았더니 표시된 개체가 10마리였다. 개체군 크기를 단순 비례로 추정하시오.","200마리","deterministic_calculation",3)
    add(s,"science","dew-condensation","medium","phase_change_reasoning","따뜻하고 습한 공기가 차가운 컵 표면에서 냉각되어 물방울이 생겼다. 물방울의 물은 어디서 왔으며 어떤 상태 변화인가?","공기 중 수증기, 응결","exact_match",2)
    add(s,"science","confounded-groups","hard","experimental_design","새 학습법 효과를 비교하면서 실험 집단은 오전에, 비교 집단은 늦은 밤에 시험을 봤다. 결론을 흐리는 혼입 변인과 최소 수정 방법을 쓰시오.","시험 시간대; 두 집단의 시험 시간대를 같게 한다","rubric_review",4,["시험 시간대를 혼입 변인으로 지적한다.","집단 간 시험 시간대를 동일하게 통제하는 수정을 제안한다."])


def logic_bank() -> None:
    s = "train"
    add(s,"logic","implication-table","medium","propositional_logic","P는 참, Q는 거짓일 때 `(P→Q)↔(¬Q→¬P)`의 진릿값을 구하시오.","참","exact_match",3)
    add(s,"logic","five-order","hard","ordering_constraints","다섯 작업 A,B,C,D,E의 순서에서 E는 D보다 앞, D는 B보다 앞, B는 A보다 앞, C는 A보다 뒤이다. 가능한 유일한 순서를 쓰시오.","E-D-B-A-C","constraint_solver",4)
    add(s,"logic","three-day-schedule","medium","scheduling","수학, 과학, 국어 발표를 월·화·수요일에 하나씩 한다. 수학은 과학보다 앞이고, 국어는 월요일이 아니며, 과학은 수요일이다. 요일별 발표를 쓰시오.","월 수학, 화 국어, 수 과학","constraint_solver",3)
    add(s,"logic","two-liars","hard","two_person_truth","A는 'B는 거짓말쟁이다'라고 하고 B는 'A와 나는 같은 유형이다'라고 했다. 각 사람은 항상 참말만 하거나 항상 거짓말만 한다. 두 사람의 유형을 쓰시오.","A 참말쟁이, B 거짓말쟁이","constraint_solver",4)
    add(s,"logic","quantifier-negation","hard","quantifier_logic","명제 '모든 학생이 수학 또는 과학 중 적어도 하나를 좋아한다'의 부정을 자연어로 정확히 쓰시오.","수학과 과학을 모두 좋아하지 않는 학생이 적어도 한 명 있다","rubric_review",3,["존재 명제로 바꾼다.","수학과 과학을 모두 좋아하지 않는다는 조건을 포함한다."])
    add(s,"logic","membership-chain","medium","set_reasoning","A⊂B, B⊂C이고 x∈A이다. 반드시 참인 소속 관계를 A를 제외하고 모두 쓰시오.","x∈B, x∈C","constraint_solver",2)
    add(s,"logic","binary-code","medium","constraint_enumeration","세 자리 이진 문자열에서 1이 정확히 두 개이고 첫 자리가 1이다. 가능한 문자열을 사전순으로 쓰시오.","101, 110","constraint_solver",3)
    add(s,"logic","dag-route-count","hard","path_counting","방향 간선이 A→B, A→C, B→C, B→D, C→D일 때 A에서 D로 가는 단순 경로 수를 구하시오.","3","constraint_solver",4)
    add(s,"logic","adjacent-seating","hard","linear_arrangement","A,B,C,D를 한 줄로 세운다. A와 B는 이웃하고, C는 맨 앞이 아니며, D는 A보다 뒤에 있다. 조건을 만족하는 배열 수를 구하시오.","4","constraint_solver",4)
    add(s,"logic","grid-path-basic","medium","combinatorial_logic","오른쪽과 위쪽으로만 움직여 (0,0)에서 (2,2)까지 가는 최단 경로 수를 구하시오.","6","constraint_solver",2)
    add(s,"logic","contrapositive-chain","hard","deduction","P이면 Q이고 Q이면 R이다. R이 거짓이라는 사실이 주어질 때 반드시 거짓인 명제를 P와 Q 중 모두 쓰시오.","P와 Q","constraint_solver",3)
    add(s,"logic","one-true-box","hard","clue_truth","상자 A에는 '보물은 B에 있다', 상자 B에는 '보물은 C에 없다', 상자 C에는 '보물은 A에 없다'라고 적혀 있다. 문장 하나만 참이고 보물은 한 상자에만 있다. 보물 위치를 구하시오.","A","constraint_solver",4)
    add(s,"logic","month-pigeonhole","easy","pigeonhole","13명이 있을 때 태어난 달이 같은 사람이 적어도 두 명 존재함을 보장할 수 있는가? 예/아니오로 답하시오.","예","exact_match",1)
    add(s,"logic","round-robin-count","medium","counting","6개 팀이 서로 한 번씩 경기하는 풀리그의 전체 경기 수를 구하시오.","15경기","calculation",2)
    add(s,"logic","balance-deduction","medium","comparison_logic","저울에서 A+B가 C와 같고, C가 D+A보다 무겁다. B와 D 중 더 무거운 것을 쓰시오.","B","constraint_solver",3)
    add(s,"logic","family-chain","medium","relation_reasoning","영희는 민수의 누나이고, 철수는 민수의 아버지이다. 철수와 영희의 관계를 철수 기준으로 쓰시오.","아버지","exact_match",2)
    add(s,"logic","calendar-offset","medium","calendar_reasoning","어떤 달의 1일이 수요일이라면 같은 달 20일은 무슨 요일인가?","월요일","calculation",2)
    add(s,"logic","digit-equation","hard","symbolic_constraints","두 자리 수 AB와 BA의 합이 121이고 A-B=3이다. A와 B를 구하시오.","A=7, B=4","constraint_solver",4)
    add(s,"logic","demorgan-choice","medium","logical_equivalence","`¬(P∧Q)`와 동치인 식을 `¬P∧¬Q`와 `¬P∨¬Q` 중 고르시오.","¬P∨¬Q","exact_match",2)
    add(s,"logic","survey-overlap","medium","venn_reasoning","30명 중 버스를 이용하는 학생 18명, 자전거를 이용하는 학생 12명, 둘 다 이용하는 학생 5명이다. 둘 다 이용하지 않는 학생 수를 구하시오.","5명","calculation",3)
    add(s,"logic","dependency-order","hard","topological_order","작업 규칙이 A가 B보다 앞, A가 C보다 앞, B와 C가 모두 D보다 앞이다. 가능한 순서 중 알파벳순으로 가장 앞선 것을 쓰시오.","A-B-C-D","constraint_solver",3)
    add(s,"logic","toggle-parity","hard","state_reasoning","전등은 처음 꺼져 있다. 스위치 A는 전등을 한 번, B는 두 번, C는 세 번 토글한다. A,B,C를 차례로 한 번씩 작동한 뒤 전등 상태를 쓰시오.","꺼짐","constraint_solver",3)
    add(s,"logic","ranking-clues","hard","ranking_constraints","달리기에서 민지는 수현보다 앞, 지우는 민지보다 뒤지만 수현보다 앞, 현우는 수현보다 뒤였다. 네 명의 순위를 쓰시오.","민지-지우-수현-현우","constraint_solver",4)
    add(s,"logic","parity-invariant","hard","invariant_reasoning","칠판에 홀수 5개가 있다. 한 번에 두 수를 지우고 그 합을 적는 일을 네 번 반복해 수 하나를 남긴다. 마지막 수는 홀수인가 짝수인가?","홀수","constraint_solver",4)
    s = "validation"
    add(s,"logic","card-rule-test","hard","conditional_testing","카드 한쪽에 모음이 있으면 반대쪽에 짝수가 있다는 규칙을 검사한다. 보이는 면이 A, D, 4, 7인 네 카드 중 반드시 뒤집어야 할 카드를 쓰시오.","A와 7","constraint_solver",4)
    add(s,"logic","island-types","hard","two_speaker_truth","섬 주민 A는 '우리 둘은 모두 거짓말쟁이다'라고 말했고, B는 'A는 거짓말쟁이다'라고 말했다. 참말쟁이와 거짓말쟁이만 있다면 A와 B의 유형을 쓰시오.","A 거짓말쟁이, B 참말쟁이","constraint_solver",3)
    add(s,"logic","room-assignment","medium","scheduling","실험실 X,Y,Z에 로봇·회로·재료 활동을 하나씩 배정한다. 로봇은 X가 아니고, 회로는 Z이며, 재료는 로봇보다 알파벳상 앞선 방이다. 배정을 쓰시오.","X 재료, Y 로봇, Z 회로","constraint_solver",3)
    add(s,"logic","euler-trail","hard","graph_reasoning","정점 A,B,C,D와 간선 AB, BC, CD, DA, AC가 있는 무방향 그래프에서 모든 간선을 정확히 한 번 지나는 경로의 시작과 끝이 될 수 있는 두 정점을 쓰시오.","A와 C","constraint_solver",4)
    add(s,"logic","lock-clues","hard","code_deduction","세 자리 암호에서 숫자는 서로 다르다. 첫째 숫자는 둘째보다 2 크고, 셋째는 첫째와 둘째의 합이며, 세 숫자의 합은 12이다. 암호를 구하시오.","426","constraint_solver",4)
    s = "private_eval"
    add(s,"logic","three-statements","hard","three_speaker_truth","A는 'B는 참말쟁이다', B는 'C는 거짓말쟁이다', C는 'A와 B는 같은 유형이다'라고 말했다. 각자 항상 참말 또는 거짓말만 할 때 유형을 쓰시오.","A 거짓말쟁이, B 거짓말쟁이, C 참말쟁이","constraint_solver",5)
    add(s,"logic","alternating-circle","hard","circular_arrangement","남학생 3명과 여학생 3명을 원탁에 성별이 번갈아 앉게 한다. 회전해 같은 배치는 하나로 볼 때 배열 수를 구하시오.","12","calculation",4)
    add(s,"logic","subset-exact","hard","set_constraints","집합 {1,2,3,4,5}의 부분집합 중 원소가 3개이고 원소의 합이 짝수인 부분집합 수를 구하시오.","6","constraint_solver",4)
    add(s,"logic","bridge-optimal","hard","optimization","다리를 건너는 데 각각 1,2,7,10분이 걸리는 네 사람이 손전등 하나로 모두 건넌다. 한 번에 두 명까지 가며 함께 갈 때 느린 사람 시간만큼 걸린다. 최소 시간을 구하시오.","17분","constraint_solver",5)
    add(s,"logic","affirming-consequent","medium","fallacy_detection","'비가 오면 운동장이 젖는다. 운동장이 젖었다. 따라서 비가 왔다.'라는 추론이 항상 타당한지 쓰고, 아니라면 오류 이름을 쓰시오.","타당하지 않음, 후건 긍정의 오류","rubric_review",3,["추론이 타당하지 않다고 판정한다.","후건 긍정 오류를 식별한다."])
    add(s,"logic","blocked-grid","hard","path_counting","오른쪽과 위쪽으로만 이동해 (0,0)에서 (3,2)까지 가되 (1,1)은 지나지 않는다. 최단 경로 수를 구하시오.","4","constraint_solver",4)
    add(s,"logic","occupation-deduction","hard","logic_grid","가은·나래·다온의 직업은 교사·의사·공학자 중 하나씩이다. 가은은 의사가 아니고, 나래는 공학자가 아니며, 교사는 다온이 아니다. 가은이 공학자라면 나머지 직업 배정을 쓰시오.","가은 공학자, 나래 교사, 다온 의사","constraint_solver",4)
    add(s,"logic","recursive-tiles","medium","sequence_inference","타일 수가 1단계 2개, 2단계 5개이며 이후 단계는 앞 단계보다 홀수가 2씩 커지도록 더한다. 3단계에는 10개, 4단계에는 17개일 때 5단계 타일 수를 구하시오.","26개","constraint_solver",3)
    add(s,"logic","pairwise-voting","hard","social_choice","세 유권자의 선호가 각각 A>B>C, B>C>A, C>A>B이다. A와 B의 일대일 대결, B와 C의 대결, C와 A의 대결 승자를 순서대로 쓰시오.","A, B, C","constraint_solver",4)


def korean_bank() -> None:
    s = "train"
    add(s,"korean","exclusive-particle","medium","particle_semantics","문장 '철수만 영희를 도왔다'가 전제하는 내용을 '철수 외의 사람'을 주어로 하여 쓰시오.","철수 외의 사람은 영희를 돕지 않았다","rubric_review",2,["보조사 '만'의 배제 의미를 반영한다.","철수 외의 사람을 주어로 표현한다."])
    add(s,"korean","subject-honorific","medium","honorifics","'할아버지가 밥을 먹었다'를 주체 높임과 특수 어휘를 사용해 자연스럽게 고치시오.","할아버지께서 진지를 드셨다.","rubric_review",2,["주격 조사 '께서'를 사용한다.","'진지'와 '드시다'를 적절히 사용한다."])
    add(s,"korean","bound-noun-spacing","easy","spacing","'나도그일을할수있다'를 맞춤법에 맞게 띄어 쓰시오.","나도 그 일을 할 수 있다.","exact_match",1)
    add(s,"korean","double-passive","medium","grammar_correction","'바람에 문이 열려졌다'에서 불필요한 이중 피동을 고쳐 문장 전체를 쓰시오.","바람에 문이 열렸다.","exact_match",2)
    add(s,"korean","contrast-connector","easy","discourse_relation","'비가 많이 왔지만 경기는 예정대로 열렸다'에서 연결 표현이 나타내는 의미 관계를 쓰시오.","대조","exact_match",1)
    add(s,"korean","pronoun-reference","medium","reference_resolution","'수아는 민지에게 우산을 건넸다. 그는 곧 버스를 탔다.'에서 '그'가 누구인지 문맥만으로 확정할 수 있는지 쓰시오.","확정할 수 없다","exact_match",2)
    add(s,"korean","ironic-context","medium","pragmatics","친구가 약속에 한 시간 늦자 '참 일찍도 왔다'라고 말했다. 이 말의 실제 의도를 쓰시오.","매우 늦었다는 비판","rubric_review",2,["표면 의미와 반대되는 반어임을 파악한다.","늦은 도착에 대한 비판을 포함한다."])
    add(s,"korean","two-sentence-summary","hard","summarization","'학교는 빈 교실의 조명을 자동으로 끄는 센서를 설치했다. 전력 사용량이 줄었지만 센서 오작동 점검이 새 과제로 남았다.'를 장점과 한계가 모두 드러나는 한 문장으로 요약하시오.","자동 조명 센서는 전력 사용을 줄였지만 오작동 점검이라는 과제를 남겼다.","rubric_review",3,["전력 사용 감소를 포함한다.","오작동 점검 과제를 포함한다.","한 문장으로 작성한다."])
    add(s,"korean","causal-order","medium","sentence_ordering","다음 세 문장을 원인에서 결과로 자연스럽게 배열하시오. (가) 도로가 얼었다. (나) 밤새 기온이 영하로 내려갔다. (다) 차량들이 속도를 줄였다.","나-가-다","exact_match",2)
    add(s,"korean","prefix-analysis","easy","morphology","'풋사과'를 어근과 접두사로 나누어 쓰시오.","접두사 풋-, 어근 사과","exact_match",1)
    add(s,"korean","pronunciation-rule","medium","phonology","표준 발음법에 따라 '꽃잎'의 발음을 대괄호 안에 쓰시오.","[꼰닙]","exact_match",2)
    add(s,"korean","active-conversion","medium","voice_conversion","피동문 '자료가 연구팀에 의해 분석되었다'를 자연스러운 능동문으로 바꾸시오.","연구팀이 자료를 분석했다.","rubric_review",2,["연구팀을 주어로 둔다.","자료를 목적어로 두고 능동 서술어를 사용한다."])
    add(s,"korean","attachment-ambiguity","hard","syntactic_ambiguity","'나는 망원경으로 새를 본 친구를 만났다'에서 '망원경으로'가 수식할 수 있는 두 행위를 쓰시오.","새를 보는 행위, 친구를 만나는 행위","rubric_review",3,["새를 보는 행위를 제시한다.","친구를 만나는 행위를 제시한다."])
    add(s,"korean","quotation-punctuation","easy","punctuation","직접 인용이 드러나도록 다음 문장에 문장 부호를 넣으시오: 민지가 내일 출발하자라고 말했다","민지가 “내일 출발하자.”라고 말했다.","exact_match",1)
    add(s,"korean","ellipsis-subject","medium","ellipsis_resolution","'민수는 도서관에 갔다. 그리고 책을 빌렸다.'에서 둘째 문장의 생략된 주어를 쓰시오.","민수","exact_match",2)
    add(s,"korean","context-synonym","easy","lexical_choice","'새 규칙은 다음 달부터 시행된다'에서 '시행된다'와 가장 가까운 뜻을 '실행된다'와 '취소된다' 중 고르시오.","실행된다","exact_match",1)
    add(s,"korean","proverb-application","medium","proverb_meaning","작은 실수를 방치했다가 큰 사고로 이어진 상황에 가장 알맞은 속담을 '호미로 막을 것을 가래로 막는다'와 '백지장도 맞들면 낫다' 중 고르시오.","호미로 막을 것을 가래로 막는다","exact_match",2)
    add(s,"korean","causal-connector-choice","medium","connector_selection","'기온이 급격히 떨어졌다. ___ 도로에 살얼음이 생겼다.'의 빈칸에 '따라서'와 '그러나' 중 알맞은 말을 쓰시오.","따라서","exact_match",2)
    add(s,"korean","formal-rewrite","medium","register_conversion","구어적 표현 '이 장치는 전기를 엄청 아껴 준다'를 보고서에 적합한 문장으로 고치시오.","이 장치는 전력 소비를 크게 줄인다.","rubric_review",2,["구어적 표현을 제거한다.","전력 소비 감소의 의미를 유지한다.","객관적인 문체를 사용한다."])
    add(s,"korean","syllable-count","easy","phonological_count","'인공지능'을 한글 음절 단위로 셀 때 몇 음절인지 쓰시오.","4음절","exact_match",1)
    add(s,"korean","adnominal-clause","medium","grammar_analysis","'어제 읽은 책이 재미있었다'에서 관형절을 쓰시오.","어제 읽은","exact_match",2)
    add(s,"korean","indirect-quotation","medium","quotation_conversion","민지가 '나는 내일 출발해'라고 말했다. 이를 간접 인용문으로 바꾸시오.","민지는 자신이 다음 날 출발한다고 말했다.","rubric_review",3,["화자 관점에 맞게 '자신'을 사용한다.","시간 표현을 '다음 날'로 조정한다.","간접 인용 종결을 사용한다."])
    add(s,"korean","past-perfect-sequence","medium","tense_reasoning","'버스가 도착했을 때 나는 이미 정류장을 떠났다'에서 먼저 일어난 사건을 쓰시오.","내가 정류장을 떠난 일","exact_match",2)
    add(s,"korean","paragraph-topic","medium","reading_comprehension","'도시의 나무는 그늘을 만들고 미세먼지를 줄인다. 또한 빗물을 머금어 침수를 완화한다.'의 중심 내용을 쓰시오.","도시 나무의 환경적 이점","rubric_review",2,["도시 나무가 주제임을 밝힌다.","여러 환경적 이점을 포괄한다."])
    s = "validation"
    add(s,"korean","evidence-inference","hard","reading_inference","'실험 전에는 세 반의 평균이 비슷했다. 새 수업을 적용한 반만 이후 평균이 올랐지만, 학생별 공부 시간은 조사하지 않았다.'에서 성과 향상의 원인을 새 수업으로 단정할 수 없는 이유를 쓰시오.","공부 시간이라는 혼입 변인을 통제하지 않았기 때문이다.","rubric_review",3,["학생별 공부 시간을 혼입 변인으로 지적한다.","인과 단정이 어렵다고 설명한다."])
    add(s,"korean","parallel-predicate","medium","grammar_correction","'이 기계의 장점은 소음이 적고 에너지를 절약한다'의 호응이 자연스럽도록 고치시오.","이 기계의 장점은 소음이 적고 에너지가 절약된다는 것이다.","rubric_review",3,["두 장점을 명사절 또는 동일한 문장 구조로 병렬화한다.","원래 의미를 유지한다."])
    add(s,"korean","polysemy-context","medium","lexical_semantics","'회의에서 그 제안이 받아들여졌다'의 '받아들여지다'는 '수용되다'와 '물리적으로 안에 들어가다' 중 어느 뜻인가?","수용되다","exact_match",2)
    add(s,"korean","cohesive-order","hard","sentence_ordering","다음 문장을 자연스러운 글의 순서로 배열하시오. (가) 이 자료를 바탕으로 원인을 분석했다. (나) 먼저 일주일간 온도를 측정했다. (다) 분석 결과 환기 시간이 핵심 변수로 나타났다.","나-가-다","exact_match",3)
    s = "private_eval"
    add(s,"korean","counterfactual-meaning","hard","counterfactual","'조금만 일찍 출발했더라면 기차를 놓치지 않았을 텐데'가 암시하는 실제 상황 두 가지를 쓰시오.","늦게 출발했고 기차를 놓쳤다","rubric_review",3,["실제로 늦게 출발했음을 포함한다.","실제로 기차를 놓쳤음을 포함한다."])
    add(s,"korean","multiple-honorific","hard","honorifics","'제가 선생님에게 내일 보고서를 줄게요'를 주어의 겸양과 상대 높임에 맞는 공식적인 문장으로 고치시오.","제가 선생님께 내일 보고서를 드리겠습니다.","rubric_review",3,["조사 '께'를 사용한다.","겸양 동사 '드리다'를 사용한다.","공식적인 종결 표현을 사용한다."])
    add(s,"korean","spacing-meaning","medium","spacing_semantics","'아는대로 말하다'에서 의존 명사 띄어쓰기를 바로잡고 문장 전체를 쓰시오.","아는 대로 말하다","exact_match",2)
    add(s,"korean","argument-assumption","hard","argument_analysis","'온라인 설문에서 응답자의 80%가 교복 변경에 찬성했으므로 전교생의 80%가 찬성한다'는 결론이 성립하려면 필요한 표본 관련 전제를 쓰시오.","설문 응답자가 전교생을 대표하는 표본이어야 한다.","rubric_review",3,["응답 표본의 대표성을 핵심 전제로 제시한다.","전교생에 대한 일반화 조건을 설명한다."])
    add(s,"korean","balanced-summary","hard","summarization","'자전거 도로 확장은 통학 안전을 높일 수 있다. 그러나 차로 축소에 따른 교통 정체 가능성과 관리 비용도 검토해야 한다.'를 찬반 요소가 모두 포함되게 한 문장으로 요약하시오.","자전거 도로 확장은 통학 안전을 높이지만 교통 정체와 관리 비용도 고려해야 한다.","rubric_review",3,["통학 안전 향상을 포함한다.","교통 정체 가능성을 포함한다.","관리 비용을 포함한다.","한 문장으로 작성한다."])
    add(s,"korean","sentence-insertion","hard","cohesion","글 '센서는 빛의 세기를 측정한다. (  ) 측정값이 기준보다 낮으면 조명을 켠다.'의 빈칸에 두 문장을 자연스럽게 잇는 문장을 쓰시오.","제어 장치는 이 측정값을 미리 설정한 기준과 비교한다.","rubric_review",3,["측정값과 기준의 비교를 포함한다.","앞뒤 문장의 센서 측정과 조명 제어를 연결한다."])
    add(s,"korean","idiom-context","medium","idiom_meaning","팀원들이 서로 책임을 미루는 상황에서 '발을 빼다'의 의미를 문맥에 맞게 쓰시오.","관계된 일에서 책임을 피하며 빠지다","rubric_review",2,["일에서 빠져나온다는 의미를 포함한다.","책임 회피의 맥락을 반영한다."])
    add(s,"korean","reference-chain","medium","reference_resolution","'지현은 서윤에게 민호가 만든 모형을 보여 주었다. 서윤은 그것을 사진으로 남겼다.'에서 '그것'이 가리키는 대상을 쓰시오.","민호가 만든 모형","exact_match",2)
    add(s,"korean","constrained-rewrite","hard","instruction_following","'배터리 수명이 짧지만 충전 속도는 빠르다'의 의미를 유지하면서 '배터리'를 한 번만 사용하고 15자 이상 30자 이하의 한 문장으로 다시 쓰시오.","배터리 수명은 짧지만 충전 속도는 빠르다.","programmatic_constraints",3)


def build(root: Path, force: bool = False) -> dict:
    target = root / "data" / "questions" / "core_v2"
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(f"Core-v2 already exists: {target}")
    ROWS.clear()
    math_bank(); coding_bank(); science_bank(); logic_bank(); korean_bank()
    split_counts = Counter(row["split"] for row in ROWS)
    if dict(split_counts) != EXPECTED:
        raise AssertionError(f"split count mismatch: {dict(split_counts)}")
    expected_domains = {
        "math": 39, "coding": 39, "science": 39, "logic": 38, "korean": 37,
    }
    domain_counts = Counter(row["domain"] for row in ROWS)
    if dict(domain_counts) != expected_domains:
        raise AssertionError(f"domain count mismatch: {dict(domain_counts)}")
    ids = [row["question_id"] for row in ROWS]
    families = [row["template_family"] for row in ROWS]
    if len(ids) != len(set(ids)) or len(families) != len(set(families)):
        raise AssertionError("question_id or template_family is not unique")

    files = {
        "train": "train_questions.jsonl",
        "validation": "validation_questions.jsonl",
        "private_eval": "private_eval_questions.jsonl",
    }
    private_keys: list[dict] = []
    for split, filename in files.items():
        output_rows = []
        for source in (row for row in ROWS if row["split"] == split):
            row = {key: value for key, value in source.items() if not key.startswith("_")}
            answer, criteria = source["_answer"], source["_criteria"]
            if split == "private_eval":
                key_id = f"CORE2-PRIVATE-{_sha_text(row['question_id'])[:20]}"
                row["reference"] = {"visibility": "private", "key_id": key_id, "answer_sha256": _sha_text(answer)}
                private_keys.append({
                    "key_id": key_id, "question_id": row["question_id"], "answer": answer,
                    "criteria": criteria, "verification_method": row["verification_method"],
                })
            else:
                row["reference"] = {"visibility": "verification_only", "answer": answer, "answer_sha256": _sha_text(answer)}
                row["rubric"]["criteria"] = criteria
            output_rows.append(row)
        _write_jsonl(target / filename, output_rows)

    private_key_path = root / "artifacts" / "private" / "core_v2_private_key.jsonl"
    _write_jsonl(private_key_path, private_keys)
    file_hashes = {filename: hashlib.sha256((target / filename).read_bytes()).hexdigest() for filename in files.values()}
    checksum_payload = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        "dataset_name": "CoMPaSS-AI Core",
        "dataset_version": "core-v2",
        "status": "draft_pending_quality_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "question_count": len(ROWS),
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in ROWS)),
        "auto_verifiable_count": sum(row["rubric"]["auto_verifiable"] for row in ROWS),
        "file_sha256": file_hashes,
        "dataset_checksum": hashlib.sha256(checksum_payload).hexdigest(),
        "private_key_sha256": hashlib.sha256(private_key_path.read_bytes()).hexdigest(),
        "private_key_location": "artifacts/private (gitignored; transfer separately)",
        "quality_audit": None,
    }
    (target / "core_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.force), ensure_ascii=False, indent=2))
