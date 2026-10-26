# CoMPaSS ARGOS 최종 연구 보고서

실제 본실험, 수학 저하 진단과 후속 학습, 모델 동결 뒤 신규 평가를 연결한 최종 보고서입니다. 초기 계획서와 진행 중 확정한 방법을 구분하고, 성능 향상이 확인되지 않은 결과도 포함합니다.

- [전체 보고서 PDF](CoMPaSS_ARGOS_final_report_20261027.pdf)
- [GitHub에서 읽는 전체 보고서](REPORT_KO.md)
- [브라우저용 보고서](report.html)
- [집계 데이터](data/)
- [증거 파일과 해시 목록](evidence_manifest.json)

신규 평가 범위는 비공개 240문항 및 KLUE NLI, KoBEST BoolQ, GSM8K 각 40문항입니다. 두 동결 모델의 실제 720응답을 사용합니다. 검수자는 단일 AI이며 사람 검수자 2명, 독립 평가자 일치도, 공식 전체 벤치마크 점수를 수행했다고 주장하지 않습니다.

수학 80문항은 S0 68개, ARGOS 37개 정답으로 회복을 확인하지 못했습니다. 제공 자료의 요청 사실은 각각 53/60, 60/60문항에서 충족했습니다. 정확 출력 40문항은 응답 전체의 일치 여부를 측정하므로 순수한 의미 정확도와 구분해야 합니다. 단일 AI의 쌍대평가 전체 40쌍에서는 S0 20승, ARGOS 6승, 동률 14개였습니다.

CSV의 `semantic` 열은 기존 코드의 이름을 유지합니다. `instruction` 코호트에서는 전체 응답 정확 일치, 나머지 코호트에서는 내용 정답을 뜻합니다. 세부 형식 검사기는 생성 후 모델명을 해제하기 전에 확정했으며 외부 사전 등록으로 표현하지 않습니다.

비공개 질문과 정답, 원시 모델 가중치, 서비스 자격증명은 공개하지 않습니다. `data/per_item_scores.csv`는 문항 ID와 판정 및 실행 계측만 포함합니다. 공개 원자료의 리비전과 표본 인덱스는 `data/benchmark_sources.json`에 있습니다.

## 보고서 다시 만들기

저장소 루트에서 필요한 문서 라이브러리를 설치하고 다음 명령을 실행합니다.

```powershell
python -m pip install -r requirements/argos-confirmatory-report.txt
python scripts/argos_build_final_report.py --root reports/argos-final-20261027
```

Windows 기본 글꼴은 맑은 고딕입니다. 다른 운영체제에서는 `--font`와 `--bold-font`에 한국어 TTF 경로를 지정합니다. 이 명령은 공개 집계표로 PDF, HTML, Markdown과 그림을 다시 만들며 새로운 모델 추론이나 학습을 하지 않습니다.

## 평가 코드

`scripts/argos_confirmatory_inference.py`는 해시로 고정된 질문 파일만 읽는 NF4 추론 실행기입니다. 별도로 보관한 봉인 디렉터리와 허용된 모델 가중치가 필요합니다. 이 저장소 공개 자료만으로 비공개 평가 원문을 복원하거나 전체 추론을 재실행할 수 있다고 주장하지 않습니다.

`scripts/argos_confirmatory_analysis.py`는 자동 숫자 및 라벨 채점, 실제 AI 판정 입력, 주장 단위 집계, Wilson 구간과 대응 부트스트랩을 구현합니다. `tests/test_argos_confirmatory_analysis.py`는 충돌하는 최종 답, 분수 표기, 미해결과 분모 0 처리 등을 검사합니다.

```powershell
python -m unittest discover -s tests -p test_argos_confirmatory_analysis.py -v
```

모델과 데이터의 사용 조건은 각각 원 배포자의 조건을 따릅니다. 이 보고서는 모델 가중치나 별도의 사용권을 재배포하지 않습니다.
