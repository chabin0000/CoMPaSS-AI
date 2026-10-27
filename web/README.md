# ARGOS 연구 소개 사이트

CoMPaSS ARGOS 연구의 동결 확인평가와 별도 후속 개발평가를 공개 자료에 근거해 소개하는 정적 웹사이트입니다. 최종 280문항은 미실행·미측정입니다. 실시간 모델 추론 기능은 포함하지 않습니다.

## 실행

Node.js 24 이상에서 `npm ci`, `npm run build`, `npm run preview`를 실행합니다. `npm test`는 공개 자료 해시, 집계, 실제 응답과 스크롤 타임라인을 검증합니다. 빌드는 이 폴더에 포함된 공개 데이터로 수행됩니다.

`npm run data`는 원본 연구 자료가 로컬 저장소의 `reports/argos-final-20261027` 및 `reports/argos-improvement-20261027`에 있을 때만 사용합니다. 원본을 다시 읽어 공개 사본과 정규화 데이터를 갱신합니다. 해당 원본은 기존 작업 공간의 자료이며 모든 파일이 기준 Git 커밋에 포함되어 있다는 뜻은 아닙니다. 공개 사본의 원본 경로와 SHA-256은 `public/sources/manifest.json`에 기록합니다.

첫 화면의 `final-report.pdf`는 기존 ARGOS 최종 연구보고서이며, `improvement-report.pdf`는 별도의 후속 개선 보고서입니다. 둘 다 `public/sources/`에서 확인할 수 있습니다.

## Hosting

`firebase.json`은 Firebase Hosting의 정적 `dist`만 배포하도록 설정되어 있습니다. 프로젝트 및 Hosting 사이트를 명확히 지정한 다음 Hosting target `argos`를 연결하고 미리보기 채널에서 확인한 후 배포합니다. 데이터베이스, 인증, 서버 함수, 유료 업그레이드는 사용하지 않습니다.
