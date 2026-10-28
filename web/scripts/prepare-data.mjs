import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

const web=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const root=resolve(web,'..');
const improvement='reports/argos-improvement-20261027';
const previous='reports/argos-final-20261027';
const pub=`${improvement}/public-data`;
const sourceCommit=execFileSync('git',['rev-parse',process.env.ARGOS_REPORT_SOURCE_COMMIT??'HEAD'],{cwd:root,encoding:'utf8'}).trim();
const read=async path=>JSON.parse(await readFile(resolve(root,path),'utf8'));
const jsonl=async path=>(await readFile(resolve(root,path),'utf8')).trim().split(/\r?\n/).map(x=>JSON.parse(x));
const [frozen,recovery,report,runtime,seal,protocol]=await Promise.all([
  read(`${previous}/data/metrics.json`),read(`${pub}/recovery_selection.json`),read(`${pub}/report_data.json`),
  read(`${pub}/runtime_identity.json`),read(`${pub}/recovery_seal.json`),read(`${pub}/recovery_protocol.json`)
]);
const csv=(await readFile(resolve(root,`${pub}/development_metrics.csv`),'utf8')).trim().split(/\r?\n/).map(row=>row.split(','));
const rows=csv.slice(1).map(row=>Object.fromEntries(csv[0].map((key,i)=>[key,row[i]])));
const definitions={
  source_claims:{label:'제공 자료의 요청 사실 충족',short:'자료 충실도',definition:'문항에 주어진 가상 기록을 바탕으로 요청한 사실을 충족한 응답 수입니다. 일반적인 세계 지식의 정확도를 뜻하지 않습니다.',interpretation:'제공 자료에서 요청한 사실을 충족한 응답은 늘었습니다. 이 과제의 관측을 일반적인 사실성 향상으로 확대할 수는 없습니다.'},
  instruction:{label:'공백 정규화 후 전체 응답 일치',short:'정확 출력',definition:'양끝 공백과 내부 일반 공백(U+0020)을 제거한 뒤 전체 응답을 정답과 비교합니다. 내부 개행과 문장부호는 유지합니다. 내용과 형식이 결합된 지표입니다.',interpretation:'공백을 정규화한 전체 응답의 일치가 늘었습니다. 이 결과를 바이트 단위의 정확 일치나 순수한 형식 능력 점수로 해석하지 않습니다.'},
  math:{label:'수학 내용 정답',short:'수학',definition:'신규 수학 80문항에서 최종 정수·분수 값이 맞은 응답 수입니다. 모든 중간 풀이가 타당하다는 보증은 아닙니다. 미확정은 전체 분모에 남깁니다.',interpretation:'새 수학 표본에서 큰 저하가 관측됐습니다. S0와 ARGOS의 미확정 응답은 각각 1개이며, 정답으로 인정하지 않았습니다.'},
  premise:{label:'전제 판단의 결론 정답',short:'전제 판단',definition:'제공된 가상 기록에서 질문의 전제를 지지하는지에 대한 첫 줄 결론을 채점합니다. 설명의 타당성과는 별개입니다.',interpretation:'전제 판단의 결론 정답도 줄었습니다. 결론이 맞더라도 설명 요구를 충족했다고 볼 수 없습니다.'},
  gsm8k:{label:'GSM8K 표본 최종 정답',short:'GSM8K',definition:'GSM8K main test에서 뽑은 영어 40문항의 최종 정답입니다. 전체 벤치마크 점수가 아닙니다.',interpretation:'공개 수학 표본에서도 큰 저하가 나타났습니다. 이 확인평가의 S0 37/40과 후속 개발평가의 S0 35/40은 실행 조건과 평가 목적이 다릅니다.'},
  klue_nli:{label:'KLUE NLI 표본 정답',short:'KLUE NLI',definition:'KLUE NLI validation의 고정 40문항 표본입니다. 공식 전체 벤치마크 점수가 아닙니다.',interpretation:'이 작은 자연어 추론 표본에서는 정답이 늘었습니다. 표본의 크기와 사전학습 오염 가능성을 함께 고려해야 합니다.'},
  kobest_boolq:{label:'KoBEST BoolQ 표본 정답',short:'KoBEST',definition:'KoBEST BoolQ test의 고정 40문항 표본입니다. 공식 전체 벤치마크 점수가 아닙니다.',interpretation:'정답 1개의 차이가 관측됐습니다. 이 차이만으로 일반적인 한국어 능력 향상을 주장할 수 없습니다.'}
};
const confirmation=Object.entries(definitions).map(([id,def])=>({id,...def,results:['S0','ARGOS'].map(model=>{
  const row=frozen.find(r=>r.cohort===id&&r.model===model);assert.ok(row,`Missing ${id}/${model}`);
  assert.ok(row.semantic_correct<=row.n);assert.ok(Math.abs(row.semantic_rate-row.semantic_correct/row.n)<1e-10);
  return {model,correct:row.semantic_correct,n:row.n,rate:row.semantic_correct/row.n,unknown:row.unknown,complete:row.complete};
})}));
const devMetrics=[
  {id:'document',label:'문서 정답',n:96,definition:'제공된 합성 문서 96개에서 요청한 최종 정답이 맞은 응답 수.',interpretation:'R6-120은 기존 ARGOS보다 회복했지만 기본 S0 재측정에는 미달했습니다. 문서의 선정 하한은 73/96입니다.'},
  {id:'valid',label:'유효 설명',n:96,definition:'문서 정답이 맞고, 근거·계산·조건 적용에 중대한 오류가 없는 응답 수. 잘못된 부가 주장과 명시적 오인용도 실패로 처리합니다. 단일 AI 판정입니다.',interpretation:'R6-120은 기본 S0보다 유효 설명이 2개 많았습니다. 최종 정답만 맞고 설명이 틀린 경우는 이 성공 수에 넣지 않았습니다.'},
  {id:'gsm8k',label:'GSM8K',n:40,definition:'동일한 GSM8K 개발 표본 40개의 최종 정답. 후보 선정에 재사용한 부분집합이며 새 독립 최종시험이 아닙니다.',interpretation:'기존 ARGOS 6/40에서 별도 회복 후보 R6-120 34/40으로, 15%에서 85%로 회복한 관측입니다. 같은 기반 모델에서 새로 학습한 후보이며 연속 학습 곡선이 아닙니다.'},
  {id:'kobest',label:'KoBEST 공식',n:40,definition:'KoBEST BoolQ 40개 개발 표본을 공식 정답표로 채점한 결과. 정답 오류 수정 기준의 점수와 구분합니다.',interpretation:'R6-120은 기본 S0와 같고 기존 ARGOS보다 정답 1개가 적었습니다. 이 표는 공식 정답 기준이며 오류 수정 기준을 혼합하지 않습니다.'}
];
const names={S0:'기본 S0 재측정',ARGOS_old:'기존 ARGOS',gpt_oss:'GPT-OSS-20B',glm:'GLM-4.7-Flash'};
const map={document:'document_correct',valid:'valid_explanations',gsm8k:'gsm8k_correct',kobest:'kobest_official'};
const models=rows.map(row=>({id:row.model,label:names[row.model]??row.label,kind:['gpt_oss','glm'].includes(row.model)?'teacher':row.model==='S0'?'base':row.model==='ARGOS_old'?'frozen':'candidate',prompt:row.prompt,values:Object.fromEntries(devMetrics.map(m=>[m.id,Number(row[map[m.id]])])),incomplete:Number(row.incomplete),source:`${pub}/development_metrics.csv`}));
for(const [id,table]of Object.entries(recovery.tables)){
  const item={id,label:names[id]??id,kind:id==='S0'?'base':'candidate',prompt:'verify',values:{document:table.document,valid:table.valid_explanations,gsm8k:table.gsm8k,kobest:table.kobest_boolq},incomplete:0,source:`${pub}/recovery_selection.json`};
  const old=models.findIndex(m=>m.id===id);if(old>=0)models[old]=item;else models.push(item);
}
// Cross-check independent report table, not a second hand-entered UI constant.
const reportNames={'기존 ARGOS':'ARGOS_old','GPT-OSS 20B':'gpt_oss','GLM 4.7 Flash':'glm'};
for(const row of report.queries['current-results'].rows){const model=models.find(m=>m.id===(reportNames[row.model]??row.model));assert.ok(model);for(const metric of devMetrics)assert.equal(model.values[metric.id],row[metric.id],`${model.id}/${metric.id}`);}
assert.deepEqual(recovery.eligible,[]);assert.equal(recovery.selected,null);assert.equal(recovery.final_allowed,false);assert.equal(recovery.human_reviewers,0);
const finalN=seal.document_count+seal.public_retention;assert.equal(finalN,protocol.final.items);

const [early,later]=await Promise.all([jsonl(`${pub}/synthetic_development_evidence.jsonl`),jsonl(`${pub}/recovery_development_evidence.jsonl`)]);
const exampleSpecs=[{id:'development-offer-006',label:'정답과 설명의 차이',note:'업체명 정답과 계산의 타당성은 별개입니다. R6-120은 업체명을 맞혔지만 1,700 × 28을 49,600으로 잘못 계산했습니다.'},{id:'development-weighted-006',label:'반올림의 오류',note:'중간 계산을 맞히고도 마지막 값을 잘못 반올림하면 최종 정답에 실패합니다.'},{id:'development-capacity-000',label:'정답 사례',note:'제공된 정원과 인원수를 계산하고 올림을 적용하는 합성 문서 사례입니다. 성공 사례 하나가 전체 성능을 대표하지는 않습니다.'}];
const examples=exampleSpecs.map(spec=>({...spec,responses:['ARGOS_old','R6-120','S0'].map(model=>{
  const modern=model==='R6-120'||model==='S0';
  const row=(modern?later:early).find(r=>r.id===spec.id&&r.model===model);assert.ok(row,`Missing example ${model}/${spec.id}`);
  const response=modern?row.prediction:row.response;assert.equal(typeof response,'string');
  return {model,label:names[model]??model,question:row.question,response,reference:modern?row.key:row.reference,correct:row.semantic_correct,valid:row.rationale_valid,complete:row.complete,prompt:row.prompt_variant,source:`${pub}/${modern?'recovery_development_evidence.jsonl':'synthetic_development_evidence.jsonl'}`,responseSha256:createHash('sha256').update(response).digest('hex')};
})}));
const sources=[
  [`${improvement}/ARGOS_Improvement_Report_KO.pdf`,'improvement-report.pdf'],[`${previous}/CoMPaSS_ARGOS_final_report_20261027.pdf`,'final-report.pdf'],[`${improvement}/REPORT_KO.md`,'improvement-report.md'],
  [`${previous}/REPORT_KO.md`,'frozen-report.md'],[`${previous}/data/metrics.json`,'metrics.json'],
  ...['development_metrics.csv','recovery_selection.json','runtime_identity.json','report_data.json','recovery_seal.json','recovery_protocol.json','actual_prompts.json','recovery_development_evidence.jsonl','synthetic_development_evidence.jsonl'].map(f=>[`${pub}/${f}`,f])
];
await mkdir(resolve(web,'public/sources'),{recursive:true});await mkdir(resolve(web,'src/data'),{recursive:true});
const manifest=[];
for(const [path,name]of sources){const bytes=await readFile(resolve(root,path));manifest.push({repositoryPath:path,url:`/sources/${name}`,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,commit:sourceCommit});await copyFile(resolve(root,path),resolve(web,'public/sources',name));}
const data={schemaVersion:1,sourceCommit,repository:'https://github.com/chabin0000/CoMPaSS-AI',date:'2026-10-27',baseModel:runtime.base_model,baseRevision:runtime.base_revision,frozenCheckpoint:'S5_general_control-20261022',confirmation,development:{metrics:devMetrics,models,primary:['S0','ARGOS_old','R6-120','gpt_oss','glm'],final:{status:'unmeasured',n:finalN,score:null,inferences:0},eligible:recovery.eligible,selected:recovery.selected,thresholds:recovery.thresholds},review:{humanReviewers:report.report.humanReview.human_reviewers,method:'연구자 본인 1명이 자료·결과와 최종보고서를 검수했습니다. R6 문항별 판정은 단일 AI 비블라인드 방식이며, 외부 2인 독립 검수는 미수행입니다.'},examples,sources:manifest};
await writeFile(resolve(web,'src/data/research.json'),JSON.stringify(data,null,2)+'\n');
await writeFile(resolve(web,'public/sources/research.json'),JSON.stringify(data,null,2)+'\n');
await writeFile(resolve(web,'public/sources/manifest.json'),JSON.stringify({sourceCommit,files:manifest},null,2)+'\n');
console.log(`Validated ${confirmation.length} frozen metrics, ${models.length} development models, ${examples.length} recorded examples. Final ${finalN}: ${data.development.final.status}. Sources pinned to ${sourceCommit}.`);
