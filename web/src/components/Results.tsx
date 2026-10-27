import { useId, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent } from 'react';
import research from '../data/research.json';

const percentage=(n:number,d:number)=>`${(100*n/d).toFixed(1).replace(/\.0$/,'')}%`;
const tones:Record<string,string>={S0:'#7b8e9e',ARGOS:'#b5d9f6',ARGOS_old:'#8294b7','R6-120':'#a6d7f8',gpt_oss:'#d2c0aa',glm:'#95bfb8'};
function Tabs({items,value,onChange,label,panelId}:{items:{id:string;label:string}[];value:string;onChange:(id:string)=>void;label:string;panelId:string}){
  const ref=useRef<HTMLDivElement>(null),id=useId();
  function keyboard(event:KeyboardEvent<HTMLButtonElement>,index:number){
    if(event.ctrlKey||event.metaKey||event.altKey)return;
    let next:number;
    if(event.key==='ArrowRight')next=(index+1)%items.length;
    else if(event.key==='ArrowLeft')next=(index+items.length-1)%items.length;
    else if(event.key==='Home')next=0;
    else if(event.key==='End')next=items.length-1;
    else return;
    event.preventDefault();onChange(items[next].id);ref.current?.querySelectorAll<HTMLButtonElement>('button')[next].focus();
  }
  return <div className="tabs" role="tablist" aria-label={label} ref={ref}>{items.map((item,index)=><button key={item.id} role="tab" aria-controls={panelId} id={`${id}-${item.id}`} aria-selected={item.id===value} tabIndex={item.id===value?0:-1} onKeyDown={e=>keyboard(e,index)} onClick={()=>onChange(item.id)}>{item.label}</button>)}</div>;
}
function Bars({rows,n,label}:{rows:{id:string;label:string;correct:number}[];n:number;label:string}){
  return <div className="chart" role="img" aria-label={`${label}. ${rows.map(r=>`${r.label} ${r.correct}/${n}, ${percentage(r.correct,n)}`).join('. ')}. 아래 표에서 전체 수치를 확인할 수 있습니다.`}>
    <div className="chart-axis" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>
    <div className="chart-rows">{rows.map(r=><div className="chart-row" key={r.id}><div className="bar-heading"><span>{r.label}</span><span><strong>{percentage(r.correct,n)}</strong><small>{r.correct} / {n}</small></span></div><div className="bar-track"><div className="bar" style={{'--bar-width':`${r.correct/n*100}%`,'--bar-color':tones[r.id]??'#9ca9be'} as CSSProperties}/></div></div>)}</div>
    <p className="axis-label">성공률 (%) · 정답 수 / 전체 문항</p>
  </div>;
}
function TableWrap({children,label}:{children:React.ReactNode;label:string}){return <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>{children}</div>;}
const num=(n:number,d:number)=><>{n}<span className="denominator"> / {d}</span><small className="cell-percent">{percentage(n,d)}</small></>;

export function ConfirmationCard(){
  const [selected,setSelected]=useState('source_claims');
  const metric=research.confirmation.find(m=>m.id===selected)!;
  return <article className="result-card" aria-labelledby="frozen-title">
    <header className="card-header"><div className="card-meta"><span>실험 A</span><span className="pill">동결 모델 확인평가</span></div><h3 id="frozen-title">얻은 충실도, 잃은 수학.</h3><p>선택을 마친 기존 ARGOS를 새 문항으로 확인했습니다.<br/>개선과 저하가 과제마다 다른 방향으로 나타났습니다.</p></header>
    <Tabs items={research.confirmation.map(m=>({id:m.id,label:m.short}))} value={selected} onChange={setSelected} label="동결 모델 평가 지표" panelId="frozen-panel"/>
    <section className="chart-panel" role="tabpanel" tabIndex={0} id="frozen-panel" aria-label={metric.label}><div className="metric-title"><h4>{metric.label}</h4><span>높을수록 좋음</span></div><Bars rows={metric.results.map(r=>({id:r.model,label:r.model==='S0'?'기반 EXAONE S0':'기존 ARGOS',correct:r.correct}))} n={metric.results[0].n} label={metric.label}/><p className="metric-definition">{metric.definition}</p></section>
    <TableWrap label="동결 확인평가 전체 비교표. 좁은 화면에서 표 안을 가로로 스크롤할 수 있습니다."><table><caption>동결 모델 확인평가 · 성공 수 / 분모 (성공률)</caption><thead><tr><th scope="col">평가 지표</th><th scope="col">기반 EXAONE S0</th><th scope="col">기존 ARGOS</th></tr></thead><tbody>{research.confirmation.map(m=><tr key={m.id} className={selected===m.id?'selected-row':''}><th scope="row">{m.label}</th>{m.results.map(r=><td key={r.model}>{num(r.correct,r.n)}</td>)}</tr>)}</tbody></table></TableWrap>
    <div className="card-reading"><span className="reading-label">관측의 의미</span><p>{metric.interpretation}</p></div>
    <div className="card-footnote"><p><b>평가 범위</b> 새 합성 240문항과 공개 보조 120문항. 각 벤치마크는 40문항 표본이며 전체 공식 성적이 아닙니다. 교사는 이 새 240문항에 재평가하지 않았습니다. 외부의 독립 재현이나 전체 분석이 사전등록된 확증 시험은 아닙니다.</p><details><summary>체크포인트·실행 조건과 판정 보기</summary><p>ARGOS: {research.frozenCheckpoint}. 기반: {research.baseModel}. 같은 RTX 5060 Ti 8GB, NF4·BF16, greedy, 최대 512토큰으로 S0와 ARGOS를 교대로 실행했습니다. 미완료·미확정도 전체 분모에 남기고 정답으로 세지 않습니다. 문항 판정은 단일 AI이며 독립 사람 검수 완료 인원은 0명입니다.</p></details><div className="source-links"><a href="/sources/metrics.json" target="_blank" rel="noreferrer">집계 원자료 ↗</a><a href="/sources/frozen-report.md" target="_blank" rel="noreferrer">판정 기준과 보고서 ↗</a></div></div>
  </article>;
}

export function DevelopmentCard(){
  const [selected,setSelected]=useState('gsm8k'),[expanded,setExpanded]=useState(false);
  const dev=research.development,metric=dev.metrics.find(m=>m.id===selected)!;
  const main=dev.primary.map(id=>dev.models.find(m=>m.id===id)!);
  const shown=expanded?dev.models:main;
  const val=(values:Record<string,number>,key:string)=>values[key];
  return <article className="result-card" aria-labelledby="dev-title">
    <header className="card-header"><div className="card-meta"><span>실험 B</span><span className="pill warm">후속 후보 개발평가</span></div><h3 id="dev-title">회복은 관측됐다.<br className="mobile-break"/> 질문은 남았다.</h3><p>같은 기반 모델에서 새로 학습한 R6 후보.<br/>기존 ARGOS에서 이어서 학습한 모델이 아닙니다.</p></header>
    <Tabs items={dev.metrics} value={selected} onChange={setSelected} label="후속 개발평가 지표" panelId="development-panel"/>
    <section className="chart-panel" role="tabpanel" tabIndex={0} id="development-panel" aria-label={metric.label}><div className="metric-title"><h4>{metric.label} · 개발 표본</h4><span>높을수록 좋음</span></div><Bars rows={main.map(m=>({id:m.id,label:m.label,correct:val(m.values,selected)}))} n={metric.n} label={metric.label}/><p className="metric-definition">{metric.definition}</p></section>
    <TableWrap label="후속 개발평가 비교표. 좁은 화면에서 표 안을 가로로 스크롤할 수 있습니다."><table className="dev-table"><caption>동일 개발 표본 · 정답 또는 유효 설명 수 / 분모 (비율)</caption><thead><tr><th scope="col">모델·체크포인트</th>{dev.metrics.map(m=><th scope="col" key={m.id}>{m.label}</th>)}</tr></thead><tbody>{shown.map(m=><tr key={m.id} className={m.id==='R6-120'?'selected-row':''}><th scope="row">{m.label}{m.kind==='teacher'&&<span className="table-tag">교사</span>}{m.kind==='candidate'&&<span className="table-tag">후보</span>}</th>{dev.metrics.map(metric=><td key={metric.id}>{num(val(m.values,metric.id),metric.n)}</td>)}</tr>)}</tbody></table></TableWrap>
    <button className="expand-table" onClick={()=>setExpanded(!expanded)} aria-expanded={expanded}>{expanded?'주요 5개 모델만 보기':'B4와 다른 저장본까지 전체 13개 조건 보기'} <span aria-hidden="true">{expanded?'−':'+'}</span></button>
    <div className="card-reading"><span className="reading-label">관측의 의미</span><p>{metric.interpretation}</p></div>
    <div className="card-footnote"><p><b>개발평가의 한계</b> 후보 선정에 사용한 동일 개발 표본의 관측 결과입니다. 전체 벤치마크 점수나 새 독립 최종평가 결과가 아닙니다. R6-080과 R6-120의 좋은 점수만 합쳐 한 모델의 성능으로 표시하지 않습니다.</p><details><summary>모델별 실행 조건과 프롬프트 차이 보기</summary><p>학생·기본 모델: 고정 EXAONE 리비전, Transformers·PEFT, NF4 4비트·BF16. 두 교사: llama.cpp, Q4_K_M GGUF. GPT-OSS는 reasoning medium, GLM은 thinking 활성. 생성 상한은 8,192토큰입니다. 기본 S0와 R6는 verify 지시문, 기존 ARGOS와 두 교사는 direct 지시문을 사용했습니다. S0와 R6는 마지막 실행에서 측정했고 기존 ARGOS·B4·교사 결과는 같은 개발 문항의 앞선 실행에서 가져왔습니다. GLM의 미완료 5건도 분모에 포함합니다. 실행 조건이 완전히 같은 인과 비교가 아닙니다.</p></details><div className="source-links"><a href="/sources/development_metrics.csv" download>개발 집계 CSV ↓</a><a href="/sources/recovery_selection.json" target="_blank" rel="noreferrer">R6 선정 원자료 ↗</a><a href="/sources/runtime_identity.json" target="_blank" rel="noreferrer">실행 조건 ↗</a></div></div>
  </article>;
}

export function ResponseExplorer(){
  const [exampleId,setExampleId]=useState(research.examples[0].id),[model,setModel]=useState('R6-120');
  const example=research.examples.find(e=>e.id===exampleId)!,response=example.responses.find(r=>r.model===model)!;
  const task=response.question.includes('자료 식별자:')?response.question.slice(response.question.indexOf('자료 식별자:')):response.question;
  return <section className="response-section wide" aria-labelledby="response-title"><div className="section-intro"><p className="eyebrow">05 / INSIDE THE ANSWERS</p><h2 id="response-title">점수 너머의 답변.</h2><p>실제 응답 살펴보기 · 기록된 개발평가 예시입니다.</p></div>
    <div className="response-card"><Tabs items={research.examples} value={exampleId} onChange={setExampleId} label="저장된 평가 사례" panelId="response-panel"/><div className="response-body" id="response-panel" role="tabpanel" aria-label={example.label} tabIndex={0}><div className="response-question"><p className="eyebrow">제공 문서와 질문</p><pre>{task}</pre><div className="reference-answer">기준 정답 <strong>{response.reference}</strong></div></div><div className="response-answer"><label className="model-select">기록된 모델<select value={model} onChange={e=>setModel(e.target.value)}>{example.responses.map(r=><option value={r.model} key={r.model}>{r.label}</option>)}</select></label><div className="verdict"><span className={response.correct?'pass':'fail'}>최종 정답 {response.correct?'통과':'실패'}</span><span className={response.valid?'pass':'fail'}>유효 설명 {response.valid?'통과':'실패'}</span></div><pre className="actual-answer">{response.response}</pre></div></div><div className="response-note"><p>{example.note}</p><span>자동 검사·단일 AI 판정 / 독립 사람 검수 0명. 실시간 추론이 아닌 저장된 응답이며, 세 사례는 보고된 실패 양상과 정답 사례를 설명하기 위해 선택했습니다.</span><details><summary>실제 전체 프롬프트와 출처</summary><pre>{response.question}</pre><p>프롬프트 유형: {response.prompt} · 문항: {example.id}</p><a href={`/sources/${response.source.split('/').at(-1)}`} target="_blank" rel="noreferrer">이 응답을 포함한 공개 원자료 ↗</a></details></div></div>
  </section>;
}

export default function Results(){
  const old=research.development.models.find(m=>m.id==='ARGOS_old')!;
  const recovered=research.development.models.find(m=>m.id==='R6-120')!;
  const n=research.development.metrics.find(m=>m.id==='gsm8k')!.n;
  const gain=(recovered.values.gsm8k-old.values.gsm8k)/n*100;
  return <>
    <div className="prose research-opening"><p className="eyebrow">04 / OBSERVATIONS</p><h2>나아진 것과,<br/>다시 배워야 했던 것.</h2><p>검증된 교사의 답변으로 학습하면 더 좋은 학생이 될 수 있을까요? CoMPaSS는 이 질문을 실제 학습과 평가로 살펴본 연구입니다.</p><p>ARGOS는 그 안에서 만든 학생 모델·어댑터 계열입니다. 교사의 가중치를 합치는 대신, 검증한 답변 텍스트를 EXAONE 기반 모델의 LoRA 학습에 사용했습니다.</p><p>실제 본실험은 임대한 RTX A5000 24GB에서 진행했습니다. 같은 기반 모델에서 학습 조건별로 독립 어댑터 10개를 만들고 비교했습니다.</p></div>
    <div className="wide"><ConfirmationCard/></div>
    <div className="prose interlude"><p className="eyebrow">A SEPARATE RECOVERY EXPERIMENT</p><h2>잃어버린 수학을<br/>다시 배울 수 있을까.</h2><p>후속 R6는 같은 기반 모델에서 새로 시작한 별도 회복 후보입니다. 계산 풀이를 포함한 자료로 한 번 학습하고, 40·80·120단계 저장본을 같은 개발 표본에서 평가했습니다.</p><div className="recovery-stat"><div><span>{old.values.gsm8k}</span><span className="stat-arrow">→</span><span>{recovered.values.gsm8k}</span><small> / {n}</small></div><p>기존 ARGOS → R6-120<br/><b>동일 GSM8K 개발 표본 · +{gain}%p</b></p></div><p className="note">{percentage(old.values.gsm8k,n)}에서 {percentage(recovered.values.gsm8k,n)}로의 관측입니다. 새 독립 최종시험의 성적이나 연속 학습 곡선을 뜻하지 않습니다.</p></div>
    <div className="wide"><DevelopmentCard/></div>
    <section className="prose interpretation" aria-labelledby="interpret-title"><p className="eyebrow">WHAT THE EVIDENCE SAYS</p><h2 id="interpret-title">부분 회복.<br/>전반적 우월성은 미입증.</h2><p>R6-120은 문서와 수학에서 기존 ARGOS보다 회복했습니다. 그러나 기본 모델의 문서·수학 점수에 함께 도달하지 못했고, KoBEST 공식 정답은 기존 ARGOS보다 1개 줄었습니다.</p><div className="outcome"><div><strong>{research.development.eligible.length}개</strong><span>전체 승격 기준을 통과한 후보</span></div><div><strong>미측정</strong><span>마지막 봉인 최종 {research.development.final.n}문항</span></div></div><p>R6-120은 승격된 최종 모델이 아닙니다. 통과 후보가 없어 새 합성 문서 200개와 이전 미추론 공개 최종 80개에 대한 추론은 실행하지 않고 연구를 동결했습니다.</p><ul className="limitations"><li><b>검수의 한계.</b> 독립적인 사람 검수 완료 인원은 {research.review.humanReviewers}명입니다. R6는 단일 AI의 비블라인드 판정이며, AI 간 보고서 협의를 독립 사람 검수로 세지 않습니다.</li><li><b>비교의 한계.</b> 개발 표본을 후보 선정에 재사용했고, 모델별 프롬프트·엔진·양자화가 다릅니다. 반복 학습 시드와 새 독립 최종시험으로 회복의 일반성을 확인하지 못했습니다.</li><li><b>해석의 한계.</b> 과제별 개선을 일반 지능 향상이나 두 교사 초월로 바꾸어 말할 수 없습니다. 검증·자료 구성·학습량의 영향을 완전히 분리한 인과 실험도 아닙니다.</li></ul></section>
    <ResponseExplorer/>
    <section className="prose sources-section" aria-labelledby="source-title"><p className="eyebrow">06 / OPEN RECORD</p><h2 id="source-title">결론뿐 아니라,<br/>그 근거까지.</h2><p>좋아진 결과, 저하된 결과, 아직 측정하지 않은 것.<br/>이 연구의 기록을 함께 공개합니다.</p><div className="download-links"><a href="/sources/improvement-report.pdf" download="ARGOS_Improvement_Report_KO.pdf"><div><span>전체 연구 보고서</span><small>한국어 · PDF · 39쪽</small></div><span aria-hidden="true">↓</span></a><a href={research.repository} target="_blank" rel="noreferrer"><div><span>CoMPaSS-AI on GitHub</span><small>연구 저장소와 재현 자료</small></div><span aria-hidden="true">↗</span></a><a href="/sources/research.json" download><div><span>이 페이지의 데이터</span><small>두 평가군 · 모델·분모·출처가 포함된 JSON</small></div><span aria-hidden="true">↓</span></a></div><p className="provenance">자료 기준일 {research.date} · 기준 커밋 <a href={`${research.repository}/tree/${research.sourceCommit}`} target="_blank" rel="noreferrer">{research.sourceCommit.slice(0,7)}</a><br/><a href="/sources/manifest.json" target="_blank" rel="noreferrer">원자료 경로·SHA-256 목록</a> · <a href="/sources/improvement-report.md" target="_blank" rel="noreferrer">보고서 텍스트</a></p></section>
  </>;
}

