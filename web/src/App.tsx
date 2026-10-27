import { useEffect, useState } from 'react';
import Universe from './scene/Universe';
import Results from './components/Results';

function staticScene(n:number){return <img className="static-stars" loading={n===0?'eager':'lazy'} src={`/universe/static-${n}.svg`} alt="" aria-hidden="true"/>;}
export default function App(){
  const [calm,setCalm]=useState(()=>matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [paused,setPaused]=useState(false);
  const [light,setLight]=useState(false);
  const [fallback,setFallback]=useState(true);
  useEffect(()=>{const q=matchMedia('(prefers-reduced-motion: reduce)');const change=()=>setCalm(q.matches);q.addEventListener('change',change);return()=>q.removeEventListener('change',change);},[]);
  return <>
    <a className="skip-link" href="#results">연구 결과로 건너뛰기</a>
    <Universe calm={calm} paused={paused} light={light} onFallback={setFallback}/>
    <main className={fallback?'static-mode':''}>
      <section className="hero" data-scene="0" aria-labelledby="hero-title">
        <div className="scene-sticky">
          <div className="hero-eyebrow"><span className="tiny-star">✧</span> CoMPaSS RESEARCH <span className="date">2026.10</span></div>
          <h1 id="hero-title" className="sr-only">ARGOS — 다중 교사 학습의 가능성과 한계</h1>
          {fallback&&staticScene(0)}
          <div className="hero-caption"><p>다중 교사 학습의<br className="mobile-break"/> 가능성과 한계를 탐구하다.</p><span>서로 다른 답변에서 시작된, 하나의 연구.</span></div>
          <a className="hero-report" href="/sources/improvement-report.pdf" target="_blank" rel="noreferrer">ARGOS 제작과 개선 연구 최종보고서 PDF 보기 ↗</a>
          <a className="scroll-cue" href="#teachers"><span>아래로, 연구의 여정을 따라</span><span className="scroll-line"/></a>
        </div>
      </section>
      <section id="teachers" className="scene-section" data-scene="1" aria-labelledby="teacher-title">
        <div className="scene-sticky"><header className="scene-heading"><p className="eyebrow">01 / TWO TEACHERS</p><h2 id="teacher-title">서로 다른 두 시선.</h2></header>
          {fallback&&staticScene(1)}
          <div className="teacher-labels"><span>GPT-OSS-20B<small>교사 01</small></span><span>GLM-4.7-Flash<small>교사 02</small></span></div>
          <div className="scene-copy"><p>같은 질문에 서로 다른 두 교사가 답합니다.<br/>각 답변을 모으고, 그 안에서 배울 수 있는 것을 찾습니다.</p><span>학습하는 것은 교사의 답변 텍스트입니다.</span></div>
        </div>
      </section>
      <section className="scene-section" data-scene="2" aria-labelledby="verify-title">
        <div className="scene-sticky"><header className="scene-heading"><p className="eyebrow">02 / VERIFY</p><h2 id="verify-title">배우기 전에, 확인하다.</h2></header>
          {fallback&&staticScene(2)}
          <div className="scene-copy"><p>정답과 계산, 제공된 근거를 대조합니다.<br/>확인할 수 없는 답은 보류하고, 통과한 자료를 학습에 사용합니다.</p><span>자동 검사와 AI 검수 · 검증 통과가 성능 향상을 보장하지는 않습니다.</span></div>
        </div>
      </section>
      <section className="scene-section" data-scene="3" aria-labelledby="student-title">
        <div className="scene-sticky"><header className="scene-heading"><p className="eyebrow">03 / A STUDENT</p><h2 id="student-title">하나의 학생, 새로운 질문.</h2></header>
          {fallback&&staticScene(3)}
          <div className="scene-copy"><p>EXAONE-3.5-7.8B-Instruct를 기반으로<br/>LoRA 어댑터를 학습한 학생 모델, ARGOS.</p><span>실제로 평가하고, 좋아진 점과 잃어버린 능력을 함께 살폈습니다.</span></div>
        </div>
      </section>
      <section id="results" data-scene="4" className="research"><Results/></section>
      <section className="ending" data-scene="6" aria-labelledby="ending-title"><div className="scene-sticky">
        <p className="ending-kicker eyebrow">EVERY ANSWER, A NEW DIRECTION</p>
        <h2 id="ending-title" className="sr-only">COMPASS</h2>{fallback&&staticScene(6)}
        <div className="ending-copy"><p>답을 찾는 일은,<br/>다음 질문의 방향을 찾는 일.</p><span>CoMPaSS · 2026</span></div>
      </div></section>
    </main>
    <details className="display-options"><summary aria-label="화면 설정">✧<span>화면 설정</span></summary><div className="options-panel"><label><input type="checkbox" checked={paused} onChange={e=>setPaused(e.target.checked)}/>별 반짝임 멈춤</label><label><input type="checkbox" checked={light} onChange={e=>setLight(e.target.checked)}/>가벼운 화면</label><label><input type="checkbox" checked={calm} onChange={e=>setCalm(e.target.checked)}/>움직임 줄이기</label></div></details>
  </>;
}

