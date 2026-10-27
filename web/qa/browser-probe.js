/* Development-only browser probe. Inject through DevTools/CDP on ?qa=1.
 * Measures visible-tab rAF intervals, NOT GPU duration. Not shipped by Vite.
 */
window.__ARGOS_QA__ = async function runArgosProbe({ duration=18000, label='scroll' }={}) {
  const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  await sleep(1200);
  const before=window.__ARGOS_DIAGNOSTICS__.snapshot();
  if(document.hidden)throw Error('Measurements require a visible foreground tab');
  const longTasks=[];
  const observer=new PerformanceObserver(list=>longTasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration}))));
  observer.observe({type:'longtask',buffered:false});
  window.__ARGOS_DIAGNOSTICS__.reset();
  const max=document.documentElement.scrollHeight-innerHeight;
  const started=performance.now();
  await new Promise(resolve=>{
    function frame(now){
      const t=(now-started)/duration;
      let progress;
      if(t<.55)progress=t/.55; // full slow forward pass
      else if(t<.75)progress=1-(t-.55)/.20; // fast return
      else if(t<.82)progress=(t-.75)/.07*.35;
      else if(t<.88)progress=.35-(t-.82)/.06*.35; // immediate reverse within symbols
      else if(t<.94)progress=(t-.88)/.06;
      else if(t<1)progress=1-(t-.94)/.06;
      else {scrollTo(0,max);resolve();return;}
      scrollTo(0,max*progress);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  });
  await sleep(850);
  observer.disconnect();
  const after=window.__ARGOS_DIAGNOSTICS__.snapshot();
  const statistics=values=>{
    const sorted=values.slice().sort((a,b)=>a-b);
    const mean=sorted.reduce((a,b)=>a+b,0)/sorted.length;
    return {frames:sorted.length,meanMs:mean,rafPerSecond:1000/mean,p50:sorted[Math.floor(sorted.length*.5)],p95:sorted[Math.floor(sorted.length*.95)],p99:sorted[Math.floor(sorted.length*.99)],maxMs:sorted.at(-1),over22:sorted.filter(x=>x>22).length,over33:sorted.filter(x=>x>33.4).length,over50:sorted.filter(x=>x>50).length};
  };
  const transitions=after.samples.filter((_,i)=>Math.abs(after.phaseSamples[i]-Math.round(after.phaseSamples[i]))>.02);
  return {label,timestamp:new Date().toISOString(),userAgent:navigator.userAgent,viewport:after.viewport,gpu:after.gpu,visibleAtStart:before.visible,visibleAtEnd:after.visible,qualityAtStart:before.quality,qualityAtEnd:after.quality,dpr:after.dpr,renderedParticles:after.particles,stats:statistics(after.samples),transitions:statistics(transitions),longTasks,resourcesBefore:{geometry:before.geometries,textures:before.textures,programs:before.programs},resourcesAfter:{geometry:after.geometries,textures:after.textures,programs:after.programs,drawCalls:after.drawCalls},endingPhase:after.phase,horizontalOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth};
};
