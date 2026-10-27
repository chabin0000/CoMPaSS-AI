import { useEffect, useRef } from 'react';
import type * as THREE from 'three';
import { measureTimeline, phaseAt } from './timeline';
import { vertex, fragment } from './shaders';

type Options = { calm: boolean; light: boolean; paused: boolean; onFallback: (v:boolean)=>void };
type Quality = { name: string; density: number; dpr: number };
const profiles: Quality[] = [{name:'high',density:1,dpr:1.5},{name:'medium',density:2/3,dpr:1.15},{name:'low',density:1/3,dpr:1}];
const N=12000;
function seeded(seed:number){return ()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

export default function Universe({calm,light,paused,onFallback}:Options){
  const host=useRef<HTMLDivElement>(null);
  const liveOptions=useRef({light,paused});
  useEffect(()=>{liveOptions.current={light,paused};},[light,paused]);
  useEffect(()=>{
    if(calm){onFallback(true);return;}
    const container=host.current!;
    const controller=new AbortController();
    let dead=false, contextLost=false, raf=0, renderer:THREE.WebGLRenderer|undefined;
    let geometry:THREE.BufferGeometry|undefined, material:THREE.ShaderMaterial|undefined;
    let backdropGeometry:THREE.BufferGeometry|undefined, backdropMaterial:THREE.ShaderMaterial|undefined;
    let cleanup=()=>{};
    onFallback(true);
    async function init(){
      try{
        const Three=await import('./gpu');
        const buffers=await Promise.all(Array.from({length:7},(_,i)=>fetch(`/universe/shape-${i}.bin`,{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('Star asset unavailable');return r.arrayBuffer();})));
        if(dead)return;
        renderer=new Three.WebGLRenderer({alpha:true,antialias:false,powerPreference:'low-power',stencil:false,depth:false});
        renderer.debug.onShaderError=()=>{throw new Error('Star shader could not compile');};
        renderer.setClearColor(0,0);
        container.appendChild(renderer.domElement);
        renderer.domElement.setAttribute('aria-hidden','true');
        const gl=renderer.getContext();
        const gpu=gl.getExtension('WEBGL_debug_renderer_info');
        const gpuName=gpu?gl.getParameter(gpu.UNMASKED_RENDERER_WEBGL):'unavailable';
        const scene=new Three.Scene();
        const camera=new Three.Camera();
        geometry=new Three.BufferGeometry();
        buffers.forEach((b,i)=>geometry!.setAttribute(i===0?'position':`target${i}`,new Three.BufferAttribute(new Float32Array(b),3)));
        const rand=seeded(20261027), visual=new Float32Array(N*4);
        for(let i=0;i<N;i++){
          const r=rand();
          visual.set([1.9+Math.pow(r,3)*4.7+(r>.993?5:0),rand(),rand(),i/N],i*4);
        }
        geometry.setAttribute('visual',new Three.BufferAttribute(visual,4));
        const initialQuality=()=>innerWidth<520?2:((navigator.hardwareConcurrency||4)<6||innerWidth<1000?1:0);
        let quality=liveOptions.current.light?2:initialQuality();
        const uniforms={uPhase:{value:0},uTime:{value:0},uDpr:{value:1},uPointScale:{value:1},uAspect:{value:1},uDensity:{value:profiles[quality].density},uBrightness:{value:1},uBackground:{value:0}};
        material=new Three.ShaderMaterial({vertexShader:vertex,fragmentShader:fragment,uniforms,transparent:true,depthTest:false,depthWrite:false,blending:Three.AdditiveBlending});
        const points=new Three.Points(geometry,material);points.frustumCulled=false;scene.add(points);
        backdropGeometry=new Three.BufferGeometry();
        const bg=new Float32Array(1400*3),bv=new Float32Array(1400*4);
        for(let i=0;i<1400;i++){bg.set([(rand()-.5)*2.5,(rand()-.5)*3.6,0],i*3);bv.set([1.6+rand()*2.5,rand(),rand(),i/1400],i*4);}
        const bp=new Three.BufferAttribute(bg,3);
        backdropGeometry.setAttribute('position',bp);
        for(let i=1;i<7;i++)backdropGeometry.setAttribute(`target${i}`,bp);
        backdropGeometry.setAttribute('visual',new Three.BufferAttribute(bv,4));
        backdropMaterial=material.clone();backdropMaterial.uniforms.uBackground.value=1;
        const background=new Three.Points(backdropGeometry,backdropMaterial);background.frustumCulled=false;scene.add(background);
        let keys=measureTimeline(), target=phaseAt(scrollY,keys), phase=target, last=0, time=0;
        let needMeasure=true, qualityWindow:number[]=[], badWindows=0, cooldown=0, desiredQuality=quality;
        let activeDpr=0, renderedFrames=0, hiddenCount=0, transitionFrames=0;
        let previousLight=liveOptions.current.light;
        const samples=new Float64Array(8192), phaseSamples=new Float32Array(8192);
        let sampleCount=0;
        const qa=import.meta.env.DEV||new URLSearchParams(location.search).has('qa');
        const resize=()=>{needMeasure=true;wake();};
        const measure=()=>{
          keys=measureTimeline();
          const dpr=Math.min(devicePixelRatio||1,profiles[quality].dpr);
          renderer!.setPixelRatio(dpr);renderer!.setSize(innerWidth,innerHeight);activeDpr=dpr;
          uniforms.uDpr.value=dpr;uniforms.uAspect.value=innerWidth/innerHeight;
          uniforms.uPointScale.value=Math.min(1,Math.sqrt(innerWidth/1100));
          backdropMaterial!.uniforms.uDpr.value=dpr;backdropMaterial!.uniforms.uAspect.value=innerWidth/innerHeight;
          backdropMaterial!.uniforms.uPointScale.value=uniforms.uPointScale.value;
          desiredQuality=Math.max(desiredQuality,initialQuality());
          needMeasure=false;
        };
        const onScroll=()=>{target=phaseAt(scrollY,keys);wake();};
        const frame=(now:number)=>{
          raf=0;if(dead||contextLost||document.hidden)return;
          const raw=last?now-last:0;
          const dt=Math.min(raw/1000,.05);last=now;
          if(needMeasure){measure();target=phaseAt(scrollY,keys);}
          const opt=liveOptions.current;
          // Only one smoothing layer, independent of refresh rate.
          phase+=(target-phase)*(1-Math.exp(-dt*15));
          if(Math.abs(target-phase)<.00005)phase=target;
          const morphing=Math.abs(phase-Math.round(phase))>.015;
          if(morphing)transitionFrames++;
          if(!opt.paused)time+=dt;
          if(previousLight!==opt.light){desiredQuality=opt.light?2:initialQuality();previousLight=opt.light;badWindows=0;cooldown=now+10000;}
          if(raw>0&&renderedFrames>100){
            if(qa){const index=sampleCount%8192;samples[index]=raw;phaseSamples[index]=phase;sampleCount++;}
            qualityWindow.push(raw);
            if(qualityWindow.length>=180){
              qualityWindow.sort((a,b)=>a-b);
              const p95=qualityWindow[Math.floor(qualityWindow.length*.95)];
              badWindows=p95>25?badWindows+1:0;
              if(badWindows>=2&&now>cooldown&&desiredQuality<2){desiredQuality++;cooldown=now+12000;badWindows=0;}
              qualityWindow=[];
            }
          }
          // Fade density using a uniform; do not allocate or replace resources.
          const densityTarget=profiles[desiredQuality].density;
          uniforms.uDensity.value+=(densityTarget-uniforms.uDensity.value)*(1-Math.exp(-dt*1.8));
          const density=uniforms.uDensity.value;
          geometry!.setDrawRange(0,Math.min(N,Math.ceil((density+.02)*N)));
          backdropGeometry!.setDrawRange(0,Math.min(1400,Math.ceil((density+.02)*1400)));
          // Resolution changes wait for a stable symbolic hold, never a morph.
          if(quality!==desiredQuality&&!morphing){quality=desiredQuality;needMeasure=true;}
          uniforms.uPhase.value=phase;uniforms.uTime.value=time;
          backdropMaterial!.uniforms.uPhase.value=phase;backdropMaterial!.uniforms.uTime.value=time;
          backdropMaterial!.uniforms.uDensity.value=density;
          renderer!.render(scene,camera);renderedFrames++;
          if(!opt.paused||phase!==target||needMeasure||Math.abs(density-densityTarget)>.0001)raf=requestAnimationFrame(frame);
        };
        const wake=()=>{if(!raf&&!dead&&!contextLost&&!document.hidden)raf=requestAnimationFrame(frame);};
        const visibility=()=>{last=0;qualityWindow=[];if(document.hidden){hiddenCount++;cancelAnimationFrame(raf);raf=0;}else{needMeasure=true;wake();}};
        const lost=(event:Event)=>{event.preventDefault();contextLost=true;cancelAnimationFrame(raf);raf=0;onFallback(true);};
        const restored=()=>{contextLost=false;last=0;needMeasure=true;onFallback(false);wake();};
        const ro=new ResizeObserver(resize);ro.observe(document.querySelector('main')!);
        addEventListener('scroll',onScroll,{passive:true});addEventListener('resize',resize,{passive:true});
        addEventListener('argos-options',wake);document.addEventListener('visibilitychange',visibility);
        renderer.domElement.addEventListener('webglcontextlost',lost);renderer.domElement.addEventListener('webglcontextrestored',restored);
        const diagnostics={
          snapshot:()=>{const count=Math.min(sampleCount,8192),start=Math.max(0,sampleCount-8192);return {quality:profiles[quality].name,desiredQuality:profiles[desiredQuality].name,density:uniforms.uDensity.value,dpr:activeDpr,phase,target,renderedFrames,transitionFrames,hiddenCount,rafActive:!!raf,visible:!document.hidden,gpu:gpuName,viewport:[innerWidth,innerHeight],geometries:renderer!.info.memory.geometries,textures:renderer!.info.memory.textures,programs:renderer!.info.programs?.length,drawCalls:renderer!.info.render.calls,particles:geometry!.drawRange.count,samples:Array.from({length:count},(_,i)=>samples[(start+i)%8192]),phaseSamples:Array.from({length:count},(_,i)=>phaseSamples[(start+i)%8192]),keys};},
          reset:()=>{sampleCount=0;},
          loseContext:()=>renderer!.forceContextLoss(), restoreContext:()=>renderer!.forceContextRestore()
        };
        if(qa)Object.assign(window,{__ARGOS_DIAGNOSTICS__:diagnostics});
        cleanup=()=>{
          ro.disconnect();removeEventListener('scroll',onScroll);removeEventListener('resize',resize);removeEventListener('argos-options',wake);
          document.removeEventListener('visibilitychange',visibility);renderer!.domElement.removeEventListener('webglcontextlost',lost);renderer!.domElement.removeEventListener('webglcontextrestored',restored);
          if(qa)Reflect.deleteProperty(window,'__ARGOS_DIAGNOSTICS__');
        };
        measure();uniforms.uPhase.value=target;phase=target;renderer.compile(scene,camera);renderer.render(scene,camera);onFallback(false);wake();
      }catch(error){if(!dead){console.warn('Static star scene selected.',error);onFallback(true);}}
    }
    void init();
    return()=>{dead=true;controller.abort();cancelAnimationFrame(raf);cleanup();geometry?.dispose();material?.dispose();backdropGeometry?.dispose();backdropMaterial?.dispose();renderer?.dispose();renderer?.forceContextLoss();renderer?.domElement.remove();};
  },[calm,onFallback]);
  useEffect(()=>{dispatchEvent(new Event('argos-options'));},[light,paused]);
  return <div ref={host} className="universe" aria-hidden="true"/>;
}

