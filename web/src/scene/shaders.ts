export const vertex = /* glsl */`
  attribute vec3 target1;
  attribute vec3 target2;
  attribute vec3 target3;
  attribute vec3 target4;
  attribute vec3 target5;
  attribute vec3 target6;
  attribute vec4 visual;
  uniform float uPhase;
  uniform float uTime;
  uniform float uDpr;
  uniform float uPointScale;
  uniform float uAspect;
  uniform float uDensity;
  uniform float uBrightness;
  uniform float uBackground;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float p = uPhase;
    vec3 a = position, b = target1;
    if(p>=1.0){a=target1;b=target2;}
    if(p>=2.0){a=target2;b=target3;}
    if(p>=3.0){a=target3;b=target4;}
    if(p>=4.0){a=target4;b=target5;}
    if(p>=5.0){a=target5;b=target6;}
    float t = clamp(p-floor(p),0.0,1.0);
    if(p>=6.0){a=target6;b=target6;t=0.0;}
    vec3 pos = mix(a,b,t);
    float arc = sin(t*3.14159265);
    pos.z += arc * .085 * sin(visual.y*6.2831);
    pos.y += arc * .024 * sin(visual.y*12.5662);
    // Coordinates use width units, but tall displays retain round symbols.
    float aspectScale = min(uAspect,1.48);
    float letterWeight = (1.0-smoothstep(0.0,1.0,p))+smoothstep(5.0,6.0,p);
    float symbolScale = min(1.0,.92/max(.65,uAspect));
    float scale = mix(symbolScale,1.0,max(letterWeight,smoothstep(3.0,4.0,p)));
    pos.xy *= scale;
    pos.y *= aspectScale;
    pos.y += .12;
    float calm = smoothstep(3.15,4.0,p)*(1.0-smoothstep(4.8,5.7,p));
    float brightness = mix(1.0,.17,calm)*uBrightness;
    if(uBackground>0.5){pos=position;pos.y*=max(uAspect,1.0);brightness=mix(.53,.19,calm);}
    float depth = 1.0+pos.z*.12;
    gl_Position = vec4(pos.xy*depth,0.0,1.0);
    gl_PointSize = max(1.0,visual.x*uDpr*uPointScale);
    vColor = mix(vec3(.53,.73,1.0),vec3(.92,.98,1.0),visual.z);
    if(visual.z>.965) vColor=vec3(1.0,.72,.43);
    float twinkle = .88+.12*sin(uTime*.65+visual.y*30.0);
    vAlpha=twinkle*brightness*(1.0-smoothstep(uDensity-.035,uDensity+.015,visual.w));
  }
`;
export const fragment = /* glsl */`
  varying vec3 vColor;
  varying float vAlpha;
  void main(){
    float d=length(gl_PointCoord-.5)*2.0;
    if(d>1.0) discard;
    float core=exp(-d*d*18.0);
    float halo=exp(-d*d*4.5)*.18;
    gl_FragColor=vec4(vColor,(core+halo)*vAlpha);
  }
`;
