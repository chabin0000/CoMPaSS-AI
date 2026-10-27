export type Keyframe = { y: number; phase: number };
export function phaseAt(y: number, keys: Keyframe[]): number {
  for (let i = 1; i < keys.length; i++) {
    if (y <= keys[i].y) {
      const a = keys[i - 1], b = keys[i];
      const t = Math.max(0, Math.min(1, (y - a.y) / Math.max(1, b.y - a.y)));
      // Quintic interpolation has zero velocity and acceleration at each hold.
      const s = t*t*t*(t*(t*6-15)+10);
      return a.phase + (b.phase-a.phase)*s;
    }
  }
  return keys.at(-1)?.phase ?? 0;
}
export function measureTimeline(): Keyframe[] {
  const h = window.innerHeight;
  const keys: Keyframe[] = [{ y: 0, phase: 0 }];
  for (const el of document.querySelectorAll<HTMLElement>('[data-scene]')) {
    const phase = Number(el.dataset.scene);
    const top = el.getBoundingClientRect().top + window.scrollY;
    if (phase === 6) {
      keys.push({y:top-h*.28,phase:4},{y:top+h*.12,phase:5},{y:top+h*.60,phase:6});
    } else {
      keys.push({ y: Math.max(0,top-h*.06), phase });
      if (phase < 4) keys.push({y:top+el.offsetHeight-h*.8,phase});
    }
  }
  return keys.sort((a,b)=>a.y-b.y);
}
