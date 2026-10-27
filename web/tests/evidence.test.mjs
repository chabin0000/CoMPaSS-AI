import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { phaseAt } from '../src/scene/timeline.ts';
const read=async path=>JSON.parse(await readFile(new URL(path,import.meta.url),'utf8'));
const data=await read('../src/data/research.json');

test('every published source matches the pinned repository input byte for byte',async()=>{
  for(const source of data.sources){
    const published=await readFile(new URL(`../public${source.url}`,import.meta.url));
    try { const original=await readFile(new URL(`../../${source.repositoryPath}`,import.meta.url)); assert.deepEqual(published,original,source.repositoryPath); }
    catch(error) { if(error.code!=='ENOENT')throw error; }
    assert.equal(createHash('sha256').update(published).digest('hex'),source.sha256);
  }
});
test('all frozen rates retain the correct cohort, denominator and unresolved cases',async()=>{
  const raw=await read('../public/sources/metrics.json');
  for(const metric of data.confirmation)for(const row of metric.results){
    const evidence=raw.find(r=>r.model===row.model&&r.cohort===metric.id);
    assert.equal(row.correct,evidence.semantic_correct);assert.equal(row.n,evidence.n);
    assert.equal(row.unknown,evidence.unknown);assert.equal(row.rate,row.correct/row.n);
  }
  const oldBase=data.confirmation.find(m=>m.id==='gsm8k').results.find(r=>r.model==='S0');
  const newBase=data.development.models.find(m=>m.id==='S0');
  assert.equal(oldBase.correct,37);assert.equal(newBase.values.gsm8k,35);
});
test('R6 checkpoints stay separate, and an unrun final is null rather than zero',async()=>{
  const raw=await read('../public/sources/recovery_selection.json');
  for(const id of ['R6-040','R6-080','R6-120']){
    const model=data.development.models.find(m=>m.id===id),r=raw.tables[id];
    assert.deepEqual(model.values,{document:r.document,valid:r.valid_explanations,gsm8k:r.gsm8k,kobest:r.kobest_boolq});
  }
  assert.deepEqual(data.development.final,{status:'unmeasured',n:280,score:null,inferences:0});
  assert.equal(data.development.selected,raw.selected);assert.deepEqual(data.development.eligible,raw.eligible);
});
test('recorded examples preserve exact model output and AI adjudication',async()=>{
  const cache=new Map();
  for(const example of data.examples)for(const response of example.responses){
    if(!cache.has(response.source))cache.set(response.source,(await readFile(new URL(`../public/sources/${response.source.split('/').at(-1)}`,import.meta.url),'utf8')).trim().split(/\r?\n/).map(x=>JSON.parse(x)));
    const raw=cache.get(response.source).find(r=>r.id===example.id&&r.model===response.model);
    assert.equal(response.response,raw.prediction??raw.response);assert.equal(response.correct,raw.semantic_correct);assert.equal(response.valid,raw.rationale_valid);
    assert.equal(createHash('sha256').update(response.response).digest('hex'),response.responseSha256);
  }
});
test('scroll timeline is reversible, has stationary holds and reaches permanent COMPASS',()=>{
  const keys=[{y:0,phase:0},{y:300,phase:0},{y:900,phase:1},{y:1200,phase:1},{y:2000,phase:6}];
  const forward=Array.from({length:201},(_,i)=>phaseAt(i*10,keys));
  const backward=Array.from({length:201},(_,i)=>phaseAt(2000-i*10,keys)).reverse();
  assert.deepEqual(forward,backward);assert.equal(phaseAt(250,keys),0);assert.equal(phaseAt(1000,keys),1);assert.equal(phaseAt(100000,keys),6);
  assert.ok(Math.abs(phaseAt(901,keys)-phaseAt(899,keys))<.00001);
});
