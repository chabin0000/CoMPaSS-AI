"""Actual model execution only. Evaluation keys are unavailable on this host."""
from pathlib import Path
import argparse,json,time,os,hashlib,subprocess,urllib.request,traceback
from student_runtime import check,load,train,atomic,read,rows,sha
from teacher_transport import extract_final
R=Path(__file__).resolve().parent
def teacher_server(name):
 cfg=read(R/'teachers.json');t=cfg['teachers'][name];m=t['model_metadata'];setup=read(R/'teacher_setup.json');binary=R/'engine/build/bin/llama-server'
 assert sha(binary)==setup['binary_sha256']
 cmd=[str(binary),'--model',str(R/'teacher_models'/m['filename']),'--alias',t['model'],'--host','127.0.0.1','--port','8080','--n-gpu-layers','99',
 '--ctx-size','12288','--parallel','1','--batch-size','512','--ubatch-size','128','--jinja','--reasoning-format','deepseek','--flash-attn','on','--verbosity','3','--no-cors-credentials']
 env=dict(os.environ);env['LD_LIBRARY_PATH']=str(binary.parent)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
 log=(R/'logs'/('teacher-'+name+'-'+str(time.time_ns())+'.log')).open('wb');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
 try:
  for _ in range(180):
   check(R)
   if proc.poll() is not None:raise RuntimeError('Teacher engine exited')
   try:
    with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2) as x:ready=x.status==200
    if ready:return proc,log,t,dict(engine_revision=cfg['llama_commit'],model_metadata=m,command=cmd,template='embedded_official_GGUF_jinja',gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip())
   except OSError:pass
   time.sleep(2)
  raise TimeoutError('Teacher failed readiness')
 except BaseException:
  proc.terminate();proc.wait(timeout=30);log.close();raise
def teacher_generate(t,prompt,cap):
 payload=dict(model=t['model'],messages=[dict(role='user',content=prompt)],max_tokens=cap,temperature=0,seed=20261027,**t.get('extra_body',{}),stream=True,stream_options={'include_usage':True})
 req=urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 began=time.monotonic();first=None;content='';reasoning='';usage={};finish=None
 with urllib.request.urlopen(req,timeout=300) as response:
  for line in response:
   if time.monotonic()-began>300:raise TimeoutError('Generation time cap')
   if not line.startswith(b'data:'):continue
   raw=line[5:].strip()
   if raw==b'[DONE]':break
   v=json.loads(raw)
   if v.get('error'):raise RuntimeError(str(v['error'])[:300])
   usage=v.get('usage') or usage
   for choice in v.get('choices',[]):
    d=choice.get('delta',{});c=d.get('content') or '';thought=d.get('reasoning_content') or d.get('reasoning') or ''
    if (c or thought) and first is None:first=time.monotonic()-began
    content+=c;reasoning+=thought;finish=choice.get('finish_reason') or finish
 valid=True
 try:content,reasoning=extract_final(dict(content=content,reasoning_content=reasoning),finish)
 except ValueError:valid=False
 return dict(prediction=content,finish_reason=finish,response_valid=valid,reasoning_private=reasoning,usage=usage,generated_tokens=usage.get('completion_tokens'),latency=time.monotonic()-began,ttft=first)
def evaluate(name,model,question_file,cap,adapter=None):
 assert name.replace('-','').replace('_','').isalnum()
 out=R/'evaluations'/name;out.mkdir(parents=True,exist_ok=False);qs=rows(R/question_file);check(R);started=time.time()
 proc=log=None;runtime={}
 try:
  if model in ['gpt_oss','glm']:proc,log,t,runtime=teacher_server(model)
  else:
   import torch
   from transformers import set_seed
   set_seed(20261026 if name=='sentinel-S0' else 20261027);torch.set_num_threads(6);base,tok,runtime=load(R,False,adapter=Path(adapter) if adapter else None)
   if not adapter:assert not any('lora_' in n for n,_ in base.named_parameters()) and not hasattr(base,'peft_config')
   runtime.update(eval_mode=not base.training,generation_config=base.generation_config.to_dict(),eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,existing_adapter_loaded=bool(adapter))
  seconds=180 if name=='sentinel-S0' else 300
  atomic(out/'runtime.json',dict(at=time.time(),model=model,adapter=adapter,question_sha256=sha(R/question_file),max_new_tokens=cap,max_seconds=seconds,**runtime))
  path=out/'responses.jsonl'
  with path.open('x') as f:
   for i,q in enumerate(qs):
    check(R);began=time.monotonic()
    try:
     if model in ['gpt_oss','glm']:v=teacher_generate(t,q['prompt'],cap)
     else:
      prefix=tok.apply_chat_template([dict(role='user',content=q['prompt'])],tokenize=False,add_generation_prompt=True)
      inp=tok(prefix,return_tensors='pt',add_special_tokens=False).to(next(base.parameters()).device);assert inp['input_ids'].shape[1]+cap<=12288
      with torch.inference_mode():gen=base.generate(**inp,max_new_tokens=cap,do_sample=False,max_time=seconds,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
      ids=gen[0,inp['input_ids'].shape[1]:].tolist();finish='eos' if ids and ids[-1]==tok.eos_token_id else 'length' if len(ids)>=cap else 'time_or_other_incomplete'
      v=dict(prediction=tok.decode(ids,skip_special_tokens=True).strip(),finish_reason=finish,response_valid=finish=='eos',generated_tokens=len(ids),input_tokens=inp['input_ids'].shape[1],input_token_ids_sha256=hashlib.sha256(json.dumps(inp['input_ids'][0].tolist()).encode()).hexdigest(),input_mask_all_ones=bool(inp['attention_mask'].all()),latency=time.monotonic()-began)
    except Exception as e:v=dict(prediction='',finish_reason='error',response_valid=False,error_type=type(e).__name__,error=str(e)[:600],latency=time.monotonic()-began)
    v.update(id=q['id'],model=model,cohort=q['cohort'],family=q['family'],variant=q.get('variant'),prompt_sha256=hashlib.sha256(q['prompt'].encode()).hexdigest(),attempt=1)
    f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    atomic(out/'progress.json',dict(at=time.time(),completed=i+1,required=len(qs),model=model,elapsed=time.time()-started,current_id=q['id']))
  atomic(out/'DONE.json',dict(at=time.time(),completed=len(qs),responses_sha256=sha(path),adapter_reloading_verified=bool(adapter),elapsed=time.time()-started))
 finally:
  if proc:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   log.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['eval','train']);a.add_argument('--name',required=True);a.add_argument('--model');a.add_argument('--questions');a.add_argument('--cap',type=int,default=4096);a.add_argument('--adapter');a.add_argument('--steps',type=int,default=75);a.add_argument('--save-steps',type=int,default=40);a.add_argument('--lr',type=float,default=5e-5);a.add_argument('--seed',type=int,default=20261027);v=a.parse_args()
 try:
  if v.action=='eval':evaluate(v.name,v.model,v.questions,v.cap,v.adapter)
  else:train(R,v.name,'base',v.lr,v.seed,v.steps,v.save_steps)
 except Exception as e:atomic(R/'FAILED.json',dict(at=time.time(),job=v.name,error_type=type(e).__name__,error=str(e)[:1200],traceback=traceback.format_exc()));raise
