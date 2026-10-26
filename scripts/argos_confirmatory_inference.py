"""Question-only frozen NF4 inference for the final confirmation extension.

The directory must contain SEALED.json, protocol.json and inference.questions.jsonl.
No labels are read by this runner. Training is intentionally absent.
"""
import argparse, contextlib, hashlib, importlib.metadata, inspect, json, os, time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def put(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def run(root,adapter,smoke_only=False):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    from bitsandbytes.nn import Linear4bit
    from compass_ai.exaone_runtime import configure_exaone_embeddings
    root=Path(root).resolve();adapter=Path(adapter).resolve();seal=read(root/'SEALED.json');policy=read(root/'protocol.json')
    assert sha(root/'protocol.json')==seal['files']['protocol.json']
    assert sha(root/'inference.questions.jsonl')==seal['files']['inference.questions.jsonl']
    for name,h in seal['adapter_files'].items():assert sha(adapter/name)==h
    assert policy['training_performed'] is False and policy['model_choice_fixed_before_new_outputs']
    out=root/'inference';out.mkdir(exist_ok=True)
    put(out/'progress.json',dict(at=time.time(),stage='loading_nf4',completed=0,required=720))
    set_seed(policy['inference']['seed']);torch.set_num_threads(6)
    cache=read(root/'base_cache.LOCAL_ONLY.json')['path']
    tok=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=True)
    base=AutoModelForCausalLM.from_pretrained(cache,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    configure_exaone_embeddings(base);base.eval();base.config.use_cache=True
    quant=[x for x in base.modules() if isinstance(x,Linear4bit)]
    assert len(quant)==224 and base.is_loaded_in_4bit
    assert all(x.weight.device.type=='cuda' and x.weight.quant_state is not None for x in quant)
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.eval()
    def generate(prompt,cap):
        text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        enc=tok(text,add_special_tokens=False,return_tensors='pt').to('cuda')
        torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode(): g=model.generate(**enc,max_new_tokens=cap,do_sample=False,max_time=policy['inference']['max_time_seconds'],use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize();latency=time.monotonic()-start;ids=g[0,enc['input_ids'].shape[1]:].tolist()
        return dict(prediction=tok.decode(ids,skip_special_tokens=True).strip(),finish_reason='eos' if ids and ids[-1]==tok.eos_token_id else 'length' if len(ids)>=cap else 'time_or_other',generated_tokens=len(ids),input_tokens=enc['input_ids'].shape[1],latency_seconds=latency,token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    if smoke_only:
        with model.disable_adapter():s0=generate('8과 7의 합은? 숫자만 답하라.',32)
        s1=generate('8과 7의 합은? 숫자만 답하라.',32)
        put(out/'runtime_smoke.json',dict(at=time.time(),base=s0,adapter=s1,quantized_layers=len(quant),gpu=torch.cuda.get_device_name(0),peak_vram=torch.cuda.max_memory_allocated(),not_final_evaluation=True))
        print('NF4_SMOKE_PASSED',flush=True);return
    runtime=dict(at=time.time(),seal_sha256=sha(root/'SEALED.json'),runner_sha256=sha(__file__),base_code_sha256=sha(inspect.getfile(type(base))),gpu=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,quantized_layers=len(quant),quantization='NF4',compute_dtype='BF16',packages={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','bitsandbytes']},no_training=True,base_revision=seal['base_revision'])
    existing=out/'runtime.json'
    if existing.exists():
        old=read(existing)
        for k in ['seal_sha256','runner_sha256','base_code_sha256','packages']:assert runtime[k]==old[k],k
    else:put(existing,runtime)
    qs=[json.loads(s) for s in (root/'inference.questions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(set(q)=={'id','cohort','family','prompt'} for q in qs) and len(qs)==360
    output=out/'responses.jsonl';done=set()
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);pair=(r['id'],r['model']);assert pair not in done;done.add(pair)
    with output.open('a',encoding='utf-8') as f:
        for i,q in enumerate(qs):
            order=['S0','ARGOS'] if i%2==0 else ['ARGOS','S0']
            for name in order:
                if (q['id'],name) in done:continue
                if (root/'STOP_REQUESTED').exists():raise RuntimeError('Explicit stop marker')
                ctx=model.disable_adapter() if name=='S0' else contextlib.nullcontext()
                with ctx:r=generate(q['prompt'],policy['inference']['max_new_tokens'])
                r.update(id=q['id'],cohort=q['cohort'],family=q['family'],model=name,at=time.time())
                f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());done.add((q['id'],name))
                put(out/'progress.json',dict(at=time.time(),stage='inference',completed=len(done),required=720,current_id=q['id'],peak_vram_bytes=torch.cuda.max_memory_allocated()))
                if len(done)%20==0:print(f'INFERENCE {len(done)}/720',flush=True)
    assert len(done)==720
    put(out/'DONE.json',dict(at=time.time(),completed=720,responses_sha256=sha(output),seal_sha256=sha(root/'SEALED.json'),peak_vram_bytes=torch.cuda.max_memory_allocated(),training_performed=False))
    print('CONFIRMATION_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--adapter',required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    try:run(a.root,a.adapter,a.smoke_only)
    except Exception as ex:
        put(Path(a.root)/'runtime_failure.json',dict(at=time.time(),error_type=type(ex).__name__,error=str(ex)));raise
