"""Fixed reversible LoRA-strength experiment; does not update any weights."""
from pathlib import Path
from contextlib import nullcontext
import hashlib,json,math,time,torch
from student_runtime import load,check,read,rows,atomic,sha

R=Path(__file__).resolve().parent

def main():
    from peft.tuners.lora.layer import LoraLayer
    from transformers import set_seed
    set_seed(20261027);torch.set_num_threads(6)
    out=R/'evaluations/diagnostic-adapter-strength';out.mkdir(parents=True,exist_ok=False)
    model,tok,runtime=load(R,False,adapter=R/'adapter')
    modules=[m for m in model.modules()if isinstance(m,LoraLayer)]
    assert len(modules)==128
    expected=read(R/'old_adapter_manifest.json');assert sha(R/'adapter/adapter_model.safetensors')==expected['adapter_sha256']
    # Compare vendor loss to shifted token CE and its explicit token denominator.
    example=rows(R/'train_R6.jsonl')[0]
    prefix=tok.apply_chat_template([dict(role='user',content=example['prompt'])],tokenize=False,add_generation_prompt=True)
    pi=tok(prefix,add_special_tokens=False)['input_ids'];ci=tok(example['target']+tok.eos_token,add_special_tokens=False)['input_ids']
    ids=torch.tensor([pi+ci],device=next(model.parameters()).device);labels=torch.tensor([[-100]*len(pi)+ci],device=ids.device)
    with model.disable_adapter(),torch.inference_mode():
        result=model(input_ids=ids,labels=labels,num_items_in_batch=len(ci),use_cache=False)
        manual=torch.nn.functional.cross_entropy(result.logits[:,:-1].float().reshape(-1,result.logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/len(ci)
    loss_delta=abs(float(result.loss)-float(manual));assert loss_delta<1e-4
    atomic(out/'loss_audit.json',dict(at=time.time(),model_loss=float(result.loss),manual_shifted_CE=float(manual),absolute_difference=loss_delta,supervised_tokens=len(ci),prompt_masked=True,EOS_supervised=True,scope='One actual GPU forward with model loss compared to manual token-normalized CE; not a full optimizer audit.'))
    del result,manual,ids,labels;torch.cuda.empty_cache()
    qs=rows(R/'diagnostic.questions.jsonl');assert len(qs)==56;began=time.time();count=0
    atomic(out/'runtime.json',dict(at=began,**runtime,question_sha256=sha(R/'diagnostic.questions.jsonl'),scales=[0,.25,.5,1],disabled_adapter_control=True,loss_audit_passed=True))
    target=out/'responses.jsonl'
    with target.open('x')as f:
        for scale in [0,.25,.5,1]:
            for module in modules:module.set_scale('default',scale)
            ctx=model.disable_adapter()if scale==0 else nullcontext()
            with ctx:
                for q in qs:
                    check(R);start=time.monotonic();prompt=tok.apply_chat_template([dict(role='user',content=q['prompt'])],tokenize=False,add_generation_prompt=True)
                    inputs=tok(prompt,add_special_tokens=False,return_tensors='pt').to(next(model.parameters()).device)
                    with torch.inference_mode():gen=model.generate(**inputs,do_sample=False,max_new_tokens=8192,max_time=300,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
                    generated=gen[0,inputs['input_ids'].shape[1]:].tolist();finish='eos'if generated and generated[-1]==tok.eos_token_id else'length_or_time'
                    row=dict(id=q['id']+'--scale-'+str(scale),source_id=q['id'],model='B4-075-scale-'+str(scale),scale=scale,cohort=q['cohort'],family=q['family'],prediction=tok.decode(generated,skip_special_tokens=True).strip(),finish_reason=finish,response_valid=finish=='eos',generated_tokens=len(generated),latency=time.monotonic()-start,prompt_sha256=hashlib.sha256(q['prompt'].encode()).hexdigest())
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();count+=1
                    atomic(out/'progress.json',dict(at=time.time(),completed=count,required=224,scale=scale,current_id=q['id'],elapsed=time.time()-began))
    atomic(out/'DONE.json',dict(at=time.time(),completed=count,responses_sha256=sha(target),elapsed=time.time()-began,weights_updated=False))
    print('DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
