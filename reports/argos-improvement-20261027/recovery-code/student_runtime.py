"""Pinned NF4/BF16 training and question-only generation for this new phase only."""
import argparse,gc,hashlib,importlib.metadata,inspect,json,os,time,traceback,math,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def rows(p):return [json.loads(s) for s in p.read_text().split('\n') if s.strip()]
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def check(root):
    req=read(root/'runtime.json')
    if time.time()>req['deadline']-180 or (root/'STOP_REQUESTED').exists():raise TimeoutError('New phase bounded stop')
    lease=root/'controller_lease.json'
    if not lease.exists() or time.time()-read(lease)['at']>600:raise TimeoutError('Local controller lease expired')
    return req

def load(root,training=False,initialization='selected',adapter=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model,LoraConfig,TaskType
    from bitsandbytes.nn import Linear4bit
    from exaone_compat import configure_exaone_embeddings
    req=check(root)
    assert shutil.disk_usage(root).free>=8*1024**3,'Insufficient free disk reserve'
    packages={k:importlib.metadata.version(k) for k in req['expected_packages']}
    assert packages==req['expected_packages'],packages
    assert torch.cuda.is_available() and torch.__version__.startswith('2.8.0')
    assert torch.cuda.get_device_properties(0).total_memory>=23*1024**3
    tok=AutoTokenizer.from_pretrained(root/'adapter',local_files_only=True,trust_remote_code=True)
    assert digest(dict(template=tok.chat_template,vocab=tok.get_vocab()))==req['tokenizer_fingerprint']
    base=AutoModelForCausalLM.from_pretrained(req['base_model'],revision=req['base_revision'],trust_remote_code=True,token=False,
      dtype=torch.bfloat16,device_map='auto',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert sha(Path(inspect.getfile(type(base))))==req['base_code_sha256']
    compatibility=configure_exaone_embeddings(base);base.loss_type='ForCausalLM'
    quant=[m for m in base.modules() if isinstance(m,Linear4bit)]
    assert getattr(base,'is_loaded_in_4bit',False) and len(quant)==224
    assert all(m.weight.device.type=='cuda' and m.weight.quant_state is not None for m in quant)
    base.config.use_cache=not training
    if training:
        base=prepare_model_for_kbit_training(base)
        if initialization=='selected':model=PeftModel.from_pretrained(base,root/'adapter',is_trainable=True)
        else:
            assert initialization=='base'
            model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','out_proj'],bias='none'))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in n for n,p in trainable)
        trainable_count=sum(p.numel() for n,p in trainable)
    else:
        adapter_integrity={}
        if adapter:
            for filename in ['adapter_config.json','adapter_model.safetensors']:
                adapter_integrity[filename]=sha(adapter/filename)
            if 'training' in adapter.parts:
                integrity=read(adapter/'integrity.json')
                assert all(integrity['files'][k]==v for k,v in adapter_integrity.items()),'Candidate checkpoint integrity mismatch'
        model=PeftModel.from_pretrained(base,adapter,is_trainable=False) if adapter else base
        model.eval();trainable_count=0
    return model,tok,dict(packages=packages,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,quantization='NF4',compute_dtype='BF16',quantized_linear_modules=len(quant),compatibility=compatibility,trainable_parameters=trainable_count,base_revision=req['base_revision'],adapter_input_sha256=adapter_integrity if not training else None)

def train(root,name,initialization,lr,seed,steps,save_steps=40):
    import torch
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
    from future_training_tokens import compose
    assert name.replace('-','').replace('_','').isalnum() and 1<=steps<=160
    assert 1<=save_steps<=steps and 0<lr<=2e-4
    assert name=='R6' and initialization=='base' and lr==1e-5 and steps==120 and save_steps==40,'Single frozen recovery run only'
    out=root/'training'/name
    assert not out.exists(),'Never overwrite or silently resume a prior candidate'
    out.mkdir(parents=True);started=time.time();check(root);set_seed(seed)
    model,tok,runtime=load(root,True,initialization)
    examples=rows(root/('train_'+name+'.jsonl'));items=[];supervised=0
    assert sha(root/('train_'+name+'.jsonl'))==read(root/('train_'+name+'.receipt.json'))['sha256']
    for row in examples:
        prefix=tok.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=False,add_generation_prompt=True)
        pi=tok(prefix,add_special_tokens=False)['input_ids'];ci=tok(row['target']+tok.eos_token,add_special_tokens=False)['input_ids']
        item=compose(pi,ci,max_length=1536);assert ci[-1]==tok.eos_token_id
        items.append(item);supervised+=len(ci)
    class Data:
        def __len__(self):return len(items)
        def __getitem__(self,i):return items[i]
    def collate(batch):
        longest=max(len(r['input_ids']) for r in batch)
        return {k:torch.tensor([r[k]+[(-100 if k=='labels' else tok.pad_token_id if k=='input_ids' else 0)]*(longest-len(r[k])) for r in batch],dtype=torch.long) for k in ['input_ids','attention_mask','labels']}
    class Durable(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kw):
            if logs:
                for name in ['loss','grad_norm']:
                    if name in logs and not math.isfinite(float(logs[name])):raise FloatingPointError('Nonfinite '+name)
        def on_step_end(self,args,state,control,**kw):
            check(root)
            atomic(out/'progress.json',dict(at=time.time(),phase='training',step=state.global_step,required=steps,elapsed=time.time()-started))
            return control
        def on_save(self,args,state,control,**kw):
            p=Path(args.output_dir)/('checkpoint-'+str(state.global_step))
            files={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
            assert 'adapter_model.safetensors' in files and 'adapter_config.json' in files
            atomic(p/'integrity.json',dict(at=time.time(),files=files,step=state.global_step))
    atomic(out/'STARTED.json',dict(at=started,name=name,initialization=initialization,learning_rate=lr,seed=seed,steps=steps,save_steps=save_steps,runtime=runtime,rows=len(items),supervised_tokens_one_pass=supervised,training_data_sha256=sha(root/('train_'+name+'.jsonl')),old_artifacts_changed=False))
    args=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=steps,learning_rate=lr,per_device_train_batch_size=1,gradient_accumulation_steps=8,
       bf16=True,fp16=False,seed=seed,data_seed=seed,logging_steps=1,logging_nan_inf_filter=False,save_strategy='steps',save_steps=save_steps,save_total_limit=3,
       report_to='none',disable_tqdm=True,gradient_checkpointing=True,dataloader_num_workers=0,warmup_ratio=.03,weight_decay=0,
       optim='adamw_torch',max_grad_norm=1.0,lr_scheduler_type='linear')
    trainer=Trainer(model=model,args=args,train_dataset=Data(),data_collator=collate,callbacks=[Durable()])
    atomic(out/'TRAINER_CONFIG.json',dict(at=time.time(),training_arguments=args.to_dict(),model_accepts_loss_kwargs=trainer.model_accepts_loss_kwargs,prompt_masked=True,EOS_supervised=True,max_length=1536,sequence_truncation=0))
    trainer.train();trainer.state.save_to_json(str(out/'trainer_state.json'))
    adapter=out/'adapter';model.save_pretrained(adapter);tok.save_pretrained(adapter)
    assert trainer.state.global_step==steps
    from safetensors import safe_open
    with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        names=list(f.keys());assert names
        for key in names:assert bool(torch.isfinite(f.get_tensor(key)).all())
    atomic(out/'TRAINING_DONE.json',dict(at=time.time(),name=name,steps=steps,seed=seed,learning_rate=lr,initialization=initialization,
      elapsed=time.time()-started,adapter_sha256={p.name:sha(p) for p in adapter.iterdir() if p.is_file()},finite_tensor_count=len(names),peak_vram_bytes=torch.cuda.max_memory_allocated(),runtime=runtime,
      adapter_reloading_verified=False,notice='Training complete and tensors finite; separate inference process must verify reloading.'))
    print('TRAINING_DONE '+name,flush=True)
