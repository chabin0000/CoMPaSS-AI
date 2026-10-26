"""Transparent confirmation scoring; subjective judgments must be supplied separately."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import argparse, csv, hashlib, json, math, random, re, time

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(p,v):Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in v),encoding='utf-8')

def numeric(text):
    """Extract only explicit final-answer declarations or a standalone rational."""
    t=text.strip().replace('−','-').replace('**','').replace('\\,','').replace(',','')
    t=re.sub(r'\\(?:d?frac)\s*\{\s*(-?\d+)\s*\}\s*\{\s*(\d+)\s*\}',r'\1/\2',t)
    hits=re.findall(r'(?:정답|Answer|답)\s*(?:은|는)?\s*[:：=]\s*\$?\s*(?:\\boxed\{)?\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)',t,re.I)
    if hits:
        try:
            vals=[Fraction(x.replace(' ','')) for x in hits]
            return str(vals[-1]) if all(x==vals[-1] for x in vals) else None
        except (ValueError,ZeroDivisionError):return None
    plain=t.strip('$ {}\\')
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?',plain):
        try:return str(Fraction(plain.replace(' ','')))
        except (ValueError,ZeroDivisionError):pass
    return None

def label(text,labels):
    t=text.strip().replace('**','').strip('`').strip()
    for s in sorted(labels,key=len,reverse=True):
        if re.match(r'^(?:(?:정답|답)\s*[:：]\s*)?'+re.escape(s)+r'(?:$|[\s.,。:：])',t):return s
    return None

def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def structural_instruction(text,cohort):
    """Literal requested output shape; presence of a reason is not proof validity."""
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if cohort in ['math','gsm8k']:
        marker='정답' if cohort=='math' else 'Answer'
        number=r'[-+]?\d+(?:\s*/\s*\d+)?' if cohort=='math' else r'[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?'
        return len(lines)>=2 and re.fullmatch(marker+r'\s*[:：]\s*'+number,lines[-1],re.I) is not None
    if cohort=='premise':
        return len(lines)==2 and lines[0] in ['지지','미지지'] and len(re.findall(r'[.!?](?:\s|$)',lines[1]))<=1
    raise ValueError(cohort)

def prepare_review(root):
    root=Path(root);done=read(root/'inference/DONE.json');assert sha(root/'inference/responses.jsonl')==done['responses_sha256']
    out=root/'blind_review';out.mkdir(exist_ok=False)
    q={r['id']:r for r in rows(root/'inference.questions.jsonl')};k={r['id']:r for r in rows(root/'private.keys.LOCAL_ONLY.jsonl')+rows(root/'aux.keys.LOCAL_ONLY.jsonl')}
    resp=rows(root/'inference/responses.jsonl');rng=random.Random(9463107);rng.shuffle(resp);mapping={};packets=[];unresolved=[]
    for r in resp:
        pid=hashlib.sha256((r['id']+r['model']+'blind-publication').encode()).hexdigest()[:16]
        mapping[pid]=dict(id=r['id'],model=r['model'],response_sha256=hashlib.sha256(r['prediction'].encode()).hexdigest())
        packet=dict(packet_id=pid,question=q[r['id']]['prompt'],cohort=r['cohort'],prediction=r['prediction'],finish_reason=r['finish_reason'],reference=k[r['id']])
        if r['cohort']=='source_claims':packets.append(packet)
        elif r['cohort'] in ['math','gsm8k'] and r['finish_reason']=='eos' and numeric(r['prediction']) is None:unresolved.append(packet)
    jsonl(out/'source_claim_packets.LOCAL_ONLY.jsonl',packets);jsonl(out/'numeric_unresolved.LOCAL_ONLY.jsonl',unresolved)
    lookup={(r['id'],r['model']):r for r in resp};pairs=[];pairmap={}
    for cohort in ['source_claims','premise']:
        ids=sorted([i for i,r in q.items() if r['cohort']==cohort],key=lambda i:hashlib.sha256(('pair-sample-'+i).encode()).hexdigest())[:20]
        for i in ids:
            order=['S0','ARGOS'];rng.shuffle(order);pid=hashlib.sha256(('pair-'+i).encode()).hexdigest()[:16];pairmap[pid]=dict(id=i,A=order[0],B=order[1])
            pairs.append(dict(packet_id=pid,cohort=cohort,question=q[i]['prompt'],reference=k[i],A=lookup[i,order[0]]['prediction'],B=lookup[i,order[1]]['prediction']))
    rng.shuffle(pairs);jsonl(out/'pair_packets.LOCAL_ONLY.jsonl',pairs);put(out/'PRIVATE_MAP.json',dict(scalar=mapping,pairwise=pairmap))
    put(out/'preparation.json',dict(at=time.time(),response_sha256=done['responses_sha256'],source_claim_packets=len(packets),numeric_unresolved=len(unresolved),pair_packets=len(pairs),sampling='lowest SHA256(pair-sample-ID) per prespecified cohort, 20 each; output-independent',reviewer_count=1,human_reviewers=0,analysis_script_sha256=sha(__file__)))
    print(json.dumps(dict(source_claim_packets=len(packets),numeric_unresolved=len(unresolved),pair_packets=len(pairs))))

def summarize(root):
    import numpy as np
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    review=root/'blind_review';assert (review/'REVIEWS_CLOSED.json').exists()
    mapping=read(review/'PRIVATE_MAP.json');claims=read(review/'source_judgments.LOCAL_ONLY.json');num=read(review/'numeric_judgments.LOCAL_ONLY.json');pair=read(review/'pair_judgments.LOCAL_ONLY.json')
    rr=rows(root/'inference/responses.jsonl');q={r['id']:r for r in rows(root/'inference.questions.jsonl')};keys={r['id']:r for r in rows(root/'private.keys.LOCAL_ONLY.jsonl')+rows(root/'aux.keys.LOCAL_ONLY.jsonl')}
    reverse={(v['id'],v['model']):pid for pid,v in mapping['scalar'].items()};scored=[];claimtable=[]
    for r in rr:
        k=keys[r['id']];pid=reverse[r['id'],r['model']];t=r['prediction'].strip();cohort=r['cohort'];complete=r['finish_reason']=='eos';semantic=None;instruction=False;method=''
        if complete:
            if cohort in ['math','gsm8k']:
                extracted=numeric(t);method='explicit_numeric'
                if extracted is None:
                    assert pid in num,pid;extracted=num[pid]['value'];method='single_AI_semantic_numeric'
                semantic=(Fraction(extracted)==Fraction(k['answer'].replace(',',''))) if extracted is not None else None
                if extracted is None and pid in num and 'semantic_correct' in num[pid]:semantic=num[pid]['semantic_correct']
                instruction=structural_instruction(t,cohort)
            elif cohort=='source_claims':
                j=claims[pid];semantic=all(j['required'][field]=='supported' for field in k['answer']);instruction=j['instruction'];method='single_AI_atomic_claim_review'
                for c in j['claims']:claimtable.append(dict(id=r['id'],model=r['model'],**c))
            elif cohort=='premise':
                extracted=label(t,['지지','미지지']);semantic=extracted==k['answer'] if extracted is not None else None;instruction=structural_instruction(t,cohort);method='first_label'
            elif cohort=='instruction':
                semantic=t.replace(' ','')==k['answer'].replace(' ','');method='exact_output'
                if k['family']=='instruction_2':instruction=bool(re.fullmatch(r'\d+',t))
                elif k['family']=='instruction_3':instruction=len(t.split(','))==4 and set(t.split(','))=={'가람','누리','도담','라온'}
                else:instruction=t in ['가람','누리','도담','라온']
            else:
                options=['함의','중립','모순'] if cohort=='klue_nli' else ['예','아니오'];extracted=label(t,options);semantic=extracted==k['answer'] if extracted is not None else None;instruction=t in options;method='first_label'
        scored.append(dict(id=r['id'],cohort=cohort,family=r['family'],model=r['model'],complete=complete,semantic=semantic,instruction=instruction,joint=semantic is True and instruction,method=method,latency_seconds=r['latency_seconds'],generated_tokens=r['generated_tokens']))
    groups=defaultdict(list)
    for r in scored:groups[r['cohort'],r['model']].append(r)
    metrics=[]
    for (cohort,model),rs in sorted(groups.items()):
        n=len(rs);success=sum(r['semantic'] is True for r in rs)
        metrics.append(dict(cohort=cohort,model=model,n=n,complete=sum(r['complete'] for r in rs),semantic_correct=success,unknown=sum(r['semantic'] is None for r in rs),semantic_rate=success/n,wilson95=wilson(success,n),instruction=sum(r['instruction'] for r in rs),joint=sum(r['joint'] for r in rs)))
    family_metrics=[]
    fg=defaultdict(list)
    for r in scored:fg[r['cohort'],r['family'],r['model']].append(r)
    for (cohort,family,model),rs in sorted(fg.items()):
        family_metrics.append(dict(cohort=cohort,family=family,model=model,n=len(rs),correct=sum(r['semantic'] is True for r in rs),unknown=sum(r['semantic'] is None for r in rs)))
    comparisons=[];rng=np.random.default_rng(20261026)
    for cohort in sorted({r['cohort'] for r in scored}):
        a={r['id']:r for r in groups[cohort,'S0']};b={r['id']:r for r in groups[cohort,'ARGOS']};ids=sorted(a);assert set(a)==set(b)
        delta=np.array([int(b[i]['semantic'] is True)-int(a[i]['semantic'] is True) for i in ids]);samples=delta[rng.integers(0,len(ids),(10000,len(ids)))].mean(axis=1)
        families=sorted({a[i]['family'] for i in ids});family=None
        if len(families)>1:
            sums=np.array([sum(delta[j] for j,i in enumerate(ids) if a[i]['family']==f) for f in families]);sizes=np.array([sum(a[i]['family']==f for i in ids) for f in families]);ix=rng.integers(0,len(families),(10000,len(families)));fm=sums[ix].sum(1)/sizes[ix].sum(1);family=(np.quantile(fm,[.025,.975])*100).tolist()
        win=int((delta==1).sum());loss=int((delta==-1).sum());discord=win+loss;p=min(1,2*sum(math.comb(discord,j) for j in range(min(win,loss)+1))/2**discord) if discord else 1
        comparisons.append(dict(cohort=cohort,n=len(ids),delta_pp=float(delta.mean()*100),item_bootstrap95_pp=(np.quantile(samples,[.025,.975])*100).tolist(),family_bootstrap95_pp=family,families=len(families),wins=win,losses=loss,mcnemar_exact=p,exploratory_multiple_comparisons=True))
    factual=[]
    for model in ['S0','ARGOS']:
        cs=[c for c in claimtable if c['model']==model];counts={x:sum(c['verdict']==x for c in cs) for x in ['supported','contradicted','unsupported','unresolved']};den=counts['supported']+counts['contradicted']
        js=[j for pid,j in claims.items() if mapping['scalar'][pid]['model']==model]
        factual.append(dict(model=model,**counts,emitted_atomic_claims=len(cs),determinate_claims=den,factual_error_rate=counts['contradicted']/den if den else None,unsupported_rate=counts['unsupported']/len(cs) if cs else None,responses_with_claims=len({c['id'] for c in cs}),response_denominator=60,required_facts_supported=sum(v=='supported' for j in js for v in j['required'].values()),required_facts=180,responses_with_unsupported=len({c['id'] for c in cs if c['verdict']=='unsupported'}),reviewer='one AI',scope='Provided fictional records only; not open-world facts'))
    preferences=[]
    for model in ['S0','ARGOS']:
        wins=losses=ties=0;natural=[];reason=[]
        for pid,j in pair.items():
            m=mapping['pairwise'][pid];side='A' if m['A']==model else 'B';w=j['winner'];wins+=w==side;ties+=w=='tie';losses+=w not in [side,'tie'];natural.append(j['naturalness_'+side]);reason.append(j['explanation_'+side])
        natural=[x for x in natural if x is not None]
        preferences.append(dict(model=model,pairs=len(pair),wins=wins,losses=losses,ties=ties,win_rate=wins/len(pair),naturalness_n=len(natural),naturalness_mean=sum(natural)/len(natural) if natural else None,explanation_mean=sum(reason)/len(reason),independent_interrater_kappa=None,human_reviewers=0))
    pair_domains=[]
    for cohort in ['source_claims','premise']:
        for model in ['S0','ARGOS']:
            ws=ls=ts=0
            for pid,j in pair.items():
                m=mapping['pairwise'][pid]
                if q[m['id']]['cohort']!=cohort:continue
                side='A' if m['A']==model else 'B';ws+=j['winner']==side;ts+=j['winner']=='tie';ls+=j['winner'] not in [side,'tie']
            pair_domains.append(dict(cohort=cohort,model=model,wins=ws,losses=ls,ties=ts,n=ws+ls+ts))
    put(out/'metrics.json',metrics);put(out/'family_metrics.json',family_metrics);put(out/'pair_domains.json',pair_domains);put(out/'comparisons.json',comparisons);put(out/'claim_metrics.json',factual);put(out/'qualitative.json',preferences);jsonl(out/'scored.jsonl',scored);jsonl(out/'atomic_claims.LOCAL_ONLY.jsonl',claimtable)
    with (out/'per_item_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scored[0]));w.writeheader();w.writerows(scored)
    put(out/'analysis_receipt.json',dict(at=time.time(),responses=len(rr),script_sha256=sha(__file__),review_closure_sha256=sha(review/'REVIEWS_CLOSED.json'),raw_outputs_changed=False,training_performed=False))
    print(json.dumps(metrics,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-review','summarize']);p.add_argument('--root',required=True);a=p.parse_args()
    (prepare_review if a.action=='prepare-review' else summarize)(a.root)
