"""Recompute R6 aggregates and frozen selection using Python's standard library.

Run: python recovery-code/recompute_round.py
Recorded AI judgments are evidence inputs, not independently re-adjudicated here.
"""
from pathlib import Path
import json,hashlib,collections,statistics
ROOT=Path(__file__).resolve().parent.parent;DATA=ROOT/'public-data'
def read(name):return json.loads((DATA/name).read_text(encoding='utf-8'))
def rows(name):return [json.loads(s)for s in (DATA/name).read_text(encoding='utf-8').split('\n')if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def verify():
    p=read('recovery_protocol.json');s=read('recovery_selection.json');e=rows('recovery_development_evidence.jsonl')
    resolved=rows('recovery_development_resolved.jsonl');reviews=rows('recovery_development_reviews.jsonl')
    assert sha(DATA/'recovery_protocol.json')==s['protocol_sha256']
    assert sha(DATA/'recovery_development_reviews.jsonl')==s['review_sha256']
    assert sha(DATA/'recovery_development_resolved.jsonl')==s['resolved_sha256']
    assert sha(DATA/'recovery_train_receipt.json')==p['train_receipt_sha256']
    assert sha(DATA/'recovery_seal.json')==p['seal_sha256']
    assert len(e)==len(resolved)==704 and len({(r['model'],r['id'])for r in e})==704
    by={(r['model'],r['id']):r for r in e};review_by={(r['model'],r['id']):r for r in reviews}
    assert len(review_by)==len(reviews)==428
    for r in resolved:
        published=by[(r['model'],r['id'])]
        assert all(published[k]==v for k,v in r.items())
        assert textsha(r['prediction'])==r['response_sha256']and r['complete']and r['human_reviewers']==0
        if r['cohort']=='document':
            assert isinstance(r['rationale_valid'],bool)
            assert not r['rationale_valid']or r['semantic_correct']
            assert published['prompt_variant']=='verify'and textsha(published['question'])==published['prompt_sha256']
        else:assert published['question']is None
        if not r['auto_correct']:assert (r['model'],r['id'])in review_by
    tables={}
    for model in ['S0','R6-040','R6-080','R6-120']:
        g=[r for r in e if r['model']==model];assert len(g)==176
        t={}
        for c,n in [('document',96),('gsm8k',40),('kobest_boolq',40)]:
            group=[r for r in g if r['cohort']==c];assert len(group)==n
            t[c]=sum(r['semantic_correct']for r in group)
        t['valid_explanations']=sum(r['rationale_valid']is True for r in g if r['cohort']=='document')
        t['kobest_corrected']=sum(r['corrected_correct']for r in g if r['cohort']=='kobest_boolq')
        assert all(s['tables'][model][k]==v for k,v in t.items())
        assert s['tables'][model]['explanations_unreviewed']==0
        tables[model]=t
    floor=p['selection_historical_floors'];base=tables['S0']
    limits=dict(document=max(base['document']+5,floor['document_correct']),gsm8k=max(base['gsm8k'],floor['gsm8k_correct']),
      kobest_boolq=max(base['kobest_boolq'],floor['kobest_official']),kobest_corrected=max(base['kobest_corrected'],floor['kobest_corrected']),
      valid_explanations=max(base['valid_explanations'],floor['valid_document_explanations']))
    assert limits==s['thresholds'];eligible=[]
    for model,t in tables.items():
        if model=='S0':continue
        gates={k:t[k]>=v for k,v in limits.items()};assert gates==s['tables'][model]['gate_passes']
        if all(gates.values()):eligible.append(model)
    eligible.sort(key=lambda m:(-tables[m]['document'],-tables[m]['gsm8k'],int(m.split('-')[1])))
    assert eligible==s['eligible']==[]and s['selected']is None and not s['final_allowed']
    historical=rows('synthetic_development_evidence.jsonl')
    for f in read('recovery_family_scores.json')['rows']:
        for model in tables:
            group=[r for r in e if r['model']==model and r['cohort']=='document'and r['family']==f['family']]
            assert len(group)==8 and sum(r['semantic_correct']for r in group)==f[model]
        for model in ['ARGOS_old','B4-075','gpt_oss','glm']:
            group=[r for r in historical if r['model']==model and r['family']==f['family']]
            assert len(group)==8 and sum(r['semantic_correct']for r in group)==f[model]
    for r in read('recovery_paired_development.json')['rows']:
        g=[x for x in e if x['model']==r['model']and x['cohort']==r['cohort']]
        assert len(g)==r['n']
        assert sum(x['semantic_correct']and not by[('S0',x['id'])]['semantic_correct']for x in g)==r['model_only']
        assert sum(by[('S0',x['id'])]['semantic_correct']and not x['semantic_correct']for x in g)==r['base_only']
        assert r['model_only']-r['base_only']==r['model_correct']-r['base_correct']
        assert statistics.mean(x['generated_tokens']for x in g)==r['mean_tokens']
    diag=rows('recovery_diagnostic_evidence.jsonl');assert len(diag)==224 and len({r['id']for r in diag})==224
    for r in read('recovery_adapter_diagnosis.json')['semantic_table']:
        g=[x for x in diag if x['scale']==r['scale']and x['cohort']==r['cohort']]
        assert len(g)==r['n']and sum(x['semantic_correct']for x in g)==r['correct']
    training=read('recovery_training_summary.json');receipt=read('recovery_train_receipt.json')
    assert training['training_runs']==1 and training['steps']==120 and receipt['unique_rows']==960
    assert training['training_data_sha256']==receipt['sha256']
    assert sum(v['supervised_tokens']for v in receipt['by_source'].values())==98092==training['supervised_tokens']
    assert sum(v['rows']for v in receipt['by_source'].values())==960
    assert [c['step']for c in training['checkpoints']]==[40,80,120]
    frozen=read('recovery_freeze.json');assert frozen['selected']is None and frozen['final_responses']=={}
    assert sum(frozen['development_and_diagnostic_responses'].values())==len(e)+len(diag)==928
    cost=read('recovery_cost_and_stop.json');assert cost['stop_confirmed']and cost['verified_backup']
    assert 0<cost['round_account_debit_USD']<2 and 0<cost['original_anchor_account_debit_USD']<10
    for name,digest in read('recovery_source_manifest.json')['files'].items():assert sha(ROOT/'recovery-code'/name)==digest
    old_total=sum(j.get('records')or 0 for j in read('execution_ledger.json')['jobs'])
    assert old_total==3128 and old_total+928==4056
    report=read('report_data.json');assert report['buildStatus']=='complete'
    assert len(report['report']['sections'])==51
    current=report['queries']['current-results']['rows']
    for model,t in tables.items():
        r=next(x for x in current if x['model']==model)
        assert (r['document'],r['valid'],r['gsm8k'],r['kobest'],r['corrected'])==(t['document'],t['valid_explanations'],t['gsm8k'],t['kobest_boolq'],t['kobest_corrected'])
    result=dict(status='verified',development_responses=704,diagnostic_responses=224,document_judgments=384,new_candidate_document_reviews=288,
      total_recorded_reviews=428,additional_training_runs=1,steps=120,selected=None,final_inference=0,all_gates_recomputed=True,
      original_phase_standard_responses=old_total,combined_standard_responses=4056,human_reviewers=0,
      tables=tables,scope='Recomputed recorded judgments and integrity links, not independent semantic adjudication, human review, or final-test evidence.')
    (DATA/'recovery_aggregate_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':verify()
