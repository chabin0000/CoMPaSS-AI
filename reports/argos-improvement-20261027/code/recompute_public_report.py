"""Recompute published aggregates from recorded AI judgments (not a new adjudication).

Run from a published report directory: python code/recompute_public_report.py
Only Python's standard library is required. No GPU, credentials, or final-test keys.
"""
from pathlib import Path
import json, math, statistics, hashlib
from collections import Counter

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'public-data'

def read(name):return json.loads((DATA/name).read_text(encoding='utf-8'))
def rows(name):return [json.loads(line)for line in (DATA/name).read_text(encoding='utf-8').splitlines()if line.strip()]


def audit_check(docs):
    by={(x['model'],x['id']):x for x in docs};claims=rows('claim_evidence.jsonl');ca=read('claim_aggregate.json')
    assert len(claims)==120 and len({x['review_id']for x in claims})==120
    count=0
    for m,expected in ca['table'].items():
        group=[x for x in claims if x['model']==m];assert len(group)==24
        counts=Counter()
        for x in group:
            d=by[(m,x['id'])]
            assert x['response']==d['response'] and x['question']==d['question']
            assert hashlib.sha256(x['response'].encode()).hexdigest()==x['response_sha256']
            assert len({c['text']for c in x['claims']})==len(x['claims'])
            for c in x['claims']:
                assert c['label']in ['supported','contradicted','unverifiable'];counts[c['label']]+=1
        for label in ['supported','contradicted','unverifiable']:assert counts[label]==expected[label]
        total=sum(counts.values());assert total==expected['claims']
        assert expected['contradicted_fraction']==counts['contradicted']/total
        assert expected['unsupported_fraction']==(counts['contradicted']+counts['unverifiable'])/total
        assert expected['complete']==sum(x['response_complete']for x in group)
        count+=total
    pairs=rows('pair_evidence.jsonl');pa=read('pair_aggregate.json');assert len(pairs)==72
    labels=['A','B','tie'];ratings={r:{}for r in ['Codex','ProjectPro']}
    assert len({p['question_id']for p in pairs})==24
    for p in pairs:
        assert len(p['judgments'])==2
        for side in ['A','B']:
            assert p[side]==by[(p[side+'_model'],p['question_id'])]['response']
        for j in p['judgments']:
            assert j['overall']in labels and j['naturalness']in labels
            ratings[j['rater']][p['review_id']]=j
    assert all(len(v)==72 for v in ratings.values())
    assert max(x['at']for x in ratings['Codex'].values())<min(x['at']for x in ratings['ProjectPro'].values())
    for metric in ['overall','naturalness']:
        first=[ratings['Codex'][p['review_id']][metric]for p in pairs];second=[ratings['ProjectPro'][p['review_id']][metric]for p in pairs]
        x,y=Counter(first),Counter(second);matched=sum(i==j for i,j in zip(first,second));n=len(pairs)
        chance=sum(x[k]*y[k]for k in labels)/n**2;expected=pa['agreement'][metric]
        assert matched==expected['agree'] and math.isclose(100*matched/n,expected['percent_agreement'],abs_tol=1e-12)
        kappa=(matched/n-chance)/(1-chance)if chance<1 else None
        assert kappa==expected['cohen_kappa']
        assert all(expected['marginals_first'][k]==x[k]and expected['marginals_second'][k]==y[k]for k in labels)
    for rater in ratings:
        for comparator in ['S0','gpt_oss','glm']:
            group=[p for p in pairs if p['comparator']==comparator];assert len(group)==24
            for metric in ['overall','naturalness']:
                actual=Counter()
                for p in group:
                    vote=ratings[rater][p['review_id']][metric]
                    actual['tie'if vote=='tie'else'candidate'if p[vote+'_model']==p['candidate']else'comparator']+=1
                assert all(actual[k]==pa['preferences'][rater][comparator][metric][k]for k in ['candidate','comparator','tie'])
    return dict(claim_responses=len(claims),atomic_claims=count,pairs=len(pairs),AI_pair_judgments=144)

def verify():
    a=read('development_analysis.json');docs=rows('synthetic_development_evidence.jsonl');public=rows('public_retention_scores.jsonl')
    assert len(docs)==960 and len(public)==800
    assert len({(r['model'],r['id'])for r in docs})==960
    assert len({(r['model'],r['id'])for r in public})==800
    common={};summaries=[]
    for model in a['models']:
        name=model['model'];d=[r for r in docs if r['model']==name];g=[r for r in public if r['model']==name and 'gsm8k-'in r['id']];k=[r for r in public if r['model']==name and 'kobest_boolq-'in r['id']]
        assert (len(d),len(g),len(k))==(96,40,40)
        for r in d:
            payload=(r['question'],r['reference'],r['reference_rationale'],r['family'])
            assert common.setdefault(r['id'],payload)==payload
            assert not r['rationale_valid'] or (r['semantic_correct']and r['complete'])
            assert not r['semantic_correct']or r['complete']
        values=dict(document_correct=sum(r['semantic_correct']for r in d),valid_explanations=sum(r['rationale_valid']is True for r in d),gsm8k_correct=sum(r['official_correct']for r in g),kobest_official=sum(r['official_correct']for r in k),kobest_corrected=sum(r['corrected_reference_correct']for r in k))
        assert all(model[field]==value for field,value in values.items())
        for r in g+k:
            assert not r['explanation_assessed'] and not r['question_text_redistributed']
            assert not r['official_correct']or r['complete']
            if r['id']!='AUX-kobest_boolq-143':assert r['official_correct']==r['corrected_reference_correct']
        for f in [r for r in a['families']if r['model']==name]:
            group=[r for r in d if r['family']==f['family']]
            assert f['n']==len(group)==8
            assert f['correct']==sum(r['semantic_correct']for r in group)
            assert f['valid_explanations']==sum(r['rationale_valid']is True for r in group)
        if model['qualified'] is not None:
            counts=dict(document=values['document_correct'],valid_document_explanations=values['valid_explanations'],gsm8k=values['gsm8k_correct'],kobest_official=values['kobest_official'],kobest_corrected=values['kobest_corrected'])
            assert model['qualified']==all(counts[key]>=limit for key,limit in a['gates'].items())
        timing=next(r for r in a['document_timings']if r['model']==name)
        times=sorted(r['latency_seconds']for r in d)
        assert timing['median_seconds']==statistics.median(times)
        assert timing['p90_seconds_nearest_rank']==times[math.ceil(.9*len(times))-1]
        assert timing['completed']==sum(r['complete']for r in d)
        summaries.append(dict(model=name,**values))
    for p in a['paired_outcomes']:
        left={r['id']:r for r in docs if r['model']==p['candidate']};right={r['id']:r for r in docs if r['model']==p['comparator']}
        ids=[i for i,r in left.items()if p['scope']=='all_document'or not(r['family']=='exchange'and str(r['reference'])=='0')]
        assert len(ids)==p['n']
        actual=dict(both_correct=0,candidate_only=0,comparator_only=0,both_wrong=0)
        for i in ids:
            x,y=left[i]['semantic_correct'],right[i]['semantic_correct']
            actual['both_correct'if x and y else'candidate_only'if x else'comparator_only'if y else'both_wrong']+=1
        assert all(p[key]==value for key,value in actual.items())
    assert a['formally_selected']is None and not any(r['qualified']for r in a['models'])
    assert a['teacher_superiority_demonstrated']is False and a['human_reviewers']==0
    audits=audit_check(docs)
    print(json.dumps(dict(status='verified',models=summaries,document_responses=len(docs),retention_responses=len(public),**audits,scope='Recomputed recorded judgments and descriptive statistics; does not independently establish semantic validity, human review, causality, or teacher superiority.'),ensure_ascii=False,indent=2))

if __name__=='__main__':verify()
