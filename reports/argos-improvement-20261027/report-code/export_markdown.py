"""Export the exact same narrative, evidence tables and figures for GitHub."""
from pathlib import Path
import argparse,json
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.data.read_text(encoding='utf-8'))
def cell(x):return str('미측정'if x is None else x).replace('|','\\|').replace('\n','<br>')
parts=['# '+d['title'],'',d['report']['subtitle'],'',d['report']['introduction'],'','기준일: '+d['report']['asOf'],'','[PDF 최종보고서](ARGOS_Improvement_Report_KO.pdf) · [오프라인 HTML](report.html) · [공개 근거 자료](public-data/) · [재계산 코드](code/recompute_public_report.py)','',d['report']['peerReviewNote'],'','## 목차','']
for i,s in enumerate(d['report']['sections'],1):parts.append(f"{i}. [{s['title']}](#section-{i:02d})")
for i,s in enumerate(d['report']['sections'],1):
 if s.get('chapter'):parts+=['','## '+s['chapter'],'']
 parts+=['',f'<a id="section-{i:02d}"></a>','',f"### {i:02d}. {s['title']}",'',s['body'],'']
 if s.get('figure'):parts+=['![물리적 실행·평가 기록에 근거한 연구 그림]('+s['figure']+')','',s.get('figureCaption',''),'']
 q=d['queries'].get(s.get('queryId'),{});cols=s.get('columns',[])
 if cols and q.get('rows'):
  parts+=['| '+' | '.join(cell(c['label'])for c in cols)+' |','| '+' | '.join('---:'if c.get('numeric')else'---'for c in cols)+' |']
  parts+=['| '+' | '.join(cell(r.get(c['field']))for c in cols)+' |'for r in q['rows']];parts.append('')
 if q.get('source',{}).get('label'):parts+=['근거: '+q['source']['label']+' · 공개 데이터 키: `'+s['queryId']+'`','']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(parts).rstrip()+'\n',encoding='utf-8');print('Exported',len(d['report']['sections']),'sections to',a.output)
