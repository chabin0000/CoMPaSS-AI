"""Render the complete Korean report from its public canonical JSON and figures."""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse,hashlib,json,re,statistics
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether,CondPageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader
from PIL import Image as PILImage

ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
d=json.loads(args.data.read_text(encoding='utf-8'));WIDTH=A4[0]-104;INK=colors.HexColor('#202020');BLUE=INK;GRAY=colors.HexColor('#636363');LIGHT=colors.HexColor('#F5F5F5')
font=Path('C:/Windows/Fonts/malgun.ttf');bold=Path('C:/Windows/Fonts/malgunbd.ttf')
pdfmetrics.registerFont(TTFont('Malgun',str(font)));pdfmetrics.registerFont(TTFont('MalgunBold',str(bold)))
ST={
 'title':ParagraphStyle('title',fontName='MalgunBold',fontSize=28,leading=40,textColor=INK,wordWrap='CJK',spaceAfter=22),
 'subtitle':ParagraphStyle('subtitle',fontName='Malgun',fontSize=12,leading=21,textColor=INK,wordWrap='CJK',spaceAfter=18),
 'chapter':ParagraphStyle('chapter',fontName='MalgunBold',fontSize=16.5,leading=25,textColor=INK,wordWrap='CJK',spaceBefore=17,spaceAfter=15,keepWithNext=True),
 'heading':ParagraphStyle('heading',fontName='MalgunBold',fontSize=12,leading=18.5,textColor=INK,wordWrap='CJK',spaceBefore=14,spaceAfter=9,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='Malgun',fontSize=10.1,leading=16.3,textColor=INK,wordWrap='CJK',spaceAfter=9,allowWidows=0,allowOrphans=0),
 'small':ParagraphStyle('small',fontName='Malgun',fontSize=8,leading=12.2,textColor=GRAY,wordWrap='CJK',spaceAfter=7),
 'coverlink':ParagraphStyle('coverlink',fontName='MalgunBold',fontSize=10.5,leading=15,textColor=colors.HexColor('#225F91'),wordWrap='CJK',spaceBefore=4,spaceAfter=2),
 'coverurl':ParagraphStyle('coverurl',fontName='Malgun',fontSize=8.8,leading=13,textColor=colors.HexColor('#225F91'),wordWrap='CJK',spaceAfter=5),
 'cell':ParagraphStyle('cell',fontName='Malgun',fontSize=8.2,leading=12.4,textColor=INK,wordWrap='CJK',allowWidows=0,allowOrphans=0),
 'headcell':ParagraphStyle('headcell',fontName='MalgunBold',fontSize=8.2,leading=12.4,textColor=INK,wordWrap='CJK'),
 'toc0':ParagraphStyle('toc0',fontName='MalgunBold',fontSize=10.5,leading=16,textColor=INK,spaceBefore=11,spaceAfter=3,wordWrap='CJK'),
 'toc1':ParagraphStyle('toc1',fontName='Malgun',fontSize=9.2,leading=14.5,textColor=INK,leftIndent=12,firstLineIndent=0,spaceBefore=2.5,wordWrap='CJK'),
}
def clean(x):
 if x is None:return'미측정'
 if isinstance(x,(list,dict)):x=json.dumps(x,ensure_ascii=False)
 return str(x).replace('\u2212','-').replace('\u2011','-').replace('\u2013','–').replace('\u2060','')
def markup(x):
 s=escape(clean(x));s=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',r'<link href="\2" color="#225F91">\1</link>',s);s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
 return s.replace('\n','<br/>')
def para(x,style='body'):return Paragraph(markup(x),ST[style])
def column_widths(columns,rs):
 weights=[]
 for c in columns:
  vals=[clean(r.get(c['field']))for r in rs];med=statistics.median(map(len,vals))if vals else 6
  w=1+min(110,med)/22
  if c['field']in ['meaning','note','detail','definition','limit','purpose','finding','description','work']:w=max(w,2.5)
  if c['field']=='sha256':w=3.3
  weights.append(w)
 floor=min(43,WIDTH/len(columns)*.6);remain=WIDTH-floor*len(columns)
 return[floor+remain*w/sum(weights)for w in weights]
def table(columns,rs):
 content=[[para(c['label'],'headcell')for c in columns]]+[[para(r.get(c['field']),'cell')for c in columns]for r in rs]
 t=Table(content,colWidths=column_widths(columns,rs),repeatRows=1,hAlign='LEFT',splitByRow=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5.5),('BOTTOMPADDING',(0,0),(-1,-1),5.5),('LINEABOVE',(0,0),(-1,0),.7,INK),('LINEBELOW',(0,0),(-1,0),.5,INK),('LINEBELOW',(0,1),(-1,-1),.22,colors.HexColor('#D4D4D4')),('LINEBELOW',(0,-1),(-1,-1),.5,INK)]))
 return t
def decorate(c,doc):
 c.saveState();c.setFont('Malgun',7.7);c.setFillColor(GRAY)
 if doc.page>1:
  c.drawString(52,A4[1]-30,'CoMPaSS / ARGOS')
  c.drawRightString(A4[0]-52,A4[1]-30,'제작과 개선 연구 최종보고서')
 c.drawString(52,26,'2026. 10. 27.')
 c.drawCentredString(A4[0]/2,26,str(doc.page));c.restoreState()
class ReportDoc(SimpleDocTemplate):
 def beforeDocument(self):self.section_pages=[];self.counter=0
 def afterFlowable(self,f):
  if isinstance(f,Paragraph)and f.style.name in ['chapter','heading']:
   self.counter+=1;key=f'heading-{self.counter}';level=0 if f.style.name=='chapter'else 1
   self.canv.bookmarkPage(key);self.canv.addOutlineEntry(f.getPlainText(),key,level=level,closed=False)
   self.notify('TOCEntry',(level,f.getPlainText(),self.page,key));self.section_pages.append(dict(level=level,title=f.getPlainText(),page=self.page))

flow=[Spacer(1,30),para('CoMPaSS RESEARCH / FINAL RECORD','small'),Spacer(1,31),para('ARGOS\n제작과 개선 연구\n최종보고서','title'),para(d['report']['subtitle'],'subtitle'),Spacer(1,14),para(d['report']['introduction']),Spacer(1,25),table([{'field':'item','label':'구분'},{'field':'value','label':'기록'}],[{'item':'연구 상태','value':'추가 학습·후보 선택 종료, 결과 동결'},{'item':'기준일','value':'2026년 10월 27일'},{'item':'비용·장비 가동','value':'사용자 보고 총40달러 / PC4일·96시간'},{'item':'검수와 협업','value':'연구자 본인 1명 검수 / AI 판정·웹 GPT 자문'}]),Spacer(1,17),para('연구 총괄·기획·조건 선택·진행 관리·결과 해석·최종 검수: 연구자 본인\n구현·실행·분석 지원: Codex / 설계·해석 자문: 웹 GPT','small'),para('공개 연구 저장소','coverlink'),para('[github.com/chabin0000/CoMPaSS-AI](https://github.com/chabin0000/CoMPaSS-AI)','coverurl'),para('ARGOS 연구 웹사이트','coverlink'),para('[compass-argos-research-2026.web.app](https://compass-argos-research-2026.web.app/)','coverurl'),PageBreak(),para('읽는 순서와 목차','chapter')]
flow.append(para('핵심 결과는1–2절, 제작 과정은6–22절, 보완 실험은23–39절, 주장·선호·통계는40–44절, 비용·시간·결론은45–50절에 정리했다. 그림과 표는 해당 절의 근거 파일·표본 범위에 연결된다.','small'))
toc=TableOfContents();toc.levelStyles=[ST['toc0'],ST['toc1']];toc.dotsMinLevel=1;flow += [toc,PageBreak()]
for i,s in enumerate(d['report']['sections'],1):
 if s.get('chapter'):
  if i!=1:flow.append(CondPageBreak(160))
  flow.append(para(s['chapter'],'chapter'))
 flow.append(para(f"{i:02d}. {s['title']}",'heading'))
 for p in s['body'].split('\n\n'):
  if p.strip():flow.append(para(p))
 if s.get('figure'):
  path=args.figures/Path(s['figure']).name
  with PILImage.open(path)as im:ratio=im.height/im.width
  w=WIDTH;h=w*ratio
  if h>625:w*=625/h;h=625
  flow +=[Spacer(1,6),KeepTogether([Image(str(path),width=w,height=h,hAlign='CENTER'),Spacer(1,6),para(s.get('figureCaption',''),'small')]),Spacer(1,7)]
 query=d['queries'].get(s.get('queryId'),{});rs=query.get('rows',[])
 if s.get('columns')and rs:flow.extend([table(s['columns'],rs),Spacer(1,10)])
 source=query.get('source',{})
 # A compact source line keeps the narrative readable; full lineage remains in canonical JSON.
 if source.get('label'):flow.append(para('근거: '+source['label']+' · 공개 데이터 키: '+s.get('queryId',''),'small'))
 flow.append(Spacer(1,6))
args.output.parent.mkdir(parents=True,exist_ok=True)
doc=ReportDoc(str(args.output),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=54,bottomMargin=48,title=d['title'],author='CoMPaSS ARGOS researcher; AI-assisted by Codex and ChatGPT',subject='Verified history, observed model comparisons, cost/time scope and research closure',pageCompression=1)
doc.multiBuild(flow,onFirstPage=decorate,onLaterPages=decorate)
reader=PdfReader(args.output);text='\n'.join(p.extract_text()or''for p in reader.pages)
receipt=dict(pdf=args.output.name,sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),pages=len(reader.pages),narrative_sections=len(d['report']['sections']),figures=sum('figure'in s for s in d['report']['sections']),canonical_data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),extracted_characters=len(text),section_pages=doc.section_pages,font_sha256=hashlib.sha256(font.read_bytes()).hexdigest(),visual_verification='pending rendered-page inspection')
args.output.with_suffix('.build.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');args.output.with_suffix('.extracted.txt').write_text(text,encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items()if k!='section_pages'},ensure_ascii=False))
