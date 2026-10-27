"""Publication figures from the reviewed canonical report. No model execution."""
from pathlib import Path
import argparse,json,hashlib,datetime,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
d=json.loads(args.data.read_text(encoding='utf-8'));Q=d['queries'];args.out.mkdir(parents=True,exist_ok=True)
font=Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams.update({'font.size':12,'axes.titlesize':15,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A6B4BE','axes.labelcolor':'#243B4A','text.color':'#193447','xtick.color':'#445B68','ytick.color':'#445B68','axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
INK='#193447';BLUE='#225F91';TEAL='#157F80';ORANGE='#A54B1E';PURPLE='#68558D';RED='#B63F48';GRAY='#899BA6';LIGHT='#EAF0F3'
manifest=[]
def rows(k):return Q[k]['rows']
def finish(fig,name,queries):
 fig.savefig(args.out/(name+'.png'),dpi=230,bbox_inches='tight',pad_inches=.18)
 fig.savefig(args.out/(name+'.svg'),bbox_inches='tight',pad_inches=.18)
 plt.close(fig)
 manifest.append(dict(figure=name,queries=queries,files={suffix:hashlib.sha256((args.out/(name+suffix)).read_bytes()).hexdigest()for suffix in ['.png','.svg']}))
def rate_axis(ax):ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('관측 비율 (%)');ax.grid(axis='x',color=LIGHT);ax.set_axisbelow(True)
def color(model):
 if model=='S0':return BLUE
 if model in ['ARGOS_old','기존 ARGOS','ARGOS S5 general']:return RED
 if model.startswith('R6'):return ORANGE
 if model in ['gpt_oss','glm','GPT-OSS 20B','GLM 4.7 Flash']:return PURPLE
 return TEAL
def box(ax,x,y,w,h,text,edge=BLUE,fill='#F2F6F8',size=12):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.012,rounding_size=.015',fc=fill,ec=edge,lw=1.15))
 ax.text(x,y,text,ha='center',va='center',fontsize=size,linespacing=1.55)
def arrow(ax,a,b,style='solid',color=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.3,linestyle=style,color=color))

# 1. A continuous date axis preserves the long gap between the early and late work.
t=rows('synthesis-timeline');fig,ax=plt.subplots(figsize=(10.8,6.5));dates=[datetime.date.fromisoformat(r['date'])for r in t];y=np.arange(len(t))[::-1]
ax.hlines(y,mdates.date2num(datetime.date(2026,8,9)),[mdates.date2num(x)for x in dates],color=LIGHT,lw=2)
for yy,dt,r in zip(y,dates,t):
 ax.scatter(dt,yy,s=62,color=ORANGE if dt.month==10 else BLUE,zorder=3)
 right=dt<datetime.date(2026,10,10);ax.annotate(r['stage'],(mdates.date2num(dt),yy),xytext=(9 if right else-9,0),textcoords='offset points',ha='left'if right else'right',va='center',fontsize=12)
ax.set_yticks(y,[r['date'][5:]for r in t]);ax.set_xlim(datetime.date(2026,8,6),datetime.date(2026,10,30));ax.set_ylim(-.7,len(t)-.3)
ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'));ax.set_xlabel('2026년 실제 날짜 간격');ax.set_title('초기 설계부터 동결까지: 문서상79일에 걸친 기록',loc='left',pad=18);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);fig.tight_layout();finish(fig,'01_timeline',['synthesis-timeline'])

# 2. Process diagram: boxes are stages, never scaled quantities.
fig,ax=plt.subplots(figsize=(10.8,5.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('두 교사의 답을 검증하고, 같은 기반 모델에서 따로 학습했다',loc='left',pad=14)
nodes=[(.16,.80,'학습·평가 질문 분리\nID·정답·해시 기록'),(.50,.80,'교사 A / 교사 B\nGPT-OSS / GLM'),(.84,.80,'원응답 보존\n성공·실패·회수 구분'),(.84,.49,'정답·자료·계산 검증\nPASS / FAIL / 보류'),(.50,.49,'조건별 데이터 구성\nmatched / full / QA'),(.16,.49,'EXAONE + LoRA\n조건마다 독립 학습'),(.16,.18,'보정·개발 평가\nS0·교사와 비교'),(.50,.18,'전체 승격 요건\n정답·설명·유지'),(.84,.18,'조건부 최종 평가\nR6는 추론 미실행')]
for x,yv,label in nodes:box(ax,x,yv,.27,.19,label,ORANGE if '미실행'in label else BLUE)
for a,b in [((.30,.80),(.36,.80)),((.64,.80),(.70,.80)),((.84,.70),(.84,.59)),((.70,.49),(.64,.49)),((.36,.49),(.30,.49)),((.16,.39),(.16,.28)),((.30,.18),(.36,.18)),((.64,.18),(.70,.18))]:arrow(ax,a,b)
finish(fig,'02_workflow',['synthesis-conditions','synthesis-evaluation-generations'])

# 3. Pilot accuracy with paired outcomes.
fig,axes=plt.subplots(1,2,figsize=(10.8,4.6),gridspec_kw={'width_ratios':[1,1.05]});r=rows('synthesis-pilot');ax=axes[0]
bars=ax.bar([x['model']for x in r],[100*x['correctRate']for x in r],color=[BLUE,TEAL],width=.55)
for bar,x in zip(bars,r):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f"{x['correct']}/{x['n']}",ha='center',fontsize=14)
ax.set_ylim(0,100);ax.set_ylabel('통과율 (%)');ax.set_title('새20문항: 40% → 60%');ax.grid(axis='y',color=LIGHT);ax.set_axisbelow(True)
matrix=np.array([[8,4],[0,8]]);ax=axes[1];im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=8);ax.set_xticks([0,1],['기본 정답','기본 오답']);ax.set_yticks([0,1],['Pilot 정답','Pilot 오답']);ax.set_title('같은 문항의 대응 결과')
for ij in np.ndindex(matrix.shape):ax.text(ij[1],ij[0],str(matrix[ij]),ha='center',va='center',fontsize=24,color='white'if matrix[ij]>=6 else INK)
fig.suptitle('ARGOS-Pilot: 관측 증가, 정확 McNemar p=.125',x=.01,ha='left',fontsize=17);fig.tight_layout(rect=[0,0,1,.93]);finish(fig,'03_pilot',['synthesis-pilot'])

# 4. Reconciliation flow uses arithmetic boxes, not an incorrectly scaled funnel.
fig,ax=plt.subplots(figsize=(10.8,6.5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('회수한 응답만 학습 입력에 포함한 데이터 장부',loc='left',pad=15)
box(ax,.2,.86,.33,.16,'원격 본수집 성공 19,923\n생성 실패77과 별도')
box(ax,.75,.86,.36,.16,'로컬 본수집 회수 19,839\n성공84개 미회수',ORANGE);arrow(ax,(.37,.86),(.56,.86))
box(ax,.75,.61,.36,.16,'보충287 추가 → 20,126\n288목표 중1개 생성 실패');arrow(ax,(.75,.77),(.75,.70))
box(ax,.2,.61,.33,.16,'동결 전22응답 제외\n원본 전체 20,104');arrow(ax,(.56,.61),(.37,.61))
box(ax,.2,.34,.33,.17,'검증 통과 14,903\n고유질문 7,974',TEAL);arrow(ax,(.2,.52),(.2,.43))
box(ax,.75,.34,.36,.17,'검증 실패 1,228\n추가 검토 보류 3,973',ORANGE);arrow(ax,(.37,.53),(.57,.42))
ax.text(.5,.12,'14,903 + 1,228 + 3,973 = 20,104',ha='center',fontsize=18,fontweight='bold');ax.text(.5,.035,'상자 크기는 응답 수에 비례하지 않음 · 원격 미회수와 생성 실패를 합치지 않음',ha='center',fontsize=11)
finish(fig,'04_collection',['historical-collection'])

# 5. Every original condition, with common denominators.
rs=rows('synthesis-original-models');fig,axes=plt.subplots(1,2,figsize=(10.8,7.8),sharey=True)
for ax,field,title in zip(axes,['semantic','joint'],['의미 정답','내용·지시 동시 통과']):
 vals=[100*r[field]/240 for r in rs];ax.barh(np.arange(len(rs)),vals,color=[color(r['model'])for r in rs],height=.66)
 for i,(v,r)in enumerate(zip(vals,rs)):ax.text(v-1.4 if v>87 else v+1,i,f"{r[field]}/240",ha='right'if v>87 else'left',va='center',fontsize=10.5,color='white'if v>87 else INK)
 rate_axis(ax);ax.set_title(title,pad=13)
axes[0].set_yticks(np.arange(len(rs)),[r['model']for r in rs]);axes[0].invert_yaxis();fig.suptitle('원래 본실험13조건: 형식 통과 증가와 의미 정답 감소를 구분',x=.01,ha='left',fontsize=16);fig.tight_layout(rect=[0,0,1,.95]);finish(fig,'05_original_models',['synthesis-original-models'])

# 6. Local teacher advantages and their limits.
rs=rows('synthesis-specialized');cohorts=list(dict.fromkeys(r['cohort']for r in rs));models=list(dict.fromkeys(r['model']for r in rs));fig,ax=plt.subplots(figsize=(10.8,5.3));x=np.arange(len(cohorts));w=.18
for j,m in enumerate(models):
 sub=[next(r for r in rs if r['cohort']==c and r['model']==m)for c in cohorts];vals=[100*r['correctRate']for r in sub];xx=x+(j-1.5)*w
 ax.bar(xx,vals,w,color=color(m),label=m)
 for a,v,r in zip(xx,vals,sub):ax.text(a,v+1.4,f"{r['correct']}/{r['n']}",ha='center',fontsize=10.5)
ax.set_xticks(x,cohorts);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('의미 정답률 (%)');ax.grid(axis='y',color=LIGHT);ax.set_axisbelow(True);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.24),frameon=False);fig.subplots_adjust(top=.75,bottom=.12,left=.08,right=.99);finish(fig,'06_specialized',['synthesis-specialized'])

# 7. Original frozen S5 confirmation (not R6).
rs=rows('synthesis-confirmation');fig,ax=plt.subplots(figsize=(10.8,5.6));yy=np.arange(len(rs))
for i,r in enumerate(rs):
 a=r['baseRate']*100;b=r['argosRate']*100;ax.plot([a,b],[i,i],color='#B9C7CF',lw=2,zorder=1)
 ax.scatter(a,i,color=BLUE,s=70,zorder=3,label='기본 S0'if i==0 else None);ax.scatter(b,i,color=ORANGE,s=70,zorder=3,label='원래 ARGOS'if i==0 else None)
 ax.annotate(str(r['S0']),(a,i),xytext=(0,8),textcoords='offset points',ha='center',color=BLUE,fontsize=11)
 ax.annotate(str(r['ARGOS']),(b,i),xytext=(0,-16),textcoords='offset points',ha='center',color=ORANGE,fontsize=11)
ax.set_yticks(yy,[f"{r['cohort']} /{r['n']}"for r in rs]);ax.invert_yaxis();ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('과제별 관측 비율 (%) · 점 옆 숫자는 정답 수');ax.set_ylim(len(rs)-.4,-.6);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,frameon=False);ax.grid(axis='x',color=LIGHT);fig.tight_layout();finish(fig,'07_confirmation',['synthesis-confirmation'])

# 8. All thirteen current development comparisons; fixed percentage domains.
rs=rows('synthesis-all-development');fig,axes=plt.subplots(2,2,figsize=(10.8,12.0));fields=[('document',96,'문서 정답 /96'),('gsm8k',40,'GSM8K 최종답 /40'),('valid',96,'유효 설명 /96'),('kobest',40,'KoBEST 공식 /40')]
for ax,(field,n,title)in zip(axes.flat,fields):
 vals=[100*r[field]/n for r in rs];ax.barh(np.arange(len(rs)),vals,color=[color(r['model'])for r in rs],height=.68)
 for i,(v,r)in enumerate(zip(vals,rs)):ax.text(v-1.5 if v>84 else v+1,i,f"{r[field]}/{n}",ha='right'if v>84 else'left',va='center',fontsize=10.5,color='white'if v>84 else INK)
 ax.set_yticks(np.arange(len(rs)),[r['label']for r in rs]);ax.invert_yaxis();rate_axis(ax);ax.set_title(title,pad=11)
fig.suptitle('모든 기준·후보의 개발 관측:13행을 빠짐없이 비교',x=.01,ha='left',fontsize=17);fig.tight_layout(rect=[0,0,1,.96],h_pad=2.8,w_pad=2.0);finish(fig,'08_all_development',['synthesis-all-development'])

# 9. Adapter dose on fixed exposed subsets.
rs=rows('recovery-adapter-dose');fig,axes=plt.subplots(1,3,figsize=(10.8,4.2));x=[r['scale']for r in rs]
for ax,field,n,title in zip(axes,['document','gsm8k','mean_document_tokens'],[24,16,None],['문서 정답 /24','수학 정답 /16','문서 평균 출력 토큰']):
 vals=[r[field]/n*100 if n else r[field]for r in rs];ax.plot(x,vals,'o-',color=BLUE,lw=2,ms=7)
 for xx,v,r in zip(x,vals,rs):ax.annotate(f"{r[field]}/{n}"if n else f"{r[field]:.1f}",(xx,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10.5)
 ax.set_xticks([0,.25,.5,1]);ax.set_xlim(-.09,1.09);ax.set_ylim(0,108 if n else 210);ax.set_ylabel('정답률 (%)'if n else'토큰');ax.set_xlabel('어댑터 적용 강도');ax.set_title(title,fontsize=13);ax.grid(color=LIGHT)
fig.tight_layout(w_pad=2);finish(fig,'09_adapter',['recovery-adapter-dose'])

# 10. Same-run checkpoint trajectories, not a serial old->new training claim.
rs=[r for r in rows('synthesis-all-development')if r['model'].startswith('R6')];base=next(r for r in rows('synthesis-all-development')if r['model']=='S0');x=[40,80,120];fig,axes=plt.subplots(1,3,figsize=(10.8,4.6))
for ax,field,n,title in zip(axes,['document','gsm8k','valid'],[96,40,96],['문서 정답 /96','수학 최종답 /40','유효 설명 /96']):
 vals=[100*r[field]/n for r in rs];ax.plot(x,vals,'o-',lw=2,color=ORANGE,label='R6 저장본');ax.axhline(100*base[field]/n,color=BLUE,ls='--',label='기본 S0')
 if field=='document':ax.axhline(100*73/96,color=GRAY,ls=':',label='문서 선정 하한')
 for a,v,r in zip(x,vals,rs):ax.annotate(str(r[field]),(a,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=13)
 ax.set_ylim(0,100);ax.set_xlim(30,130);ax.set_xticks(x);ax.set_yticks([0,25,50,75,100]);ax.set_title(title,fontsize=13);ax.set_xlabel('한 학습의 업데이트 단계');ax.grid(axis='y',color=LIGHT)
axes[0].set_ylabel('관측 비율 (%)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=[0,0,1,.88],w_pad=2);finish(fig,'10_r6_checkpoints',['synthesis-all-development','recovery-selection-design'])

# 11. Equal-sized family heatmap.
rs=rows('recovery-families');models=['S0','ARGOS_old','B4-075','R6-040','R6-080','R6-120','gpt_oss','glm'];labels=['S0','기존 ARGOS','B4-075','R6-040','R6-080','R6-120','GPT-OSS','GLM'];a=np.array([[r[m]for m in models]for r in rs]);fig,ax=plt.subplots(figsize=(10.8,7.5));im=ax.imshow(a,cmap='Blues',vmin=0,vmax=8,aspect='auto');ax.set_xticks(range(8),labels,rotation=25,ha='right');ax.set_yticks(range(12),[r['family_label']for r in rs]);ax.set_title('문서 유형별 정답: 모든 칸의 분모는8',loc='left',pad=16)
for ij in np.ndindex(a.shape):ax.text(ij[1],ij[0],str(a[ij]),ha='center',va='center',fontsize=13,color='white'if a[ij]>=5 else INK)
fig.colorbar(im,ax=ax,ticks=[0,2,4,6,8],label='정답 수 /8',fraction=.025,pad=.03);fig.tight_layout();finish(fig,'11_families',['recovery-families'])

# 12. Paired gains and losses, same count scale in both panels.
rs=rows('recovery-paired-results');fig,axes=plt.subplots(1,2,figsize=(10.8,4.1));mods=['R6-040','R6-080','R6-120']
for ax,cohort,title in zip(axes,['document','gsm8k'],['문서96문항','수학40문항']):
 rr=[next(r for r in rs if r['cohort']==cohort and r['model']==m)for m in mods];yy=np.arange(3)
 ax.barh(yy,[-r['base_only']for r in rr],color=BLUE,label='S0만 정답');ax.barh(yy,[r['model_only']for r in rr],color=ORANGE,label='후보만 정답')
 for i,r in enumerate(rr):ax.text(-r['base_only']-.8,i,str(-r['base_only']),ha='right',va='center');ax.text(r['model_only']+.8,i,'+'+str(r['model_only']),va='center')
 ax.axvline(0,color=INK,lw=.8);ax.set_xlim(-32,32);ax.set_xticks([-30,-15,0,15,30]);ax.set_yticks(yy,mods);ax.invert_yaxis();ax.set_xlabel('같은 문항에서의 정답 수 차이');ax.set_title(title);ax.grid(axis='x',color=LIGHT);ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False);fig.tight_layout(rect=[0,0,1,.86]);finish(fig,'12_paired',['recovery-paired-results'])

# 13. Atomic claims: show denominators and verbosity separately.
rs=rows('baseline-claims');fig,axes=plt.subplots(1,2,figsize=(10.8,5),gridspec_kw={'width_ratios':[1.6,1]});left=np.zeros(len(rs));yy=np.arange(len(rs))
for field,c,label in [('supported',TEAL,'자료 지지'),('contradicted',RED,'자료 모순'),('unverifiable',GRAY,'판정불가')]:
 vals=np.array([r[field]/r['claims']*100 for r in rs]);axes[0].barh(yy,vals,left=left,color=c,label=label,height=.65)
 if field=='contradicted':
  for i,(v,l,r)in enumerate(zip(vals,left,rs)):axes[0].text(101,i,f"{r['contradicted']}/{r['claims']}",va='center',fontsize=11)
 left+=vals
axes[0].set_yticks(yy,[r['model']for r in rs]);axes[0].invert_yaxis();axes[0].set_xlim(0,122);axes[0].set_xticks([0,25,50,75,100]);axes[0].set_xlabel('분해된 전체 주장 중 비율 (%)');axes[0].set_title('주장 지지 상태 · 오른쪽은 모순/전체');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False,fontsize=10)
vals=[r['claims']/24 for r in rs];axes[1].barh(yy,vals,color=GRAY,height=.65);axes[1].set_yticks(yy,[]);axes[1].invert_yaxis();axes[1].set_xlim(0,12);axes[1].set_title('계획 응답당 주장 수');axes[1].set_xlabel('주장 / 응답 (각24응답)')
for i,v in enumerate(vals):axes[1].text(v+.15,i,f'{v:.2f}',va='center')
fig.tight_layout(w_pad=3);finish(fig,'13_claims',['baseline-claims'])

# 14. Recompute the cross-tab from first judgments, not from marginals.
rs=rows('synthesis-pair-matrix');fig,axes=plt.subplots(1,2,figsize=(10.8,4.9));cats=['A','B','tie'];labels=['A 선호','B 선호','동률']
for ax,criterion,title in zip(axes,['overall','naturalness'],['종합 선호: 57/72 일치 · κ=.636','자연스러움: 19/72 일치 · κ=.136']):
 a=np.array([[next(r['count']for r in rs if r['criterion']==criterion and r['codex']==c and r['project_pro']==p)for p in cats]for c in cats]);im=ax.imshow(a,cmap='Blues',vmin=0,vmax=40);ax.set_xticks(range(3),labels);ax.set_yticks(range(3),labels);ax.set_xlabel('프로젝트 ChatGPT Pro');ax.set_ylabel('Codex');ax.set_title(title,fontsize=13,pad=13)
 for ij in np.ndindex(a.shape):ax.text(ij[1],ij[0],str(a[ij]),ha='center',va='center',fontsize=23,color='white'if a[ij]>=23 else INK)
fig.tight_layout(w_pad=3);finish(fig,'14_agreement',['synthesis-pair-matrix','pair-agreement'])

# 15. Evaluated vs conditional unexecuted stages.
fig,ax=plt.subplots(figsize=(10.8,5.8));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('실행된 새 확인과 미실행 조건부 최종을 구분',loc='left',pad=16)
for y,label,nodes in [(.81,'원래 S5',['대표 S5 동결','새 합성240\n공개120','S0·S5 추론\n720응답','새 확인 결과\n실측']),(.49,'A4/B4/C5',['3회 학습\n6개 저장본','보정·개발\n선정용 실행','승격 후보\n0개','최종160+80\n추론0건']),(.17,'마지막 R6',['1회120단계\n3개 저장본','S0 재측정\n개발704응답','승격 후보\n0개','최종200+80\n추론0건'])]:
 for i,text in enumerate(nodes):
  x=.13+i*.245;box(ax,x,y,.205,.19,text,TEAL if y>.7 else ORANGE,size=12)
  if i<3:arrow(ax,(x+.108,y),(x+.136,y),style='dashed'if i==2 and y<.7 else'solid')
 ax.text(.01,y+.126,label,fontsize=12,fontweight='bold')
finish(fig,'15_evaluation_flow',['synthesis-evaluation-generations'])

# 16. Original confirmation intervals with item/family sensitivity clearly separate.
rs=rows('synthesis-confirmation-statistics');order=['math','source_claims','premise','instruction','gsm8k','klue_nli','kobest_boolq'];rs.sort(key=lambda r:order.index(r['cohort_id']));fig,ax=plt.subplots(figsize=(10.8,5.7));yy=np.arange(len(rs))
for i,r in enumerate(rs):
 ax.plot([r['low'],r['high']],[i-.08,i-.08],color=BLUE,lw=3,label='문항 재표집95%'if i==0 else None);ax.scatter(r['delta'],i-.08,color=BLUE,s=45,zorder=3)
 if r['family_low']is not None:ax.plot([r['family_low'],r['family_high']],[i+.14,i+.14],color=ORANGE,lw=2,ls='--',label='유형 재표집95%'if i==0 else None);ax.scatter(r['delta'],i+.14,color=ORANGE,s=28,zorder=3)
 ax.text(100,i,f"{r['delta']:+.2f}",va='center',ha='right',fontsize=11)
ax.axvline(0,color=GRAY,ls=':',lw=1.2);ax.set_yticks(yy,[f"{r['cohort']} (n={r['n']})"for r in rs]);ax.invert_yaxis();ax.set_xlim(-100,105);ax.set_xticks([-100,-50,0,50,100]);ax.set_xlabel('원래 ARGOS - S0 (퍼센트포인트)');ax.set_title('새 확인평가의 대응 차이: 탐색적·체크포인트 조건부 구간',loc='left',pad=18);ax.grid(axis='x',color=LIGHT);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.25),ncol=2,frameon=False);fig.tight_layout();finish(fig,'16_intervals',['synthesis-confirmation-statistics'])

# 17. Incommensurable evidence is shown as separate cards, never an additive stack.
fig,ax=plt.subplots(figsize=(10.8,4.7));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
cards=[(.25,.74,'프로젝트 전체 지출','$40','사용자 보고 · 전체 청구서 대조 없음'),(.75,.74,'PC 연속 가동','96시간','사용자 보고 · 인간/GPU 작업시간 아님'),(.25,.25,'문서로 확인한 연구 경과','79일','8월9일~10월27일 · 매일 작업한 기간 아님'),(.75,.25,'Codex 메시지 연결 구간','13.92시간','30분 규칙 · 실제 노동시간 측정 아님')]
for x,y,title,value,note in cards:
 box(ax,x,y,.45,.40,'',BLUE);ax.text(x,y+.12,title,ha='center',fontsize=13);ax.text(x,y-.005,value,ha='center',fontsize=30,fontweight='bold',color=BLUE);ax.text(x,y-.13,note,ha='center',fontsize=10.5)
finish(fig,'17_cost_time',['synthesis-cost','synthesis-activity'])

# 18. Daily record counts and rule sensitivity, not an estimate confidence interval.
rs=rows('synthesis-activity-daily');ss=rows('synthesis-activity');fig,axes=plt.subplots(1,2,figsize=(10.8,5.2),gridspec_kw={'width_ratios':[1.15,1]})
x=np.arange(len(rs));axes[0].bar(x,[r['messages']for r in rs],color=BLUE,width=.62)
for i,r in enumerate(rs):axes[0].text(i,r['messages']+1,str(r['messages']),ha='center',fontsize=11)
axes[0].set_xticks(x,[r['date'][5:]for r in rs],rotation=45,ha='right');axes[0].set_ylim(0,50);axes[0].set_title('필터 후 사용자 역할 기록177건',fontsize=13);axes[0].set_ylabel('기록 수');axes[0].set_xlabel('기록이 있는 날짜만 표시');axes[0].grid(axis='y',color=LIGHT);axes[0].set_axisbelow(True)
x=np.arange(3);w=.34
for field,offset,c,label in [('all_hours',-.17,BLUE,'전체5개 로그'),('late_hours',.17,ORANGE,'10/23~27')]:
 vals=[r[field]for r in ss];axes[1].bar(x+offset,vals,w,color=c,label=label)
 for xx,v in zip(x+offset,vals):axes[1].text(xx,v+.4,f'{v:.2f}',ha='center',fontsize=10.5)
axes[1].set_xticks(x,[f"≤{r['gap_minutes']}분"for r in ss]);axes[1].set_ylim(0,25);axes[1].set_ylabel('연결 구간 합계 (시간)');axes[1].set_title('연결 간격 규칙에 따른 민감도',fontsize=13);axes[1].legend(loc='upper left',frameon=False,fontsize=10);axes[1].set_xlabel('실제 노동시간의 범위·신뢰구간 아님');axes[1].grid(axis='y',color=LIGHT);axes[1].set_axisbelow(True);fig.tight_layout(w_pad=2.5);finish(fig,'18_activity',['synthesis-activity','synthesis-activity-daily'])

result={'canonical_data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'figures':manifest,'number_of_figures':len(manifest),'software':'Matplotlib scientific figures, PNG and SVG from the same reviewed data','no_new_inference':True}
(args.out/'figure_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Saved',len(manifest),'scientific figures as PNG and SVG')
