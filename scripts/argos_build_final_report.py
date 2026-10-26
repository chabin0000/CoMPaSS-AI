"""Rebuild the public research report from public aggregate files only.

Requires argos_report_content.py alongside this script, reportlab, matplotlib and Pillow.
On Windows use Malgun Gothic; --font and --bold-font override it elsewhere.
"""
from pathlib import Path
import argparse,json,html,re,hashlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether,CondPageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from argos_report_content import INTRO,CHAPTERS,REFERENCES,REVIEW_CONSENSUS

LABEL={'math':'수학 80','source_claims':'자료 사실 60','premise':'전제 60','instruction':'정확 출력 40','klue_nli':'KLUE 40','kobest_boolq':'KoBEST 40','gsm8k':'GSM8K 40'}
FAMILY={**{f'math_{i:02}':s for i,s in enumerate(['조합','비복원 확률','최대공약수','최소공배수','할인과 세금','왕복 평균 속력','직육면체 겉넓이','등차수열 양끝 차','배수 개수','두 점 사이 거리','연립방정식 x','모집단 분산','독립 사건의 합집합','진법 변환','연립방정식 y','수열 인접항 차'])},**{f'premise_{i}':s for i,s in enumerate(['온도 임계값','조건문의 역','정원 경계','일부와 전체','필요 조건의 충족','전원과 작동'])},**{f'instruction_{i}':s for i,s in enumerate(['최댓값 이름','최솟값 이름','두 값의 합','이름 정렬'])}}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def fmt(x,d=1):return '미측정' if x is None else f'{x:.{d}f}'
def pc(k,n):return f'{k}/{n} ({100*k/n:.1f}%)' if n else '미측정'
def compact(name):return name.replace('S5_general_control','S5 general').replace('S4_full_verified','S4 full').replace('S4_teacher_only','S4 matched').replace('S3_matched','S3 matched').replace('-20261022',' seed22').replace('-20261023',' seed23')
def figures(out,d,font):
    f=out/'figures';f.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.dpi':160})
    a=read(d/'training_audit.json')['rows'];lookup={r['condition']:r for r in a}
    fig,axes=plt.subplots(1,2,figsize=(9,3.3));cols=['#627a91','#183d61']
    for ax,values,title in [(axes[0],[4047,27],'설명형 응답 수'),(axes[1],[23.80,3.59],'답변 평균 길이 (문자)')]:
        bars=ax.bar(['원본 전체','검증 통과 전체'],values,color=cols,width=.55);ax.set_title(title);ax.bar_label(bars,padding=4);ax.margins(y=.22)
    fig.tight_layout();fig.savefig(f/'data_filter.png',bbox_inches='tight');plt.close(fig)
    old=read(d/'historical_metrics.json');rs=[r for r in old if r['regime']=='extended_8192' and r['scope']=='private']
    fig,ax=plt.subplots(figsize=(9,5));y=np.arange(len(rs));ax.barh(y-.17,[r['semantic_correct']/240*100 for r in rs],.32,label='의미 정답',color='#183d61');ax.barh(y+.17,[r['passes']/240*100 for r in rs],.32,label='의미 정답 + 지시 준수',color='#91b3cc');ax.set_yticks(y,[compact(r['variant']) for r in rs]);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('기존 비공개 240문항에서의 비율 (%)');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(f/'historical_accuracy.png',bbox_inches='tight');plt.close(fig)
    metrics=read(d/'metrics.json');order=list(LABEL);look={(r['cohort'],r['model']):r for r in metrics};fig,ax=plt.subplots(figsize=(9,4));x=np.arange(7)
    for offset,model,color in [(-.18,'S0','#7b8b99'),(.18,'ARGOS','#183d61')]:
        vals=[look[c,model]['semantic_rate']*100 for c in order];low=[max(0,v-look[c,model]['wilson95'][0]*100) for c,v in zip(order,vals)];high=[max(0,look[c,model]['wilson95'][1]*100-v) for c,v in zip(order,vals)]
        ax.bar(x+offset,vals,.34,color=color,label=model,yerr=[low,high],capsize=2,error_kw={'elinewidth':.8})
    ax.set_xticks(x,['수학','자료 사실','전제 판단','정확 출력','KLUE NLI','KoBEST','GSM8K']);ax.set_ylim(0,110);ax.set_ylabel('과제별 성공률 (%)');ax.legend();fig.tight_layout();fig.savefig(f/'new_evaluation.png',bbox_inches='tight');plt.close(fig)
    claims=read(d/'claim_metrics.json');fig,ax=plt.subplots(figsize=(8,3));bottom=np.zeros(2)
    for kind,color in [('supported','#1c5970'),('contradicted','#ae4f42'),('unsupported','#c99847'),('unresolved','#bec8cf')]:
        vals=[r[kind] for r in claims];ax.bar(['S0','ARGOS'],vals,bottom=bottom,label={'supported':'자료와 일치','contradicted':'자료와 모순','unsupported':'근거 없음','unresolved':'미해결'}[kind],color=color,width=.5);bottom+=vals
    ax.set_ylabel('주석한 원자적 주장 수');ax.legend(loc='upper right');ax.margins(y=.25);fig.tight_layout();fig.savefig(f/'atomic_claims.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5));bars=ax.bar(['기반 EXAONE','초기 ARGOS-Pilot'],[40,60],color=['#7b8b99','#183d61'],width=.5);ax.bar_label(bars,labels=['8/20 (40%)','12/20 (60%)'],padding=6);ax.set_ylim(0,100);ax.set_ylabel('신규 20문항 정답률 (%)');ax.set_title('2026년 8월 초기 Pilot: +20%p, 정확 McNemar p=0.125');fig.tight_layout();fig.savefig(f/'early_pilot.png',bbox_inches='tight');plt.close(fig)
    return f

def build(root,font,bold):
    out=Path(root);d=out/'data';assert (d/'review_closure.json').exists(),'No report before actual reviews close'
    f=figures(out,d,font);metrics=read(d/'metrics.json');compare=read(d/'comparisons.json');claims=read(d/'claim_metrics.json');qual=read(d/'qualitative.json');runtime=read(d/'runtime.json');old=read(d/'historical_metrics.json');follow=read(d/'followup_summary.json');audit=read(d/'training_audit.json')['rows'];bench=read(d/'benchmark_sources.json');manifest=read(out/'evidence_manifest.json')
    lookup={(r['cohort'],r['model']):r for r in metrics};blocks=[]
    def h(t):blocks.append(('h',t))
    def prose(t):
        for p in t.strip().split('\n\n'):blocks.append(('p',p.replace('\n',' ')))
    def table(caption,headers,rs,widths=None):blocks.append(('table',caption,headers,[[str(x) for x in r] for r in rs],widths))
    def img(name,caption):blocks.append(('image',f/name,caption))
    h('연구 요약');prose(INTRO)
    a=lookup['math','ARGOS'];b=lookup['math','S0'];prose(f'신규 수학 80문항의 확인된 정답은 S0 {b["semantic_correct"]}개, ARGOS {a["semantic_correct"]}개였다. 새 문항에서도 결과를 바꾸기 위한 추가 학습이나 후보 교체는 하지 않았다. 아래 본문은 기존 실험과 신규 평가를 구분하고, 부록은 모든 관측 조건을 보존한다.')
    table('최종 연구 상태',['항목','실제 수행 및 판단'],[
        ['본실험','학생 학습 10회와 기존 13조건 평가 완료'],['후속 수학','교사 재수집 및 5개 체크포인트 보존, 새 후보 채택 없음'],['신규 확인','비공개 240문항과 공개 120문항 × 2모델'],['검수','단일 AI, 사람 0명, 독립 평가자 카파 미산출'],['모델 선택','기존 S5_general_control 유지, 범용 우월성 주장 없음'],['공개','보고서와 집계 및 재현 코드, 비공개 문항과 가중치 제외']],[35,135])
    for index,(title,body) in enumerate(CHAPTERS):
        h(f'{index+1} {title}');prose(body)
        if title=='연구의 출발과 ARGOS의 의미':
            table('실제로 진행된 연구의 시간 순서',['시기','단계','장비와 수행 내용'],[['2026.08.09~11','설계 동결','DGX 교사 실행 준비와 데이터 분리·승인 장치'],['2026.08.15','첫 ARGOS-Pilot','DGX 교사 기록 50응답 중 PASS30 → 로컬 15스텝 학습'],['2026.10.22~23','클라우드 전환','A40 48GB 임대, 150문항 프로파일 비교와 학습 시간 측정'],['2026.10.23~24','24GB 수집','A5000으로 이전, 기존 응답 회수 후 수집·별도 재생성'],['2026.10.25~26','본실험','A5000 24GB에서 독립 학생 학습 10회와 13조건 평가'],['2026.10.26','수학 보충','후속 교사 응답과 5개 체크포인트 진단, 새 후보 미채택'],['2026.10.26~27','동결 확인','로컬 RTX 5060 Ti에서 새 비공개240 + 공개120 × 2모델']],[32,30,108])
        if title=='DGX Spark 구상과 초기 자동화 파이프라인':
            table('계획 속 장비·교사와 실제 실행의 구분',['단계','교사 / 학생','증거와 해석'],[['초기 구상','gpt-oss-120B + GLM-4.5-Air','학교 DGX Spark 활용 계획. 최종 본실험 사양이 아님'],['첫 Pilot','gpt-oss-20B / EXAONE-3.5-7.8B','DGX 교사 수집 기록. 학생 학습은 로컬 GPU'],['10월 본실험','gpt-oss-20B + GLM-4.7-Flash / 같은 EXAONE','임대 GPU에서 두 교사의 답변을 사용한 조건별 학습']],[29,64,77])
        if title=='첫 ARGOS-Pilot의 제작과 신규 20문항 평가':
            img('early_pilot.png','그림 H1 초기 Pilot의 소규모 개선. 새로 읽은 원본 파일이 아니라 당시 원본을 읽었던 보존 도구 출력과 검증 보고서로 복원한 값이다. 본실험 및 신규 240문항과 직접 합산하지 않는다. 자료 H01.')
            table('초기 Pilot의 실제 학습과 검증',['항목','기록'],[['교사 데이터','50응답 중 PASS30, FAIL20. PASS30만 학습'],['학생 설정','4bit, batch1 × accumulation4, length768, lr0.0002, LoRA rank16/alpha32'],['실제 학습','15 optimizer steps, epoch1.9333, 428.9092초, loss1.08326'],['런타임 확인','4bit 실학습, 저장 어댑터 재로딩 성공, 비양자화 fallback 없음'],['짝지은 결과','둘 다 정답8 / ARGOS만 정답4 / 기본만 정답0 / 둘 다 오답8'],['길이 미완료','기본3개 → Pilot0개'],['형식 위반','기본4개 → Pilot3개'],['일반화 한계','20문항·단일 시드·교사 동시평가 없음. 통계적 확증 아님']],[40,130])
        if title=='클라우드 GPU 임대로 옮긴 과정':
            table('장비별 실제 역할과 비용 기록',['장비','역할','비용 또는 측정 범위'],[['DGX Spark','초기 구상과 초기 교사 수집 기록','10월 본실험 장비로 표기하지 않음'],['A6000 Community','임대 후보 견적','당시 $0.33/h 견적, 재고 없음. 실제 사용 주장 없음'],['A40 48GB Secure','실제 임대 파일럿','GPU0.59 + root0.007 + volume0.001 = $0.598/h'],['A5000 24GB','대량 수집·본학습·기존 평가','$0.27/h 기록, 24,564MiB 물리 메모리'],['로컬 RTX 5060 Ti','8월 학생 Pilot·10월 신규 확인','새 클라우드 비용 없음. 전기요금 미계측']],[43,54,73])
            table('A40 교사 프로파일 비교: 동일 150문항의 재실행',['프로파일','완료 응답','PASS / REVIEW / FAIL','응답 생성 구간'],[['GPT 초기','150','89 / 54 / 7','87.8초'],['GLM 초기','9','길이 문제로 중단','비교 불완전'],['GLM thinking budget512','149 (+1 실패)','17 / 130 / 2','872.1초'],['GPT strict, 4slots','150','108 / 31 / 11','108.5초'],['GLM thinking off','150','45 / 32 / 73','29.7초'],['GLM thinking on, cap3072','150','114 / 31 / 5','1,024초']],[62,28,46,34]);prose('표의 시간은 모델 다운로드·로딩·대기·임대 전체 시간이 아니라 해당 응답 생성 구간이다. 758개 완료 응답은 여러 프로파일에서 같은 150문항을 반복한 수이며 758개의 새 문항이 아니다. 빠른 thinking off보다 정답 검증을 통과하는 최종 프로파일을 채택했다. 자료 H03 및 당시 연구 기록 H05.')
        if title=='48GB에서 24GB로의 이전과 응답 회수':
            table('수집 수량이 달라진 이유',['범위','응답 수','관계'],[['원격 주 수집 성공','19,923','생성 실패77개와 별도'],['로컬 주 수집 회수','19,839','원격 성공 중84개 미회수'],['별도 cloud24-r2 회수','287','144문항 × 2교사 중1개 생성 실패'],['원격 성공 합산','20,210','19,923 + 287. 과거 대화의20,212와2개 차이, 미해명'],['로컬 통합 입력','20,126','19,839 + 287'],['동결 전 제외','22','로컬 통합 입력에서 제외'],['본실험 원본 전체','20,104','20,126 − 22, 고유 질문10,133'],['검증 통과 전체','14,903','고유 질문7,974'],['학습 미포함','5,201','FAIL1,228 + REVIEW3,973']],[48,25,97]);prose('수량을 비교할 때 같은 시점·상태·저장 위치의 분모를 맞춰야 한다. 과거 대화의 20,212라는 수치는 현재 회수 원장으로 그대로 재현되지 않는다. 현재 확인되는 원격 성공 합은20,210, 실제 로컬 통합 입력은20,126이다. 둘의 차이84개는 미회수 성공이며, 로컬 입력과 동결본 사이22개는 제외 항목이다. 20,212에서20,104를 단순히108개 중복 제거로 설명하지 않는다.')
        if title=='교사 답변에서 학생 학습본까지':
            runs=read(d/'history_training_runs.json');table('실제 완료된 독립 학생 학습 10회',['학습본','응답 수','시드','스텝','시간 분','loss'],[[compact(r['variant']),f"{r['samples']:,}",str(r['seed'])[-2:],r['steps'],fmt(r['train_runtime_seconds']/60,1),fmt(r['train_loss'],4)] for r in runs],[59,26,17,22,23,23]);prose('시드 열의22와23은 각각20261022와20261023의 끝 두 자리다. 모든 학습은 같은 기반 모델에 새 어댑터를 붙여 시작했고 실제4bit 학습과 저장 후 어댑터 재로딩 완료가 기록되었다. 시간은 훈련 함수의 train_runtime이며 전체 GPU 임대 시간과 같지 않다. 자료 H06.')
        if title=='실행 중 수정한 연구 설계':
            table('계획과 실행 중 확정한 내용',['항목','계획의 변화','최종 기록'],[['장비','DGX Spark 이후 48GB 임대 계획','A40 파일럿 → A5000 24GB 본실험 → 8GB 로컬 확인'],['검수','사람 2명 이상','사용자 요청에 따른 단일 AI 응답 판정'],['평가','선봉인 최종 평가','기존 평가의 선정 사용 공개 후 새 평가'],['학습','사전 조건 비교','본실험 이후 별도 진단 및 재학습'],['벤치마크','여러 공개 평가 후보','KMMLU100 보존 및 3종 각40 보완'],['완결','가설 검증과 결과 정리','성능 향상 없는 결과도 보고']],[28,60,82])
        if title=='응답 수집과 데이터 계보':img('data_filter.png','그림 1 검증 후 응답 유형과 길이가 함께 달라졌다. 표본 선택은 정확도만의 개입이 아니다. 자료 E05.')
        if title=='학습 조건과 통제 수준':
            table('본실험 학습 데이터와 감독량',['조건','응답 수','설명형','평균 답변 문자','평균 감독 토큰'],[[r['condition'],f"{r['n']:,}",r['response_format_counts'].get('explanation',0),fmt(r['mean_answer_characters'],2),fmt(r['mean_supervised_tokens'],2)] for r in audit],[50,25,23,36,36])
        if title=='기존 평가 결과와 모델 선택':
            contrast=read(d/'historical_comparisons.json');primary=[r for r in old if r['scope']=='private' and (r['variant'].startswith('S3_matched') or r['variant'].startswith('S4_teacher_only'))];table('핵심 통제 비교: 같은 질문·감독 토큰 예산의 두 시드',['출력 조건','학습본','의미 정답 /240','동시 통과 /240'],[[r['regime'],compact(r['variant']),r['semantic_correct'],r['passes']] for r in primary],[33,69,34,34]);table('S4_teacher_only − S3_matched: 두 시드 평균의 동시 통과율 차이',['출력 조건','차이 %p','문항 재표집95%','계열 재표집95%'],[[r['regime'],fmt(r['difference_pp'],3),f"{r['lower_pp']:.3f} ~ {r['upper_pp']:.3f}",f"{r['family_lower_pp']:.3f} ~ {r['family_upper_pp']:.3f}"] for r in contrast if r['scope']=='private' and r['comparison']=='S4_teacher_only - S3_matched'],[36,26,54,54]);prose('위 구간은 관측된 두 시드 결과의 문항·계열 재표집에 따른 변동이며 학습 시드 전체의 변동을 충분히 추정하지 못한다. 양쪽 구간 모두0을 포함하므로 검증본의 확정적 우월성을 주장하지 않는다. 자료 E01·E02.')
            img('historical_accuracy.png','그림 2 기존 확장 비공개 240문항에서 의미 정확도와 지시를 포함한 통과율은 다르다. 기존 최종 평가를 모델 선택에도 사용했으므로 독립 확인 점수가 아니다. 자료 E01.')
            rs=[r for r in old if r['regime']=='extended_8192' and r['scope']=='private'];table('기존 확장 비공개 평가 전 조건',['조건','의미 정답','지시 준수','동시 통과','미완료'],[[compact(r['variant']),pc(r['semantic_correct'],240),pc(r['instruction_passes'],240),pc(r['passes'],240),r['generation_failures']] for r in rs],[48,34,34,34,20])
        if title=='후속 수학 증류와 실패 결과':
            names=list(dict.fromkeys(r['generation'] for r in follow['development']));look={(r['generation'],r['cohort']):r for r in follow['development']};table('후속 개발 평가 AI 내용 판정',['후보','수학 답만 32','수학 풀이 32','출처 16','기록 16'],[[n.replace('-development',''),look[n,'math_plain']['semantic'],look[n,'math_worked']['semantic'],look[n,'source_fidelity']['semantic'],look[n,'record_guard']['semantic']] for n in names],[66,28,28,24,24]);prose('이 표는 AI 내용 검수 후의 의미 정답 수다. 본문의 기계 추출 결과와 다를 수 있으며 그 차이를 숨기지 않았다. 수학 답만과 풀이의 32문제는 동일한 원문제를 반복한 것이다. 기본 모델 풀이 1개와 기존 선택 모델 풀이 1개는 길이 미완료다.')
        if title=='신규 평가의 설계와 봉인':
            h('신규 평가 결과');img('new_evaluation.png','그림 3 신규 동결 평가의 과제별 성공률. 정확 출력 영역은 공백 정규화 후 전체 응답 일치, 나머지는 내용 정답이다. 막대의 구간은 Wilson 95% 구간이다. 영역을 합산한 순위는 만들지 않았고 미완료와 미해결은 성공에 넣지 않았다.')
            table('신규 과제별 성공과 출력 준수',['영역','모델','과제별 성공','형식 조건','동시 통과','미확정'],[[LABEL[r['cohort']],r['model'],pc(r['semantic_correct'],r['n']),pc(r['instruction'],r['n']),pc(r['joint'],r['n']),r['unknown']] for r in sorted(metrics,key=lambda r:(list(LABEL).index(r['cohort']),r['model']))],[30,18,34,34,34,20])
            prose('전제 판단은 첫 줄 결론의 정답을 계산했고, 근거 설명의 타당성은 별도 표본 검수에서 확인했다. 정확 출력40문항은 내용과 형식이 결합된 공백 정규화 후 전체 응답 일치다. 예를 들어 틀린 숫자라도 숫자만 쓰라는 형식은 지킬 수 있고, 정확한 이름에 설명을 덧붙이면 정확 출력에는 실패할 수 있다. 수학은 정수 또는 정확한 분수의 값만 비교하며 모든 중간 풀이의 타당성을 보증하지 않는다.')
            h('공개 보조 평가의 범위');table('공개 표본 출처',['자료','사용 split','전체 수','평가 수','해석'],[[x['dataset'],x['split_file'].split('/')[0],x['available'],x['selected'],'표본 및 별도 프롬프트'] for x in bench],[44,28,24,24,50]);prose('KLUE는 NLI validation, KoBEST는 BoolQ test, GSM8K는 main test를 사용했다. 각 자료에서 고정된 난수 표본 40개를 뽑았으며 원문과 기준 정답을 사용했다. 영어 GSM8K는 한국어 문항으로 번역하지 않았다. 자료마다 언어와 과제가 다르므로 120개를 합친 점수를 일반 지능 수치로 해석하지 않는다. 공개 보조 문항은 사전학습 오염 가능성을 배제하지 못한다.')
        if title=='주장 단위의 사실 오류와 검수 원칙':
            img('atomic_claims.png','그림 4 제공 자료 응답의 실제 원자적 주장 분포. 주장하지 않은 내용은 가상 주장으로 채우지 않는다. 주장 수와 응답 범위를 함께 확인한다.')
            table('원자적 주장 집계',['모델','일치','모순','근거 없음','미해결','자료 모순율','주장 응답'],[[r['model'],r['supported'],r['contradicted'],r['unsupported'],r['unresolved'],fmt(100*r['factual_error_rate'])+'%' if r['factual_error_rate'] is not None else '미산출',str(r['responses_with_claims'])+'/60'] for r in claims],[22,22,22,28,24,27,25])
            table('사실 충실성의 분모와 범위',['모델','요청 사실 충족','근거 없음 / 전체 주장','근거 없는 주장이 나온 응답'],[[r['model'],pc(r['required_facts_supported'],180),pc(r['unsupported'],r['emitted_atomic_claims']),pc(r['responses_with_unsupported'],60)] for r in claims],[24,42,56,48])
            prose('두 모델의 기록 모순율은 모두 0%지만 S0의 근거 없는 추가 주장은 49/228개다. 따라서 모순율 0%를 환각이 전혀 없다는 뜻으로 해석하면 안 된다. 새로운 기후, 재료의 내구성, 미래 기술, 색상의 상징 같은 설명은 기록이 뒷받침하지 않았다. ARGOS의 181개 주장은 기록과 일치했지만 한 응답은 요청하지 않은 사실을 덧붙여 출력 지시를 위반했다.')
            table('눈가림 AI 쌍대 평가',['모델','승','패','동률','자연스러움','산문 표본','설명 점수'],[[r['model'],r['wins'],r['losses'],r['ties'],fmt(r['naturalness_mean'],2),r['naturalness_n'],fmt(r['explanation_mean'],2)] for r in qual],[24,18,18,18,34,26,32]);prose('자연스러움은 실제 산문이 있는 응답에 한해 1~5점, 설명은 0~2점으로 평가했다. 0은 근거 없음 또는 잘못된 근거, 1은 부분적 근거, 2는 질문의 결론을 뒷받침하는 충분한 근거다. 선호는 사실과 지시 충족을 우선하며 유창한 오답을 정답보다 우선하지 않는다. 사람의 판정을 대신한 독립 신뢰도 수치는 없다.')
        if title=='통계 처리와 불확실성':
            table('신규 과제별 성공 차이 ARGOS 빼기 S0',['영역','차이 %p','문항 95% 구간','계열 95% 구간','승 패'],[[LABEL[r['cohort']],fmt(r['delta_pp']),f"{r['item_bootstrap95_pp'][0]:.1f} ~ {r['item_bootstrap95_pp'][1]:.1f}",('미산출' if r['family_bootstrap95_pp'] is None else f"{r['family_bootstrap95_pp'][0]:.1f} ~ {r['family_bootstrap95_pp'][1]:.1f}"),f"{r['wins']} / {r['losses']}"] for r in compare],[34,24,40,40,32])
        if title=='계산 자원과 비용의 해석':
            eff=read(d/'local_efficiency.json');table('로컬 신규 평가 생성 지연',['모델','호출','중앙값 초','95백분위 초','생성 토큰','미완료'],[[r['model'],r['requests'],fmt(r['latency_median'],3),fmt(r['latency_p95'],3),f"{r['output_tokens']:,}",r['incomplete']] for r in eff],[25,20,32,36,35,22]);prose(f"이번 실행의 PyTorch는 {runtime['packages']['torch']}, Transformers는 {runtime['packages']['transformers']}, PEFT는 {runtime['packages']['peft']}, bitsandbytes는 {runtime['packages']['bitsandbytes']}이다. 최대 torch 할당 메모리는 {read(d/'inference_completion.json')['peak_vram_bytes']/1024**3:.2f}GiB였다. 이는 드라이버와 화면 표시를 포함한 전체 GPU 사용량과 다르다.")
    h('신규 세부 계열과 쌍대 결과');families=read(d/'family_metrics.json')
    for cohort in ['math','premise','instruction']:
        fs=sorted({r['family'] for r in families if r['cohort']==cohort});fl={(r['family'],r['model']):r for r in families if r['cohort']==cohort}
        table(LABEL[cohort]+' 계열별 성공',['계열','문항 수','S0','ARGOS'],[[fam+' '+FAMILY.get(fam,''),fl[fam,'S0']['n'],fl[fam,'S0']['correct'],fl[fam,'ARGOS']['correct']] for fam in fs],[76,30,32,32])
    table('쌍대 선호의 과제별 분해',['과제','모델','승','패','동률'],[[LABEL[r['cohort']],r['model'],r['wins'],r['losses'],r['ties']] for r in read(d/'pair_domains.json')],[50,36,28,28,28])
    prose('각 수학 계열은 5문항으로 작다. 계열별 수치는 실패 위치를 기술하는 자료이며 계열 순위를 확정하는 근거가 아니다. 쌍대 표본의 자료 사실과 전제는 각각 20쌍이고, 전체 선호는 이 두 과제의 표본 구성에 조건부다.')
    h('부록 A 기존 평가 전체 관측표');prose('기존 원본과 확장 평가를 별도로 보존한다. 범위가 private이면 240문항, public이면 KMMLU100이며 다른 범위는 원문 및 신규 민감 문항의 부분집합이다. 응답 단위 사실 오류를 주장 단위 오류로 바꾸어 표기하지 않는다. 원문 재검사와 신규 문항은 통계적으로 같은 코호트가 아니다.')
    for regime in sorted({r['regime'] for r in old}):
        for scope in sorted({r['scope'] for r in old if r['regime']==regime}):
            rs=[r for r in old if r['regime']==regime and r['scope']==scope]
            table(f'기존 {regime} {scope}',['조건','N','의미 정답','동시 통과','응답 사실 오류','미완료'],[[compact(r['variant']),r['total_questions'],r['semantic_correct'],r['passes'],f"{r['factual_errors']}/{r['factual_known_claims']}",r['generation_failures']] for r in rs],[56,18,24,24,30,18])
    h('부록 B 판정 사전과 분석 재현');table('공개 문항별 점수의 열 정의',['열','정의'],[['id','문항 식별자이며 비공개 원문은 포함하지 않음'],['cohort family','영역과 사전 정의 계열'],['complete','정상 EOS 종료 여부'],['semantic','과제별 성공 true false null. 정확 출력 영역만 전체 응답 일치'],['instruction','정해진 출력 계약 충족 여부'],['joint','semantic=true 그리고 instruction=true'],['method','자동 추출 또는 단일 AI 판정 경로'],['latency_seconds','generate 호출 시간, 로딩과 토큰화 제외'],['generated_tokens','해당 모델 토크나이저의 생성 토큰 수']],[46,124]);prose('기본 정확도 = 확인된 정답 수 / 전체 문항 수. 동시 통과율 = 내용 정답이며 출력 계약을 충족한 수 / 전체 문항 수. 주장 오류율 = 모순 주장 수 / 판정 가능한 주장 수. 근거 없는 주장률 = 근거 없는 주장 수 / 주석한 전체 주장 수. 쌍대 승률 = 승리 수 / 전체 쌍수이며 동률을 별도로 보고한다. 이 정의들은 서로 다른 측정 단위다.')
    h('부록 C 자료 무결성과 고정 리비전');seal=read(d/'confirmation_seal.json');prose('신규 봉인 시각은 기록된 Unix 시각으로 '+str(seal['at'])+'이다. 파일명에 쓰인 20261026은 작업 식별자이며 보고서 발행 기준일은 2026년 10월 27일이다. 봉인 이후 질문과 정답, 모델을 수정하지 않았다. 아래 값은 원문을 공개하지 않고 파일 동일성을 확인하기 위한 해시다.')
    for name in ['private.questions.jsonl','private.keys.LOCAL_ONLY.jsonl','inference.questions.jsonl','protocol.json']:
        prose(name+' SHA256 '+seal['files'][name])
    for b in bench:prose(b['dataset']+' revision '+b['revision']+' 파일 SHA256 '+b['sha256'])
    prose('base revision '+seal['base_revision']);prose('adapter_model.safetensors SHA256 '+seal['adapter_files']['adapter_model.safetensors'])
    h('부록 D 근거 자료 목록');prose('E번호는 평가 집계, H번호는 복원한 개발 이력의 공개 근거와 연결된다. 원자료의 상대 경로와 전체 해시는 evidence_manifest.json에 보존했다. 집계 파일이 원문 전체를 대신하지는 않지만, 공개한 수치와 그림은 이 파일에서 재생성할 수 있다.')
    table('공개 근거 파일',['식별자','파일'],[[s['id'],s['public_file']] for s in manifest['sources']],[48,122])
    h('부록 E AI 방법론 공동 검토와 합의');prose(REVIEW_CONSENSUS)
    (out/'AI_REVIEW_CONSENSUS_KO.md').write_text('# AI 방법론 공동 검토와 합의\n\n'+REVIEW_CONSENSUS+'\n',encoding='utf-8')
    h('참고문헌');
    for rid,title,url in REFERENCES:prose(f'[{rid}] {title}\n{url}')
    # Human-readable Markdown and portable HTML share exactly the PDF source blocks.
    md=['# CoMPaSS ARGOS 최종 연구 보고서','\n2026년 10월 27일 | 실제 실행 기록·단일 AI 문항 판정·AI 공동 방법론 검토\n']
    ht=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>CoMPaSS ARGOS 최종 연구 보고서</title><style>body{max-width:1000px;margin:48px auto;padding:0 24px;font:17px/1.85 system-ui,sans-serif;color:#182430}h1,h2{color:#000;line-height:1.4}h2{margin-top:2.8em}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #d9d9d9;padding:8px}th{background:#e7eef4}img{max-width:100%}caption{text-align:left;font-weight:bold}p{word-break:keep-all}a{overflow-wrap:anywhere}</style><h1>CoMPaSS ARGOS 최종 연구 보고서</h1><p>2026년 10월 27일 | 단일 AI 검수</p>']
    for b in blocks:
        if b[0]=='h':md+=['\n## '+b[1]+'\n'];ht+=['<h2>'+html.escape(b[1])+'</h2>']
        elif b[0]=='p':md+=[b[1]+'\n'];ht+=['<p>'+html.escape(b[1])+'</p>']
        elif b[0]=='image':
            rel='figures/'+b[1].name;md+=[f'![{b[2]}]({rel})\n'];ht+=[f'<figure><img src="{rel}"><figcaption>{html.escape(b[2])}</figcaption></figure>']
        else:
            md+=['\n'+b[1]+'\n','| '+' | '.join(b[2])+' |','| '+' | '.join(['---']*len(b[2]))+' |']+['| '+' | '.join(r)+' |' for r in b[3]]+['\n'];ht+=['<table><caption>'+html.escape(b[1])+'</caption><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in b[2])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in r)+'</tr>' for r in b[3])+'</tbody></table>']
    (out/'REPORT_KO.md').write_text('\n'.join(md),encoding='utf-8');(out/'report.html').write_text('\n'.join(ht)+ '</html>',encoding='utf-8')
    pdfmetrics.registerFont(TTFont('Korean',font));pdfmetrics.registerFont(TTFont('KoreanBold',bold));pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='KoreanBold')
    style={
        'body':ParagraphStyle('body',fontName='Korean',fontSize=10,leading=16.4,spaceAfter=8,wordWrap='CJK'),
        'h':ParagraphStyle('h',fontName='KoreanBold',fontSize=15,leading=22,spaceBefore=18,spaceAfter=11,keepWithNext=True,textColor=colors.black,wordWrap='CJK'),
        'cell':ParagraphStyle('cell',fontName='Korean',fontSize=8.2,leading=12,wordWrap='CJK'),
        'cap':ParagraphStyle('cap',fontName='Korean',fontSize=8.5,leading=13,spaceBefore=6,spaceAfter=10,wordWrap='CJK'),
        'tablecap':ParagraphStyle('tablecap',fontName='Korean',fontSize=8.5,leading=13,spaceBefore=6,spaceAfter=10,wordWrap='CJK'),
        'title':ParagraphStyle('title',fontName='KoreanBold',fontSize=27,leading=40,spaceAfter=20,textColor=colors.black),
        'toc':ParagraphStyle('toc',fontName='Korean',fontSize=9,leading=11.5,spaceBefore=0,leftIndent=0,rightIndent=20),
        'subtitle':ParagraphStyle('subtitle',fontName='KoreanBold',fontSize=15,leading=22,spaceBefore=18,spaceAfter=11,textColor=colors.black,wordWrap='CJK'),
    }
    def p(t,kind='body'):
        # Malgun's PDF subset lacks U+2212; keep negative signs visible.
        text=html.escape(t.replace('\u2212','-'))
        # Long checksums and URLs must remain printable within the text area.
        text=re.sub(r'([a-f0-9]{64})',lambda m:'<font size="8">'+'<br/>'.join(m[0][i:i+32] for i in range(0,64,32))+'</font>',text)
        return Paragraph(text,style[kind])
    class Doc(BaseDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name=='h':
                t=flowable.getPlainText();key='sec-'+hashlib.sha1(t.encode()).hexdigest()[:16];self.canv.bookmarkPage(key);self.canv.addOutlineEntry(t,key,0,False);self.notify('TOCEntry',(0,t,self.page,key))
    def page(c,doc):
        c.saveState();w,h=A4;c.setFont('Korean',8);c.setFillColor(colors.HexColor('#53606d'));c.drawString(20*mm,h-13*mm,'CoMPaSS ARGOS 연구 결과');c.drawRightString(w-20*mm,13*mm,str(doc.page));c.drawString(20*mm,13*mm,'2026.10.27  |  공개 보고서  |  AI 검수');c.restoreState()
    pdf=out/'CoMPaSS_ARGOS_final_report_20261027.pdf';doc=Doc(str(pdf),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=23*mm,bottomMargin=22*mm,title='CoMPaSS ARGOS 최종 연구 보고서',author='CoMPaSS ARGOS project; Codex AI review')
    doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='normal')],onPage=page))
    story=[Spacer(1,27*mm),p('CoMPaSS ARGOS\n최종 연구 보고서'.replace('\n',' '),'title'),p('DGX Pilot에서 임대 GPU 본실험까지','subtitle'),p('2026년 10월 27일'),Spacer(1,12*mm),p('8월 DGX 교사 수집과 로컬 Pilot, 10월 A40 48GB 임대 파일럿과 A5000 24GB 본실험, 후속 수학 보충 및 신규 동결 평가를 실제 실행 기록으로 연결하였다.'),p('핵심 결론은 영역별 성능 변화와 검증의 한계를 함께 보고하는 것이다. 문항 판정은 단일 AI로 수행했고 보고서는 웹 ChatGPT와 세 차례 방법론 자문을 거쳤다. 사람 평가자2명의 독립 판정을 완료했다고 주장하지 않는다.'),Spacer(1,15*mm),p('문항 판정·자료 조사·보고서 작성  Codex AI'),p('방법론·편집 자문  웹 ChatGPT Pro → xhigh → Pro'),p('공개 범위  보고서 집계표 도표 재현 코드'),p('비공개 보관  평가 문항 정답 모델 가중치'),PageBreak(),p('목차','subtitle')]
    toc=TableOfContents();toc.levelStyles=[style['toc']];toc.tableStyle=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]);story+=[toc,PageBreak()]
    for b in blocks:
        if b[0]=='h':
            if b[1]=='참고문헌':story.append(PageBreak())
            story.append(p(b[1],'h'))
        elif b[0]=='p':story.append(p(b[1]))
        elif b[0]=='image':
            from PIL import Image as PILImage
            w,hh=PILImage.open(b[1]).size;im=Image(str(b[1]),width=doc.width-8,height=(doc.width-8)*hh/w);lead=[]
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='h':lead.append(story.pop())
            story+=[KeepTogether(lead+[im,p(b[2],'cap')])]
        else:
            headers,rs=b[2],b[3];widths=[x*mm for x in b[4]] if b[4] else [doc.width/len(headers)]*len(headers)
            # Fit exactly to the frame's actual available width.
            factor=(doc.width-12)/sum(widths);widths=[x*factor for x in widths]
            vals=[[p(c,'cell') for c in headers]]+[[p(c,'cell') for c in r] for r in rs]
            tab=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT');commands=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
            for ix in range(2,len(vals),2):commands.append(('BACKGROUND',(0,ix),(-1,ix),colors.HexColor('#f8fafb')))
            tab.setStyle(TableStyle(commands));story+=[CondPageBreak(105),p(b[1],'tablecap'),tab,Spacer(1,10)]
    doc.multiBuild(story)
    receipt=dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),source_blocks=len(blocks),body_characters=sum(len(b[1]) for b in blocks if b[0]=='p'),generated_from_public_data_only=True)
    (out/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='reports/argos-final-20261027');p.add_argument('--font',default=r'C:\Windows\Fonts\malgun.ttf');p.add_argument('--bold-font',default=r'C:\Windows\Fonts\malgunbd.ttf');a=p.parse_args();build(a.root,a.font,a.bold_font)
