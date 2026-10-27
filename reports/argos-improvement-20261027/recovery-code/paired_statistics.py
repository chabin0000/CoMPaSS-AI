"""Prespecified paired statistics; conservative retention lower bound.

The retention bound is a Bonferroni combination of exact marginal bounds on
discordant-pair probabilities P(student-only) and P(baseline-only). It remains
nondegenerate at zero discordance. For synthetic target data, family-cluster
bootstrap and exact cluster sign-flips preserve paired observations. They assume
exchangeable family units; only 16 designed families limits generalization.
"""
import math,random,statistics
from itertools import product
def binom_cdf(k,n,p):
 return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))
def cp_upper(k,n,tail):
 if k==n:return 1.
 lo,hi=0.,1.
 for _ in range(70):
  mid=(lo+hi)/2
  if binom_cdf(k,n,mid)>tail:lo=mid
  else:hi=mid
 return (lo+hi)/2
def cp_lower(k,n,tail):return 0. if k==0 else 1-cp_upper(n-k,n,tail)
def retention_lower(a,b,alpha=.05):
 assert len(a)==len(b)>0;n=len(a);win=sum(bool(x) and not y for x,y in zip(a,b));loss=sum(bool(y) and not x for x,y in zip(a,b))
 lower=cp_lower(win,n,alpha/2)-cp_upper(loss,n,alpha/2)
 return dict(n=n,wins=win,losses=loss,difference=(sum(a)-sum(b))/n,one_sided_95_lower=lower,noninferiority_margin=-.05,noninferiority_confirmed=lower>-.05,method='Conservative exact discordant-marginal Bonferroni lower bound')
def cluster_comparison(a,b,families,seed=82717):
 assert len(a)==len(b)==len(families)>0
 names=sorted(set(families));groups=[[int(x)-int(y) for x,y,f in zip(a,b,families) if f==name] for name in names]
 sums=[sum(g) for g in groups];ns=[len(g) for g in groups];n=sum(ns);obs=sum(sums)/n
 assert len(names)<=20
 extreme=sum(sum(s*v for s,v in zip(signs,sums))/n>=obs-1e-14 for signs in product([-1,1],repeat=len(names)))
 p=extreme/(2**len(names));rng=random.Random(seed);samples=[]
 for _ in range(10000):
  ix=[rng.randrange(len(names)) for __ in names];samples.append(sum(sums[i] for i in ix)/sum(ns[i] for i in ix))
 samples.sort();return dict(n=n,family_count=len(names),difference=obs,cluster_bootstrap_95=[samples[249],samples[9749]],cluster_sign_flip_p_one_sided=p,replicates=10000,assumption='paired family exchangeability; limited designed-family generalization')
def holm(pvalues):
 order=sorted(pvalues,key=pvalues.get);adjusted={};previous=0
 for i,name in enumerate(order):previous=max(previous,min(1.,(len(order)-i)*pvalues[name]));adjusted[name]=previous
 return adjusted
if __name__=='__main__':
 same=retention_lower([1]*40,[1]*40);assert same['one_sided_95_lower']<-.05 and not same['noninferiority_confirmed']
 assert abs(cp_upper(0,40,.025)-(1-.025**(1/40)))<1e-10
 c=cluster_comparison([1]*16,[0]*16,list(range(16)));assert c['cluster_sign_flip_p_one_sided']==1/65536
 assert holm({'a':.01,'b':.03})=={'a':.02,'b':.03}
 print('PAIRED_STATISTICS_EDGE_CASES_VERIFIED')
