import numpy as np
from scipy.special import xlogy

def probability_sums(q,y,p_true=None,task_types=None):
    q=np.asarray(q).ravel(); y=np.asarray(y).ravel()
    if q.shape!=y.shape: raise ValueError("shape mismatch")
    bins=np.minimum((np.clip(q,0,1)*10).astype(int),9)
    out=[]
    for b in range(10):
        mask=bins==b; qb=q[mask]; yb=y[mask]
        out.append(dict(bin=b,count=int(mask.sum()),sum_q=float(qb.sum()),sum_y=float(yb.sum()),sum_brier=float(np.sum((qb-yb)**2)),sum_logloss=float(np.sum(-xlogy(yb,np.clip(qb,1e-6,1-1e-6))-xlogy(1-yb,np.clip(1-qb,1e-6,1-1e-6))))))
    return out

def aggregate_bins(rows):
    n=sum(r["count"] for r in rows)
    if not n: return dict(count=0,brier=None,logloss=None,ece=None)
    return dict(count=n,brier=sum(r["sum_brier"] for r in rows)/n,logloss=sum(r["sum_logloss"] for r in rows)/n,ece=sum(abs(r["sum_q"]-r["sum_y"]) for r in rows)/n)

def round_stats(assignment,truth,tasks,costs,oracle_expected):
    ys=truth.uniforms<truth.probabilities
    attempted=[(i,j) for i,j in enumerate(assignment.tasks) if j>=0]
    realized=sum((tasks.reward[j] if ys[i,j] else -tasks.penalty[j])-costs[i,j] for i,j in attempted)
    expected=sum(truth.probabilities[i,j]*tasks.reward[j]-(1-truth.probabilities[i,j])*tasks.penalty[j]-costs[i,j] for i,j in attempted)
    if expected-oracle_expected>1e-9: raise ValueError("oracle expected utility bound violated")
    return dict(realized=float(realized),expected=float(expected),regret=max(0.,float(oracle_expected-expected)),attempts=len(attempted),successes=sum(int(ys[i,j]) for i,j in attempted),failures=sum(int(not ys[i,j]) for i,j in attempted),abstention=(5-len(attempted))/5)
