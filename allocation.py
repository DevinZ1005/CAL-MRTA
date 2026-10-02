import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Assignment

def utility_matrix(q,reward,penalty,cost):
    q=np.asarray(q);return q*np.asarray(reward)[None,:]-(1-q)*np.asarray(penalty)[None,:]-np.asarray(cost)

def maximize_utilities(util):
    """Common solver; arbitrary small shapes support exact fixture verification."""
    util=np.asarray(util,dtype=float)
    if util.ndim!=2 or min(util.shape)<1 or not np.all(np.isfinite(util)):raise ValueError("invalid utility matrix")
    n,m=util.shape
    full=np.concatenate([np.where(util>0,util,-1e6),np.zeros((n,n))],axis=1)
    rows,cols=linear_sum_assignment(-full)
    out=[-1]*n
    for r,c in zip(rows,cols):
        if c<m:out[r]=int(c)
    if len(set(x for x in out if x>=0))!=sum(x>=0 for x in out):raise RuntimeError("duplicate task")
    return Assignment(tuple(out))

def solve(q,reward,penalty,cost):
    q=np.asarray(q)
    if q.shape!=(5,8) or np.shape(cost)!=(5,8) or np.shape(reward)!=(8,) or np.shape(penalty)!=(8,):raise ValueError("invalid assignment shape")
    if not np.all(np.isfinite(q)) or np.any((q<0)|(q>1)):raise ValueError("invalid probability")
    return maximize_utilities(utility_matrix(q,reward,penalty,cost))

def selected_expected(assignment,p,reward,penalty,cost):
    u=utility_matrix(p,reward,penalty,cost)
    return sum(float(u[i,j]) for i,j in enumerate(assignment.tasks) if j>=0)

def selected_realized(assignment,outcomes,reward,penalty,cost):
    return sum(float((reward[j] if outcomes[i,j] else -penalty[j])-cost[i,j]) for i,j in enumerate(assignment.tasks) if j>=0)
