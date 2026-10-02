from collections import deque
import hashlib
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

BOUNDS=((0.,5.),(-6.,6.))

def transform(raw,params):
    return expit(params[0]*logit(np.clip(raw,1e-6,1-1e-6))+params[1])

def objective(params,x,y,reference,lam,offset=False):
    a,b=(reference[0],params[0]) if offset else params
    z=a*x+b; residual=expit(z)-y
    loss=np.mean(np.logaddexp(0,z)-y*z)+lam/2*((a-reference[0])**2+(b-reference[1])**2)
    grad=np.array([np.mean(residual)+lam*(b-reference[1])]) if offset else np.array([np.mean(residual*x)+lam*(a-reference[0]),np.mean(residual)+lam*(b-reference[1])])
    return float(loss),grad

def fit(raw,y,reference=(1.,0.),lam=.001,offset=False):
    raw=np.asarray(raw,dtype=float); y=np.asarray(y,dtype=float)
    if len(raw)==0 or raw.shape!=y.shape or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(y)): raise ValueError("invalid calibration data")
    x=logit(np.clip(raw,1e-6,1-1e-6))
    start=np.array([reference[1]]) if offset else np.array(reference)
    bounds=[BOUNDS[1]] if offset else BOUNDS
    result=minimize(objective,start,args=(x,y,reference,lam,offset),method="L-BFGS-B",jac=True,bounds=bounds,options={"maxiter":100,"ftol":1e-10,"gtol":1e-7})
    if not result.success: raise RuntimeError(f"L-BFGS-B failed: {result.status} {result.message}")
    return (float(reference[0]),float(result.x[0])) if offset else tuple(map(float,result.x))

class OnlineCalibrator:
    def __init__(self,source,offset=False,minimum=50,interval=25,capacity=250):
        self.source=tuple(source); self.offset=offset; self.minimum=minimum; self.interval=interval; self.capacity=capacity; self.reset()
    def reset(self):
        self.params=self.source; self.buffer=deque(maxlen=self.capacity); self.count=0; self.last_fit=0; self.events=[]
    def predict(self,raw): return transform(raw,self.params)
    def observe(self,feedback):
        for f in sorted(feedback,key=lambda x:(x.round_index,x.robot_id)):
            self.buffer.append(f); self.count+=1
        eligible=self.count>=self.minimum and (self.last_fit==0 or self.count-self.last_fit>=self.interval)
        if not feedback or not eligible: return
        raw=np.array([f.raw_probability for f in self.buffer]); y=np.array([f.outcome for f in self.buffer])
        digest=hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()
        event=dict(outcomes=self.count,size=len(self.buffer),oldest_round=self.buffer[0].round_index,newest_round=self.buffer[-1].round_index,input_hash=digest)
        started=time.perf_counter()
        try:
            self.params=fit(raw,y,self.source,.01,self.offset)
            event.update(status="ok",a=self.params[0],b=self.params[1],bound_hit=any(abs(v-b)<1e-8 for v,lims in zip(self.params,BOUNDS) for b in lims))
        except Exception as exc:
            event.update(status="failed",error=str(exc),a=self.params[0],b=self.params[1],bound_hit=False)
        event["seconds"]=time.perf_counter()-started
        event["retained_feedback_age_rounds"]=event["newest_round"]-event["oldest_round"]
        self.last_fit=self.count; self.events.append(event)
