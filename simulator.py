"""Exogenous repeated dispatch batches. Truth never enters DecisionInput."""
import numpy as np
from scipy.special import expit
from .types import RobotProfile, RobotInternal, TaskBatch, DecisionInput, SimulationTruth, sealed
from .seeds import rng

ARCHETYPES = np.array([[.95,.80,.20,.25,.80],[.60,.95,.20,.40,.90],[.65,.40,.95,.55,.70],[.50,.60,.50,.95,.70],[.72,.72,.65,.65,.75]])
TEMPLATES = np.array([[.70,.80,.10,.10,.45],[.40,.90,.10,.25,.55],[.55,.25,.85,.40,.30],[.30,.50,.40,.90,.40],[.60,.65,.55,.60,.60]])
DIRECTIONS = {"terrain": np.array([.60,.05,.05,.10,.05]), "sensor": np.array([.05,.70,.00,.05,.05]), "network": np.array([.00,.05,.00,.00,.70]), "mixed": np.array([.35,.40,.15,.30,.35])}
CONDITIONS = {"no_change": ("mixed",.05), **{f"{name}_{int(sev*100):03d}":(name,sev) for name in DIRECTIONS for sev in (.35,.70)}}

def profiles(phase,kernel,replicate,episode):
    g=rng(phase,kernel,replicate,"profiles",episode)
    return tuple(RobotInternal(RobotProfile(i,i,sealed(np.clip(ARCHETYPES[i]+g.normal(0,.03,5),.02,.98))), sealed(g.uniform(.85,1.15,5)),float(g.normal(0,.15))) for i in range(5))

def probabilities(nominal, sensitivity, reliability, battery, degradation, requirements, difficulty, distances, kernel):
    """Broadcast arrays with pair axes (..., robot, task, capability)."""
    eff=nominal*np.clip(1-degradation*sensitivity,0,1)
    eff=np.array(eff,copy=True)
    eff[..., [0,2,3]] *= (.75+.25*battery)[...,None]
    m=eff[..., :,None,:]-requirements[...,None,:,:]
    if kernel=="A":
        w=(requirements+.05)/(requirements+.05).sum(axis=-1,keepdims=True)
        z=1.2+3*np.sum(w[...,None,:,:]*m,axis=-1)-8*np.sum(w[...,None,:,:]*np.maximum(0,-m)**2,axis=-1)-.8*distances+.8*(battery[..., :,None]-.5)-.5*difficulty[...,None,:]+reliability[..., :,None]
    elif kernel=="B":
        critical=np.argsort(-requirements,axis=-1,kind="stable")[...,:2]
        idx=np.broadcast_to(critical[...,None,:,:],m.shape[:-1]+(2,))
        bottleneck=np.take_along_axis(m,idx,axis=-1).min(axis=-1)
        z=1.5+5*bottleneck-.6*distances-.4*difficulty[...,None,:]+reliability[..., :,None]
    else: raise ValueError(kernel)
    return np.clip(expit(z),.02,.98),eff

def episode(phase,kernel,replicate,episode_id,condition,rounds=160,fixed_environment=None):
    robots=profiles(phase,kernel,replicate,episode_id)
    gt=rng(phase,kernel,replicate,"tasks",episode_id)
    gb=rng(phase,kernel,replicate,"battery_positions",episode_id)
    gn=rng(phase,kernel,replicate,"telemetry",episode_id)
    gu=rng(phase,kernel,replicate,"outcomes",episode_id)
    tt=gt.integers(0,5,(rounds,8))
    req=np.clip(TEMPLATES[tt]+gt.normal(0,.05,(rounds,8,5)),.02,.98)
    difficulty=gt.uniform(0,1,(rounds,8)); reward=gt.uniform(2,5,(rounds,8)); penalty=gt.uniform(1,4,(rounds,8))
    task_pos=gt.uniform(0,1,(rounds,8,2)); battery=gb.uniform(.6,1,(rounds,5)); robot_pos=gb.uniform(0,1,(rounds,5,2))
    distances=np.linalg.norm(robot_pos[:,:,None,:]-task_pos[:,None,:,:],axis=-1)/np.sqrt(2)
    costs=.15+.60*distances+.10*difficulty[:,None,:]
    degradation=np.broadcast_to(.05*DIRECTIONS["mixed"],(rounds,5)).copy()
    if condition!="no_change": degradation[40:]=CONDITIONS[condition][1]*DIRECTIONS[CONDITIONS[condition][0]]
    if fixed_environment is not None: degradation[:]=fixed_environment[1]*DIRECTIONS[fixed_environment[0]]
    nominal=np.stack([x.profile.nominal for x in robots]); sensitivity=np.stack([x.sensitivity for x in robots]); reliability=np.array([x.reliability for x in robots])
    p,eff=probabilities(np.broadcast_to(nominal,(rounds,5,5)),np.broadcast_to(sensitivity,(rounds,5,5)),np.broadcast_to(reliability,(rounds,5)),battery,degradation[:,None,:],req,difficulty,distances,kernel)
    telemetry=np.stack([np.clip(battery+gn.normal(0,.03,(rounds,5)),0,1),np.clip(degradation[:,0,None]+gn.normal(0,.05,(rounds,5)),0,1),np.clip(1-degradation[:,1,None]+gn.normal(0,.05,(rounds,5)),0,1),np.clip(1-degradation[:,4,None]+gn.normal(0,.05,(rounds,5)),0,1)],axis=-1)
    u=gu.uniform(0,1,(rounds,5,8))
    public=[]; truth=[]
    for t in range(rounds):
        batch=TaskBatch(sealed(t*8+np.arange(8),int),sealed(tt[t],int),sealed(req[t]),sealed(difficulty[t]),sealed(reward[t]),sealed(penalty[t]),sealed(task_pos[t]))
        public.append((batch,sealed(telemetry[t]),sealed(distances[t]),sealed(costs[t])))
        truth.append(SimulationTruth(sealed(battery[t]),sealed(degradation[t]),sealed(eff[t]),sealed(p[t]),sealed(u[t])))
    return tuple(x.profile for x in robots), public, truth
