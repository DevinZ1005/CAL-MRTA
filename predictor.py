import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .seeds import rng,integer_seed
from .simulator import ARCHETYPES,TEMPLATES,DIRECTIONS,probabilities
from .features import build
PARAMS=dict(loss="log_loss",learning_rate=.05,max_iter=150,max_leaf_nodes=15,max_depth=None,
            min_samples_leaf=40,l2_regularization=1.0,max_bins=255,early_stopping=False)

def source_split(phase,kernel,replicate,split,n):
    def g(part): return rng(phase,kernel,replicate,f"source_{split}_{part}")
    gp,gt,gb,gn,gy,ge=(g(part) for part in ("profiles","tasks","battery_positions","telemetry","labels","environment"))
    a=gp.integers(0,5,n);nominal=np.clip(ARCHETYPES[a]+gp.normal(0,.03,(n,5)),.02,.98)
    sensitivity=gp.uniform(.85,1.15,(n,5));reliability=gp.normal(0,.15,n)
    tt=gt.integers(0,5,n);req=np.clip(TEMPLATES[tt]+gt.normal(0,.05,(n,5)),.02,.98)
    difficulty=gt.uniform(0,1,n)
    # Rewards/penalties are independent task attributes, excluded from features.
    gt.uniform(2,5,n);gt.uniform(1,4,n)
    task_pos=gt.uniform(0,1,(n,2));battery=gb.uniform(.6,1,n);robot_pos=gb.uniform(0,1,(n,2))
    distance=np.linalg.norm(robot_pos-task_pos,axis=1)/np.sqrt(2)
    di=ge.integers(0,4,n);severity=ge.uniform(0,.15,n)
    degradation=severity[:,None]*np.stack(list(DIRECTIONS.values()))[di]
    tele=np.stack([np.clip(battery+gn.normal(0,.03,n),0,1),
                   np.clip(degradation[:,0]+gn.normal(0,.05,n),0,1),
                   np.clip(1-degradation[:,1]+gn.normal(0,.05,n),0,1),
                   np.clip(1-degradation[:,4]+gn.normal(0,.05,n),0,1)],axis=-1)
    p,_=probabilities(nominal[:,None,:],sensitivity[:,None,:],reliability[:,None],battery[:,None],
                     degradation[:,None,:],req[:,None,:],difficulty[:,None],distance[:,None,None],kernel)
    p=p[:,0,0];y=(gy.uniform(size=n)<p).astype(np.uint8)
    X=build(nominal,req,tele,distance,difficulty,a,tt)
    return X,y,p

def fit(X,y,phase,kernel,replicate):
    model=HistGradientBoostingClassifier(**PARAMS,random_state=integer_seed(phase,kernel,replicate,"estimator"))
    model.fit(X,y);return model

def predict(model,X):
    return np.clip(model.predict_proba(X.reshape(-1,31))[:,1].reshape(X.shape[:-1]),1e-6,1-1e-6)
