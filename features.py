"""Exact 31-feature order. Supports one or many rounds."""
import numpy as np
SCHEMA = [f"nominal_{i}" for i in range(5)]+[f"requirement_{i}" for i in range(5)]+[f"margin_{i}" for i in range(5)]+["battery","terrain","sensor_quality","network_quality","distance","difficulty"]+[f"archetype_{i}" for i in range(5)]+[f"task_type_{i}" for i in range(5)]
assert len(SCHEMA)==31

def build(nominal,requirements,telemetry,distances,difficulty,archetype,task_type):
    """Inputs broadcast to a common pair shape; final axis is capability or telemetry."""
    nominal,requirements,telemetry=np.asarray(nominal),np.asarray(requirements),np.asarray(telemetry)
    shape=np.broadcast_shapes(nominal.shape[:-1],requirements.shape[:-1],telemetry.shape[:-1],np.shape(distances),np.shape(difficulty),np.shape(archetype),np.shape(task_type))
    n=np.broadcast_to(nominal,shape+(5,)); r=np.broadcast_to(requirements,shape+(5,)); tele=np.broadcast_to(telemetry,shape+(4,))
    return np.concatenate([n,r,n-r,tele,np.broadcast_to(distances,shape)[...,None],np.broadcast_to(difficulty,shape)[...,None],np.eye(5)[np.broadcast_to(archetype,shape)],np.eye(5)[np.broadcast_to(task_type,shape)]],axis=-1)

def decision_features(decision):
    n=np.stack([p.nominal for p in decision.profiles]); a=np.array([p.archetype for p in decision.profiles])
    return build(n[:,None,:],decision.tasks.requirements[None,:,:],decision.telemetry[:,None,:],decision.distances,decision.tasks.difficulty[None,:],a[:,None],decision.tasks.types[None,:])
