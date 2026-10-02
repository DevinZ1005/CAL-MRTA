"""SeedSequence([master, phase, kernel, replicate, stream, episode])."""
import numpy as np
PHASE=dict(smoke=1,pilot=2,final_source=3,final_evaluation=4,distortion=5,bootstrap=6)
KERNEL={"A":1,"B":2}
STREAM=dict(profiles=1,tasks=2,battery_positions=3,telemetry=4,outcomes=5,
            bootstrap=10,estimator=12)
for split,base in (("train",100),("cal",200),("iid",300)):
    for offset,part in enumerate(("profiles","tasks","battery_positions","telemetry","labels","environment"),1):
        STREAM[f"source_{split}_{part}"]=base+offset
MASTER=314159

def seed_tuple(phase,kernel,replicate,stream,episode=0,master=MASTER):
    return [master,PHASE[phase],KERNEL[kernel],replicate,STREAM[stream],episode]

def rng(phase,kernel,replicate,stream,episode=0,master=MASTER):
    return np.random.default_rng(np.random.SeedSequence(seed_tuple(phase,kernel,replicate,stream,episode,master)))

def integer_seed(phase,kernel,replicate,stream,episode=0,master=MASTER):
    return int(rng(phase,kernel,replicate,stream,episode,master).integers(0,2**31-1))
