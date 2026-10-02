"""Engine-owned truth, selective feedback, and atomic episode work units."""
import json,pickle,time
from collections import defaultdict
from pathlib import Path
import io
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from . import predictor
from .allocation import solve,selected_expected,utility_matrix
from .artifacts import atomic_bytes,atomic_json,file_sha,initialize,verify,write_csv,atomic_csv,canonical,sha
from .calibration import fit,transform
from .features import build,SCHEMA
from .policies import Policy,NAMES
from .seeds import seed_tuple,integer_seed
from .simulator import episode
from .types import DecisionInput,Feedback

POLICIES=(*NAMES,'oracle')
BIN_FIELDS=('count','sum_q','sum_y','sum_brier','sum_logloss','sum_estimation_squared')
ROUND_FIELDS=('kernel','replicate','condition','episode','policy','round','realized','expected','regret','attempts','successes','failures','abstention','success_fraction','disagreement','estimation_mse','positive_utility_fraction')
EVENT_FIELDS=('kernel','replicate','condition','episode','policy','outcomes','size','oldest_round','newest_round','input_hash','status','a','b','bound_hit','seconds','retained_feedback_age_rounds','error')

def npz_bytes(**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()

def pair_bins(q,y,task_types,p_true):
    q=np.asarray(q).ravel();y=np.asarray(y).ravel().astype(float);p_true=np.asarray(p_true).ravel()
    kinds=np.asarray(task_types,dtype=int).ravel()
    index=kinds*10+np.minimum((np.clip(q,0,1)*10).astype(int),9)
    clipped=np.clip(q,1e-6,1-1e-6);losses=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    return np.stack([np.bincount(index,weights=w,minlength=50) for w in (np.ones(len(q)),q,y,(q-y)**2,losses,(q-p_true)**2)],axis=-1).reshape(5,10,6)

def source_phase(c):return 'final_source' if c['phase']=='final' else c['phase']
def eval_phase(c):return 'final_evaluation' if c['phase']=='final' else c['phase']
def model_paths(run,k,r):
    stem=f'{k}_{r:02d}';return Path(run)/'models'/(stem+'.pkl'),Path(run)/'models'/(stem+'.json')
def job_id(k,r,c,e):return f'{k}_{r:02d}_{c}_{e:02d}'
def job_paths(run,key):return Path(run)/'traces'/(key+'.npz'),Path(run)/'traces'/(key+'.json')

def prepare_source(run,c,k,r):
    model_file,meta_file=model_paths(run,k,r)
    if meta_file.exists():
        meta=json.loads(meta_file.read_text())
        if file_sha(model_file)!=meta['model_hash']:raise ValueError('corrupt model')
        for split,info in meta['splits'].items():
            if file_sha(Path(run)/'data'/f'{k}_{r:02d}_{split}.npz')!=info['file_hash']:raise ValueError('corrupt source data')
        return pickle.loads(model_file.read_bytes()),tuple(meta['source_params']),meta
    phase=source_phase(c);splits={};datasets={};t0=time.perf_counter()
    for split in ('train','cal','iid'):
        n=c['source_'+split];X,y,p=predictor.source_split(phase,k,r,split,n)
        path=Path(run)/'data'/f'{k}_{r:02d}_{split}.npz';atomic_bytes(path,npz_bytes(X=X,y=y,p_true=p))
        datasets[split]=(X,y,p)
        splits[split]=dict(size=n,file_hash=file_sha(path),feature_hash=sha(X.tobytes()),label_hash=sha(y.tobytes()),seeds={part:seed_tuple(phase,k,r,f'source_{split}_{part}') for part in ('profiles','tasks','battery_positions','telemetry','labels','environment')})
    generation_seconds=time.perf_counter()-t0
    X,y,_=datasets['train'];ts=time.perf_counter();model=predictor.fit(X,y,phase,k,r);fit_seconds=time.perf_counter()-ts
    calX,caly,_=datasets['cal'];ts=time.perf_counter();calraw=predictor.predict(model,calX);source_params=fit(calraw,caly);cal_seconds=time.perf_counter()-ts
    # IID diagnostics are read only after classifier and source calibration fitting.
    iidX,iidy,iidtrue=datasets['iid'];ts=time.perf_counter();iidraw=predictor.predict(model,iidX);iidq=transform(iidraw,source_params)
    iid={}
    for name,q in (('raw',iidraw),('source_platt',iidq)):
        bins=pair_bins(q,iidy,np.argmax(iidX[:,-5:],axis=1),iidtrue).sum(axis=0)
        iid[name]=dict(brier=float(bins[:,3].sum()/len(q)),logloss=float(bins[:,4].sum()/len(q)),ece=float(np.abs(bins[:,1]-bins[:,2]).sum()/len(q)),probability_estimation_mse=float(bins[:,5].sum()/len(q)),bins=bins.tolist())
    iid_seconds=time.perf_counter()-ts
    atomic_bytes(model_file,pickle.dumps(model,pickle.HIGHEST_PROTOCOL))
    meta=dict(kernel=k,replicate=r,source_params=source_params,source_bound_hit=any(abs(v-b)<1e-8 for v,lims in zip(source_params,((0,5),(-6,6))) for b in lims),model_hash=file_sha(model_file),splits=splits,feature_schema=SCHEMA,resolved_estimator_params=model.get_params(),iid=iid,timing=dict(generation=generation_seconds,estimator_fit=fit_seconds,source_calibration=cal_seconds,iid=iid_seconds,total=time.perf_counter()-t0))
    atomic_json(meta_file,meta);return model,source_params,meta

def episode_features(profiles,public):
    nom=np.stack([p.nominal for p in profiles]);arch=np.array([p.archetype for p in profiles])
    return np.stack([build(nom[:,None,:],batch.requirements[None,:,:],tele[:,None,:],dist,batch.difficulty[None,:],arch[:,None],batch.types[None,:]) for batch,tele,dist,cost in public])

def play_episode(profiles,public,truth,raw_all,source,policy_order=NAMES,collect=True,audit=False):
    """Normal Policy constructors/predict/observe have no simulator or truth reference."""
    if set(policy_order)!=set(NAMES) or len(policy_order)!=4:raise ValueError('policy order')
    policies={name:Policy(name,source) for name in policy_order};T=len(public);P=5
    selected=np.full((P,T,5),-1,dtype=np.int16);selected_raw=np.full((P,T,5),np.nan);selected_q=np.full((P,T,5),np.nan);outcome=np.full((P,T,5),-1,dtype=np.int8)
    pair_realized=np.zeros((P,T,5));pair_expected=np.zeros((P,T,5));regret=np.zeros((P,T));params=np.full((P,T,2),np.nan)
    bin_stats=np.zeros((2,P,T,5,10,6)) if collect else None
    positive_fraction=np.zeros((P,T));q_audit=np.zeros((P,T,5,8)) if audit else None
    allocation_seconds=0.;observe_seconds=0.
    for t,(batch,tele,dist,cost) in enumerate(public):
        d=DecisionInput(t,profiles,tele,batch,dist,cost,raw_all[t]);tr=truth[t];y=tr.uniforms<tr.probabilities
        ta=time.perf_counter();oracle=solve(tr.probabilities,batch.reward,batch.penalty,cost);allocation_seconds+=time.perf_counter()-ta
        optimum=selected_expected(oracle,tr.probabilities,batch.reward,batch.penalty,cost)
        for name in (*policy_order,'oracle'):
            pi=POLICIES.index(name);policy=policies.get(name)
            # Privileged evaluator path is separate from normal policy prediction.
            q=tr.probabilities if policy is None else policy.predict_probabilities(d)
            if policy:params[pi,t]=policy.params
            if audit:q_audit[pi,t]=q
            if policy:
                ta=time.perf_counter();assignment=solve(q,batch.reward,batch.penalty,cost);allocation_seconds+=time.perf_counter()-ta
            else:assignment=oracle
            selected[pi,t]=assignment.tasks
            positive_fraction[pi,t]=np.mean(utility_matrix(q,batch.reward,batch.penalty,cost)>0)
            # Candidate statistics use the parameters available at round start.
            if collect:bin_stats[0,pi,t]=pair_bins(q,y,np.broadcast_to(batch.types[None,:],(5,8)),tr.probabilities)
            ids=np.flatnonzero(np.array(assignment.tasks)>=0);js=np.array(assignment.tasks)[ids]
            selected_raw[pi,t,ids]=raw_all[t,ids,js];selected_q[pi,t,ids]=q[ids,js];outcome[pi,t,ids]=y[ids,js]
            pair_realized[pi,t,ids]=np.where(y[ids,js],batch.reward[js],-batch.penalty[js])-cost[ids,js]
            pair_expected[pi,t,ids]=tr.probabilities[ids,js]*batch.reward[js]-(1-tr.probabilities[ids,js])*batch.penalty[js]-cost[ids,js]
            delta=optimum-pair_expected[pi,t].sum()
            if delta < -1e-9:raise ValueError('oracle bound violated')
            regret[pi,t]=max(0.,delta)
            if collect:bin_stats[1,pi,t]=pair_bins(q[ids,js],y[ids,js],batch.types[js],tr.probabilities[ids,js])
            if policy:
                feedback=tuple(Feedback(t,int(i),int(batch.ids[j]),float(raw_all[t,i,j]),int(y[i,j])) for i,j in zip(ids,js))
                ts=time.perf_counter();policy.observe(feedback);observe_seconds+=time.perf_counter()-ts
    events=[]
    for name,p in policies.items():
        if p.calibrator:
            events.extend(dict(policy=name,**ev) for ev in p.calibrator.events)
    attempts=(selected>=0).sum(axis=-1).astype(np.int8);successes=(outcome==1).sum(axis=-1).astype(np.int8)
    trace=dict(round_ids=np.arange(T),robot_ids=np.arange(5),policy_ids=np.array(POLICIES),selected=selected,selected_raw=selected_raw,selected_q=selected_q,outcome=outcome,pair_realized=pair_realized,pair_expected=pair_expected,realized=pair_realized.sum(axis=-1),expected=pair_expected.sum(axis=-1),regret=regret,attempts=attempts,successes=successes,failures=attempts-successes,parameters=params,positive_utility_fraction=positive_fraction,disagreement=(selected!=selected[1:2]).sum(axis=-1),task_ids=np.stack([x[0].ids for x in public]),task_types=np.stack([x[0].types for x in public]))
    if collect:trace['bin_stats']=bin_stats;trace['estimation_mse']=bin_stats[0,:,:,:,:,5].sum(axis=(-1,-2))/40
    if audit:trace['p_raw']=raw_all;trace['q_all']=q_audit
    return trace,events,dict(allocation=allocation_seconds,online_fitting=sum(e['seconds'] for e in events),online_observe=observe_seconds)

def episode_rows(trace,k,r,condition,ep,events):
    rows=[]
    for pi,name in enumerate(POLICIES):
        row=dict(kernel=k,replicate=r,condition=condition,episode=ep,policy=name)
        for period,sl in (('pre',slice(0,40)),('post',slice(40,160)),('first40',slice(40,80)),('last40',slice(120,160))):
            for metric in ('realized','expected','regret','attempts','successes','failures','disagreement'):
                row[period+'_'+metric]=float(trace[metric][pi,sl].mean())
            attempted=int(trace['attempts'][pi,sl].sum());succeeded=int(trace['successes'][pi,sl].sum())
            row[period+'_abstention']=float(1-trace['attempts'][pi,sl].mean()/5)
            row[period+'_success_fraction']=succeeded/attempted if attempted else None
            row[period+'_attempted_count']=attempted
        own=[ev for ev in events if ev['policy']==name]
        row.update(attempted_count=int(trace['attempts'][pi].sum()),refits=len(own),bound_hits=sum(ev['bound_hit'] for ev in own),fitting_failures=sum(ev['status']!='ok' for ev in own))
        rows.append(row)
    return rows

def evaluate(run,c,model,source,k,r,condition,ep):
    start=time.perf_counter();profiles,public,truth=episode(eval_phase(c),k,r,ep,condition)
    X=episode_features(profiles,public);tp=time.perf_counter();raw_all=predictor.predict(model,X);prediction_seconds=time.perf_counter()-tp
    trace,events,timing=play_episode(profiles,public,truth,raw_all,source,audit=ep==0)
    for event in events:event.update(kernel=k,replicate=r,condition=condition,episode=ep)
    key=job_id(k,r,condition,ep);trace_file,meta_file=job_paths(run,key)
    trace['regeneration']=np.array([eval_phase(c),k,str(r),condition,str(ep)])
    tw=time.perf_counter();atomic_bytes(trace_file,npz_bytes(**trace));truth_hash=None
    if ep==0:
        path=Path(run)/'truth'/(key+'.npz')
        atomic_bytes(path,npz_bytes(p_true=np.stack([x.probabilities for x in truth]),uniforms=np.stack([x.uniforms for x in truth]),true_battery=np.stack([x.battery for x in truth]),effective=np.stack([x.effective for x in truth]),degradation=np.stack([x.degradation for x in truth])))
        truth_hash=file_sha(path)
    timing.update(prediction=prediction_seconds,artifact=time.perf_counter()-tw,total=time.perf_counter()-start)
    meta=dict(key=key,kernel=k,replicate=r,condition=condition,episode=ep,phase=eval_phase(c),trace_hash=file_sha(trace_file),truth_hash=truth_hash,episode_rows=episode_rows(trace,k,r,condition,ep,events),fit_events=events,timing=timing,distribution=dict(quantile_levels=[0,.01,.1,.25,.5,.75,.9,.99,1],p_true_quantiles=np.quantile(np.stack([x.probabilities for x in truth]),[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),p_raw_quantiles=np.quantile(raw_all,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist()))
    atomic_json(meta_file,meta)
    return meta

def distortion(run,c):
    rows=[];t0=time.perf_counter()
    for k in c['kernels']:
        for r in range(c['replicates']):
            for ep in range(c['distortion_batches']):
                _,public,truth=episode('distortion',k,r,ep,'no_change',rounds=1,fixed_environment=('mixed',.70))
                batch,tele,dist,cost=public[0];tr=truth[0]
                oracle=solve(tr.probabilities,batch.reward,batch.penalty,cost);optimum=selected_expected(oracle,tr.probabilities,batch.reward,batch.penalty,cost)
                for bias in c['distortion_biases']:
                    assn=oracle if bias==0 else solve(expit(logit(tr.probabilities)+bias),batch.reward,batch.penalty,cost)
                    value=selected_expected(assn,tr.probabilities,batch.reward,batch.penalty,cost)
                    if optimum-value < -1e-9:raise ValueError('distortion oracle bound')
                    if bias==0 and abs(value-optimum)>1e-9:raise ValueError('zero bias is not oracle')
                    rows.append(dict(kernel=k,replicate=r,batch=ep,bias=bias,expected=value,regret=max(0.,optimum-value),attempts=sum(x>=0 for x in assn.tasks),disagreement=sum(x!=z for x,z in zip(assn.tasks,oracle.tasks))))
    write_csv(Path(run)/'distortion.csv',rows,list(rows[0]))
    atomic_json(Path(run)/'distortion_manifest.json',dict(rows=len(rows),shared_batches=len(rows)//5,assignment_solves=len(rows),file_hash=file_sha(Path(run)/'distortion.csv'),seconds=time.perf_counter()-t0,phase=5))

def check_job(run,key,hashes):
    trace,meta=job_paths(run,key)
    if key not in hashes:raise ValueError(f'missing committed hashes: {key}')
    if file_sha(trace)!=hashes[key]['trace'] or file_sha(meta)!=hashes[key]['metadata']:raise ValueError(f'corrupt completed job: {key}')
    if hashes[key].get('truth') and file_sha(Path(run)/'truth'/(key+'.npz'))!=hashes[key]['truth']:raise ValueError(f'corrupt truth: {key}')

def aggregate(run,c):
    run=Path(run);jobs=json.loads((run/'jobs.json').read_text());hashes=json.loads((run/'job_hashes.json').read_text())
    bins=defaultdict(lambda:np.zeros(6));ne=nr=0;episode_all=[];event_all=[]
    with atomic_csv(run/'round_metrics.csv.gz',ROUND_FIELDS) as writer:
        for key,status in jobs.items():
            if status!='complete':continue
            check_job(run,key,hashes);trace_file,meta_file=job_paths(run,key);meta=json.loads(meta_file.read_text())
            episode_all.extend(meta['episode_rows']);event_all.extend(meta['fit_events']);ne+=5
            with np.load(trace_file) as z:
                k,r,cond,ep=(meta[x] for x in ('kernel','replicate','condition','episode'));bs=z['bin_stats']
                arrays={name:z[name] for name in ('realized','expected','regret','attempts','successes','failures','disagreement','estimation_mse','positive_utility_fraction')}
                for pi,name in enumerate(POLICIES):
                    for t in range(160):
                        row=dict(kernel=k,replicate=r,condition=cond,episode=ep,policy=name,round=t,**{metric:float(arr[pi,t]) for metric,arr in arrays.items()})
                        n=int(arrays['attempts'][pi,t]);row.update(abstention=(5-n)/5,success_fraction=float(arrays['successes'][pi,t])/n if n else None)
                        writer.writerow(row);nr+=1
                for pop in range(2):
                    for pi,name in enumerate(POLICIES):
                        for period,start,end in (('before',0,40),('after',40,160),('last40',120,160)):
                            total=bs[pop,pi,start:end].sum(axis=0)
                            for tt in range(-1,5):
                                value=total.sum(axis=0) if tt==-1 else total[tt]
                                for b in range(10):bins[(k,cond,name,'candidate' if pop==0 else 'attempted',period,tt,b)]+=value[b]
    if episode_all:
        write_csv(run/'episode_metrics.csv',episode_all,list(episode_all[0]))
        write_csv(run/'fit_events.csv.gz',({f:e.get(f) for f in EVENT_FIELDS} for e in event_all),EVENT_FIELDS)
        rows=[dict(kernel=k,condition=cond,policy=name,population=pop,period=period,task_type=tt,bin=b,**{f:(int(v[i]) if i==0 else float(v[i])) for i,f in enumerate(BIN_FIELDS)}) for (k,cond,name,pop,period,tt,b),v in sorted(bins.items())]
        write_csv(run/'calibration_bins.csv',rows,list(rows[0]))
    atomic_json(run/'aggregation_manifest.json',{name:file_sha(run/name) for name in ('episode_metrics.csv','round_metrics.csv.gz','fit_events.csv.gz','calibration_bins.csv')})
    return ne,nr

def run(config=None,output=None,frozen_run=None,resume=False):
    if frozen_run:
        run=Path(frozen_run);m,c=verify(run)
        if not m['frozen']:raise ValueError('not frozen')
    else:
        from .config import load
        c=load(config)
        if c['phase']=='final':raise ValueError('final evaluation requires freeze and --frozen-run')
        run=Path(output);initialize(run,c)
    jobs_file=run/'jobs.json';hashes_file=run/'job_hashes.json'
    expected={job_id(k,r,cond,ep):'pending' for k in c['kernels'] for r in range(c['replicates']) for cond in c['conditions'] for ep in range(c['episodes'])}
    jobs=json.loads(jobs_file.read_text()) if jobs_file.exists() else expected
    if set(jobs)!=set(expected):raise ValueError('job matrix drift')
    if any(v!='pending' for v in jobs.values()) and not resume:raise ValueError('existing jobs require --resume')
    hashes=json.loads(hashes_file.read_text());atomic_json(jobs_file,jobs);source_manifest=[];t0=time.perf_counter()
    previous_sources={}
    if (run/'source_data_manifest.json').exists():
        for block in json.loads((run/'source_data_manifest.json').read_text())['blocks']:
            sm=block['metadata'];previous_sources[(sm['kernel'],sm['replicate'])]=block['metadata_hash']
    with threadpool_limits(limits=1):
        for k in c['kernels']:
            for r in range(c['replicates']):
                try:
                    if (k,r) in previous_sources and file_sha(model_paths(run,k,r)[1])!=previous_sources[(k,r)]:raise ValueError('source metadata changed on resume')
                    model,source,meta=prepare_source(run,c,k,r)
                except Exception as exc:
                    with (run/'failure_log.jsonl').open('a') as f:f.write(json.dumps(dict(source=[k,r],error=repr(exc)))+'\n')
                    raise
                source_manifest.append(dict(metadata=meta,metadata_hash=file_sha(model_paths(run,k,r)[1])))
                atomic_json(run/'source_data_manifest.json',dict(feature_schema=SCHEMA,blocks=source_manifest,phase=source_phase(c)))
                for cond in c['conditions']:
                    for ep in range(c['episodes']):
                        key=job_id(k,r,cond,ep);trace_file,meta_file=job_paths(run,key)
                        if jobs[key]=='complete':
                            check_job(run,key,hashes)
                            if any(e['status']!='ok' for e in json.loads(meta_file.read_text())['fit_events']):raise ValueError('unresolved cached fitting failure')
                            continue
                        if jobs[key] in ('running','failed'):
                            with (run/'failure_log.jsonl').open('a') as f:f.write(json.dumps(dict(job=key,previous_status=jobs[key],action='deterministic regeneration of incomplete work unit'))+'\n')
                        jobs[key]='running';atomic_json(jobs_file,jobs)
                        try:
                            meta=evaluate(run,c,model,source,k,r,cond,ep)
                            hashes[key]=dict(trace=file_sha(trace_file),metadata=file_sha(meta_file),truth=meta['truth_hash']);atomic_json(hashes_file,hashes)
                            jobs[key]='complete';atomic_json(jobs_file,jobs)
                            if any(ev['status']!='ok' for ev in meta['fit_events']):raise RuntimeError(f'unresolved online fitting failure in {key}; saved event and trace')
                        except Exception as exc:
                            if jobs[key]!='complete':jobs[key]='failed';atomic_json(jobs_file,jobs)
                            with (run/'failure_log.jsonl').open('a') as f:f.write(json.dumps(dict(job=key,error=repr(exc)))+'\n')
                            raise
                print(f'{c["phase"]}: {k} replicate {r+1}/{c["replicates"]}; {sum(v=="complete" for v in jobs.values())}/{len(jobs)} episodes',flush=True)
        if c['phase']=='final':
            dm=run/'distortion_manifest.json'
            if dm.exists():
                if file_sha(run/'distortion.csv')!=json.loads(dm.read_text())['file_hash']:raise ValueError('corrupt distortion diagnostic')
            else:distortion(run,c)
    ts=time.perf_counter();ne,nr=aggregate(run,c);aggregation_seconds=time.perf_counter()-ts
    atomic_json(run/'timing.json',dict(last_run_seconds=time.perf_counter()-t0,episode_rows=ne,round_rows=nr,aggregation_seconds=aggregation_seconds))
    return dict(run=str(run),complete=sum(x=='complete' for x in jobs.values()),total=len(jobs),seconds=time.perf_counter()-t0)
