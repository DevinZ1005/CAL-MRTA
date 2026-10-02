"""Prespecified paired outer-block analysis with complete matrix validation."""
import csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .artifacts import atomic_json,atomic_bytes,write_csv,verify,verify_aggregation
from .seeds import rng

METRICS=('realized','expected','regret','attempts','successes','failures','abstention','disagreement')
TRAJECTORY_METRICS=('realized','expected','regret','attempts','successes','failures','disagreement','positive_utility_fraction')

def primary_blocks(rows):
    by=defaultdict(dict)
    for r in rows:
        if r['kernel']!='A' or r['condition']!='mixed_070':continue
        key=(int(r['replicate']),int(r['episode']))
        if r['policy'] in by[key]:raise ValueError('duplicate episode method')
        by[key][r['policy']]=float(r['post_realized'])
    if not by:raise ValueError('missing primary rows')
    blocks=defaultdict(list)
    for (rep,ep),methods in by.items():
        if not {'rolling_platt','source_platt'}.issubset(methods):raise ValueError('missing method')
        blocks[rep].append(methods['rolling_platt']-methods['source_platt'])
    return {k:float(np.mean(v)) for k,v in sorted(blocks.items())}

def interval(values,indices):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):raise ValueError('nonfinite block values')
    lo,hi=np.quantile(values[indices].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()),ci_low=float(lo),ci_high=float(hi),width=float(hi-lo),n_blocks=len(values))

def analyze(run,partial=False):
    from .experiment import POLICIES,job_id,job_paths,check_job,npz_bytes
    from .report import write_report
    run=Path(run);started=time.perf_counter();manifest,c=verify(run)
    jobs=json.loads((run/'jobs.json').read_text());hashes=json.loads((run/'job_hashes.json').read_text())
    expected_jobs={job_id(k,r,cond,e) for k in c['kernels'] for r in range(c['replicates']) for cond in c['conditions'] for e in range(c['episodes'])}
    if set(jobs)!=expected_jobs:raise ValueError('inconsistent job matrix')
    if not partial and any(v!='complete' for v in jobs.values()):raise ValueError('incomplete run; use --partial only for a labeled diagnostic')
    if partial:
        partial_rows=[]
        for key,status in jobs.items():
            if status!='complete':continue
            check_job(run,key,hashes)
            _,metadata=job_paths(run,key)
            partial_rows.extend(json.loads(metadata.read_text())['episode_rows'])
        result=dict(partial=True,label='INCOMPLETE DIAGNOSTIC: no confirmatory block inference',completed_jobs=sum(v=='complete' for v in jobs.values()),total_jobs=len(jobs),primary=None)
        if partial_rows:write_csv(run/'partial_episode_metrics.csv',partial_rows,list(partial_rows[0]))
        atomic_json(run/'partial_summary.json',result)
        return result
    verify_aggregation(run)
    with (run/'episode_metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    found={};expected_rows=set()
    for k in c['kernels']:
        for r in range(c['replicates']):
            for cond in c['conditions']:
                for e in range(c['episodes']):
                    if jobs[job_id(k,r,cond,e)]=='complete':expected_rows.update((k,r,cond,e,p) for p in POLICIES)
    for row in rows:
        key=(row['kernel'],int(row['replicate']),row['condition'],int(row['episode']),row['policy'])
        if key in found:raise ValueError('duplicate episode key')
        if int(row['fitting_failures']):raise ValueError('unresolved fitting failure')
        found[key]=row
    if set(found)!=expected_rows:raise ValueError('missing or unexpected episode methods')
    indices={k:rng('bootstrap',k,0,'bootstrap').integers(0,c['replicates'],(c['bootstrap_resamples'],c['replicates'])) for k in c['kernels']}
    effects=[];absolute=[]
    for k in c['kernels']:
        for cond in c['conditions']:
            for period in ('post','first40','last40'):
                for metric in METRICS:
                    name=period+'_'+metric
                    blocks={p:np.array([np.mean([float(found[(k,r,cond,e,p)][name]) for e in range(c['episodes'])]) for r in range(c['replicates'])]) for p in POLICIES}
                    for p in POLICIES:
                        base=dict(kernel=k,condition=cond,period=period,metric=metric,policy=p)
                        absolute.append(dict(base,**interval(blocks[p],indices[k])))
                        difference=blocks[p]-blocks['source_platt']
                        effects.append(dict(base,reference='source_platt',**interval(difference,indices[k])))
    primary=primary_blocks(rows);vals=np.array(list(primary.values()));ip=interval(vals,indices['A'])
    primary_result=dict(mean=ip['mean'],blocks=primary,ci95=[ip['ci_low'],ip['ci_high']],width=ip['width'],n_blocks=len(vals),endpoint='mean realized net utility per round, rounds 40–159',kernel='A',condition='mixed_070',contrast='rolling_platt - source_platt',bootstrap_resamples=c['bootstrap_resamples'])
    write_csv(run/'secondary_effects.csv',effects,list(effects[0]));write_csv(run/'policy_estimates.csv',absolute,list(absolute[0]))
    # Trajectories first average episodes inside the outer replicate.
    shape=(len(c['kernels']),len(c['conditions']),5,c['replicates'],160,len(TRAJECTORY_METRICS))
    trajectory=np.zeros(shape);distribution=defaultdict(lambda:dict(p_true=[],p_raw=[]))
    total_fit_events=bound_hits=source_bound_hits=0;total_attempts=defaultdict(int);wall={}
    for key in jobs:
        check_job(run,key,hashes);trace_file,meta_file=job_paths(run,key);meta=json.loads(meta_file.read_text())
        ki=c['kernels'].index(meta['kernel']);ci=c['conditions'].index(meta['condition']);r=meta['replicate']
        with np.load(trace_file) as z:
            trajectory[ki,ci,:,r]+=np.stack([z[m] for m in TRAJECTORY_METRICS],axis=-1)/c['episodes']
            for pi,p in enumerate(POLICIES):total_attempts[p]+=int(z['attempts'][pi].sum())
            if meta['episode']==0:
                with np.load(run/'truth'/(key+'.npz')) as tr:distribution[(meta['kernel'],meta['condition'])]['p_true'].append(tr['p_true'].ravel())
                distribution[(meta['kernel'],meta['condition'])]['p_raw'].append(z['p_raw'].ravel())
        total_fit_events+=len(meta['fit_events']);bound_hits+=sum(e['bound_hit'] for e in meta['fit_events'])
        for name,value in meta['timing'].items():wall[name]=wall.get(name,0)+value
    atomic_bytes(run/'analysis_arrays.npz',npz_bytes(block_trajectories=trajectory,mean_trajectories=trajectory.mean(axis=3),kernels=np.array(c['kernels']),conditions=np.array(c['conditions']),policies=np.array(POLICIES),metrics=np.array(TRAJECTORY_METRICS)))
    dist_rows=[];warnings=[]
    for (k,cond),data in distribution.items():
        entry=dict(kernel=k,condition=cond,sample='first episode of each outer replicate; complete candidate pairs',quantile_levels=[0,.01,.1,.25,.5,.75,.9,.99,1])
        for name,arrays in data.items():
            values=np.concatenate(arrays);entry[name+'_count']=len(values);entry[name+'_quantiles']=np.quantile(values,entry['quantile_levels']).tolist()
            if np.ptp(values)<1e-10:warnings.append(f'{k}/{cond}: constant {name}')
        ki=c['kernels'].index(k);ci=c['conditions'].index(cond)
        entry['oracle_abstention']=float(1-trajectory[ki,ci,4,:,:,3].mean()/5)
        entry['positive_pair_fraction_by_policy']={p:float(trajectory[ki,ci,pi,:,:,-1].mean()) for pi,p in enumerate(POLICIES)}
        entry['attempts_by_policy']={p:float(trajectory[ki,ci,pi,:,:,3].mean()) for pi,p in enumerate(POLICIES)}
        dist_rows.append(entry)
        if any(v==0 for v in entry['attempts_by_policy'].values()):warnings.append(f'{k}/{cond}: a policy made zero attempts')
    atomic_json(run/'distribution_summary.json',dict(conditions=dist_rows,warnings=warnings))
    # Pool sums within each bin before computing ECE, preserving task-type panels.
    probability_groups=defaultdict(list)
    with (run/'calibration_bins.csv').open() as f:
        for b in csv.DictReader(f):probability_groups[tuple(b[x] for x in ('kernel','condition','policy','population','period','task_type'))].append(b)
    probability_rows=[]
    for key,bins in probability_groups.items():
        n=sum(int(b['count']) for b in bins);total=lambda field:sum(float(b[field]) for b in bins)
        row=dict(zip(('kernel','condition','policy','population','period','task_type'),key));row.update(count=n,brier=total('sum_brier')/n if n else None,logloss=total('sum_logloss')/n if n else None,ece=sum(abs(float(b['sum_q'])-float(b['sum_y'])) for b in bins)/n if n else None,probability_estimation_mse=total('sum_estimation_squared')/n if n else None)
        probability_rows.append(row)
    write_csv(run/'probability_metrics.csv',probability_rows,list(probability_rows[0]))
    source_manifest=json.loads((run/'source_data_manifest.json').read_text())
    source_bound_hits=sum(block['metadata']['source_bound_hit'] for block in source_manifest['blocks'])
    result=dict(partial=False,phase=c['phase'],primary=primary_result,completed_jobs=len(jobs),total_jobs=len(jobs),predictor_fits=len(source_manifest['blocks']),candidate_predictions=len(jobs)*160*40,assignment_solves=len(jobs)*160*5,online_refits=total_fit_events,online_bound_hits=bound_hits,source_bound_hits=source_bound_hits,unresolved_fitting_failures=0,attempted_pairs=total_attempts,episode_timing_seconds=wall,analysis_seconds=time.perf_counter()-started,warnings=warnings,interpretation='AI-assisted synthetic simulation analysis; calibration metrics and assignment utility are distinct endpoints.')
    if c['phase']=='final':
        dm=json.loads((run/'distortion_manifest.json').read_text())
        if dm['shared_batches']!=2000 or dm['assignment_solves']!=10000:raise ValueError('incomplete distortion diagnostic')
        result['distortion']=dm
    result['source_timing_seconds']={name:sum(block['metadata']['timing'][name] for block in source_manifest['blocks']) for name in source_manifest['blocks'][0]['metadata']['timing']}
    result['run_timing']=json.loads((run/'timing.json').read_text())
    if c['phase']=='final':
        with (run/'distortion.csv').open() as f:distortion_rows=list(csv.DictReader(f))
        ds=[]
        for k in c['kernels']:
            for bias in c['distortion_biases']:
                subset=[row for row in distortion_rows if row['kernel']==k and float(row['bias'])==bias]
                values={metric:float(np.mean([float(row[metric]) for row in subset])) for metric in ('expected','regret','attempts','disagreement')}
                ds.append(dict(kernel=k,bias=bias,batches=len(subset),unchanged_fraction=float(np.mean([int(row['disagreement'])==0 for row in subset])),**values))
        write_csv(run/'distortion_summary.csv',ds,list(ds[0]))
    atomic_json(run/'summary.json',result);write_report(run)
    return result
