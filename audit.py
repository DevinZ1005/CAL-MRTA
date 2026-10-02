"""Independent saved-artifact checks and deterministic regeneration."""
import csv,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .artifacts import verify,file_sha,atomic_json,verify_aggregation
from .experiment import POLICIES,job_id,job_paths,check_job,model_paths,episode_features,play_episode,pair_bins
from .simulator import episode
from .predictor import predict
from .calibration import transform
from .allocation import solve,selected_expected,utility_matrix
from .report import write_report


def require(condition,message):
    if not condition:raise ValueError(message)

def check_online(z,events,source):
    for pi,name in enumerate(POLICIES):
        if not name.startswith('rolling_'):continue
        own=[e for e in events if e['policy']==name];event_index=0;count=0;last=0;params=tuple(source);records=[]
        for t in range(160):
            np.testing.assert_array_equal(z['parameters'][pi,t],params)
            for i in range(5):
                if z['selected'][pi,t,i]>=0:
                    records.append((t,i,float(z['selected_raw'][pi,t,i]),int(z['outcome'][pi,t,i])));count+=1
            eligible=count>=50 and (last==0 or count-last>=25)
            if eligible:
                require(event_index<len(own),'missing refit event');e=own[event_index];event_index+=1
                require(e['outcomes']==count and e['newest_round']==t,'refit schedule')
                buf=records[-250:];raw=np.array([x[2] for x in buf]);labels=np.array([x[3] for x in buf])
                require(e['size']==len(buf) and e['oldest_round']==buf[0][0],'FIFO buffer')
                require(e['input_hash']==hashlib.sha256(raw.tobytes()+labels.tobytes()).hexdigest(),'feedback input hash')
                params=(e['a'],e['b']);last=count
        require(event_index==len(own),'unexpected refit event')

def audit(path):
    run=Path(path);t0=time.perf_counter();m,c=verify(run)
    jobs=json.loads((run/'jobs.json').read_text());hashes=json.loads((run/'job_hashes.json').read_text())
    expected={job_id(k,r,cond,e) for k in c['kernels'] for r in range(c['replicates']) for cond in c['conditions'] for e in range(c['episodes'])}
    errors=[];audited=replays=fit_failures=0;counts={s:sum(v==s for v in jobs.values()) for s in ('pending','running','complete','failed')}
    if set(jobs)!=expected:errors.append('job matrix mismatch')
    if any(v!='complete' for v in jobs.values()):errors.append('incomplete run')
    sources={};source_manifest=json.loads((run/'source_data_manifest.json').read_text())
    with threadpool_limits(limits=1):
        for block in source_manifest['blocks']:
            meta=block['metadata'];k,r=meta['kernel'],meta['replicate'];model_file,meta_file=model_paths(run,k,r)
            try:
                require(file_sha(meta_file)==block['metadata_hash'],'source metadata hash')
                require(file_sha(model_file)==meta['model_hash'],'source model hash')
                feature_rows=set();split_seeds=[]
                for split,info in meta['splits'].items():
                    path=run/'data'/f'{k}_{r:02d}_{split}.npz';require(file_sha(path)==info['file_hash'],'source split file hash')
                    with np.load(path) as data:
                        X,y=data['X'],data['y'];require(X.shape==(c['source_'+split],31) and y.shape==(len(X),),'source split shape')
                        require(hashlib.sha256(X.tobytes()).hexdigest()==info['feature_hash'],'source feature hash')
                        require(hashlib.sha256(y.tobytes()).hexdigest()==info['label_hash'],'source label hash')
                        values=set(map(bytes,X));require(len(values)==len(X),'duplicate source rows')
                        require(not values.intersection(feature_rows),'source split overlap');feature_rows.update(values)
                    split_seeds.extend(tuple(s) for s in info['seeds'].values())
                require(len(split_seeds)==len(set(split_seeds)),'overlapping source seeds')
                sources[(k,r)]=(pickle.loads(model_file.read_bytes()),tuple(meta['source_params']))
            except Exception as exc:errors.append(f'source {k}/{r}: {exc}')
        if len(sources)!=len(c['kernels'])*c['replicates']:errors.append('missing source blocks')
        paired={}
        for key,state in jobs.items():
            if state!='complete':continue
            try:
                check_job(run,key,hashes);trace_path,meta_path=job_paths(run,key);meta=json.loads(meta_path.read_text());k,r,cond,ep=(meta[x] for x in ('kernel','replicate','condition','episode'))
                model,source=sources[(k,r)];profiles,public,truth=episode(meta['phase'],k,r,ep,cond)
                with np.load(trace_path) as archive:z={name:archive[name] for name in archive.files}
                sel=z['selected'];att=z['attempts'];out=z['outcome'];ex=z['expected'];reg=z['regret']
                require(sel.shape==(5,160,5),'trace selection shape')
                require(np.all((sel==-1)|((sel>=0)&(sel<8))),'selection index')
                np.testing.assert_array_equal((sel>=0).sum(axis=2),att)
                require(np.all(out[sel==-1]==-1),'abstention outcome')
                require(np.all(np.isnan(z['selected_raw'][sel==-1])),'abstention raw score')
                require(np.all(np.isfinite(z['selected_raw'][sel>=0])),'selected raw score')
                require(np.min(reg)>=0 and np.max(ex-ex[4:5])<=1e-9,'oracle bound')
                np.testing.assert_allclose(z['pair_realized'].sum(axis=-1),z['realized'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(z['pair_expected'].sum(axis=-1),ex,rtol=0,atol=1e-12)
                bs=z['bin_stats'];np.testing.assert_allclose(bs[0,:,:,:,:,0].sum(axis=(-1,-2)),40)
                np.testing.assert_allclose(bs[1,:,:,:,:,0].sum(axis=(-1,-2)),att)
                check_online(z,meta['fit_events'],source)
                for t,(batch,tele,dist,cost) in enumerate(public):
                    tr=truth[t];y=tr.uniforms<tr.probabilities;oracle=solve(tr.probabilities,batch.reward,batch.penalty,cost);opt=selected_expected(oracle,tr.probabilities,batch.reward,batch.penalty,cost)
                    for pi,name in enumerate(POLICIES):
                        used=sel[pi,t][sel[pi,t]>=0];require(len(set(used))==len(used),'task uniqueness')
                        ids=np.flatnonzero(sel[pi,t]>=0);js=sel[pi,t,ids]
                        np.testing.assert_array_equal(out[pi,t,ids],y[ids,js])
                        er=tr.probabilities[ids,js]*batch.reward[js]-(1-tr.probabilities[ids,js])*batch.penalty[js]-cost[ids,js]
                        rr=np.where(y[ids,js],batch.reward[js],-batch.penalty[js])-cost[ids,js]
                        np.testing.assert_allclose(er,z['pair_expected'][pi,t,ids],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(rr,z['pair_realized'][pi,t,ids],rtol=0,atol=1e-12)
                        require(abs(opt-ex[pi,t]-reg[pi,t])<1e-9,'expected regret')
                        q=tr.probabilities[ids,js] if name=='oracle' else transform(z['selected_raw'][pi,t,ids],z['parameters'][pi,t])
                        np.testing.assert_allclose(q,z['selected_q'][pi,t,ids],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(pair_bins(q,y[ids,js],batch.types[js],tr.probabilities[ids,js]),bs[1,pi,t],rtol=0,atol=1e-10)
                if ep==0:
                    with np.load(run/'truth'/(key+'.npz')) as tr:
                        np.testing.assert_array_equal(tr['p_true'],np.stack([x.probabilities for x in truth]));np.testing.assert_array_equal(tr['uniforms'],np.stack([x.uniforms for x in truth]))
                        context=(k,r);baseline=paired.get(context)
                        if baseline is None:paired[context]=(tr['uniforms'].copy(),z['p_raw'][:40].copy(),tr['p_true'][:40].copy(),z['task_types'].copy())
                        else:
                            for a,b in zip(baseline,(tr['uniforms'],z['p_raw'][:40],tr['p_true'][:40],z['task_types'])):np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(predict(model,episode_features(profiles,public)),z['p_raw'])
                    for pi in range(5):
                        for t in range(160):
                            batch=public[t][0];tr=truth[t]
                            np.testing.assert_allclose(pair_bins(z['q_all'][pi,t],tr.uniforms<tr.probabilities,np.broadcast_to(batch.types,(5,8)),tr.probabilities),bs[0,pi,t],rtol=0,atol=1e-10)
                    if r==0:
                        replay,_,_=play_episode(profiles,public,truth,z['p_raw'],source,policy_order=tuple(reversed(POLICIES[:4])),collect=False)
                        for name in ('selected','selected_q','outcome','pair_realized','pair_expected','parameters'):np.testing.assert_array_equal(replay[name],z[name])
                        replays+=1
                fit_failures+=sum(e['status']!='ok' for e in meta['fit_events']);audited+=1
            except Exception as exc:errors.append(f'{key}: {exc}')
    if c['phase']=='final':
        try:
            dm=json.loads((run/'distortion_manifest.json').read_text());require(file_sha(run/'distortion.csv')==dm['file_hash'],'distortion hash')
            with (run/'distortion.csv').open() as f:dr=list(csv.DictReader(f))
            require(len(dr)==10000,'distortion workload');seen=set()
            for row in dr:
                key=(row['kernel'],int(row['replicate']),int(row['batch']),float(row['bias']));require(key not in seen,'duplicate distortion row');seen.add(key)
                require(float(row['regret'])>=0,'negative distortion regret')
                if float(row['bias'])==0:require(float(row['regret'])<1e-9 and int(row['disagreement'])==0,'zero distortion oracle agreement')
        except Exception as exc:errors.append(f'distortion: {exc}')
    try:verify_aggregation(run)
    except Exception as exc:errors.append(str(exc))
    result=dict(audited_jobs=audited,job_counts=counts,fit_failures=fit_failures,errors=errors,valid=not errors and fit_failures==0,replayed_episodes=replays,seconds=time.perf_counter()-t0,checks=['source split hashes and overlap','source/dependency/config/schema snapshot','committed artifact hashes','oracle expected bound','outcomes regenerated from U','utility arithmetic','selective raw-score feedback hashes','round-end refit schedule and FIFO','pre-decision parameters','paired exogenous contexts','candidate and attempted bin sums','first-episode predictor replay','reversed policy order with metric collection disabled'])
    atomic_json(run/'audit.json',result);write_report(run);return result
