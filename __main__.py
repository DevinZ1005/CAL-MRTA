import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-cal-mrta')
import argparse,json,time
from pathlib import Path
from .config import load
from .artifacts import initialize,verify,atomic_json,file_sha,atomic_bytes

def benchmark(run,target):
    import numpy as np
    from .analysis import analyze
    from .report import write_report
    path=Path(run);_,c=verify(path);target=load(target)
    if c['phase']!='pilot':raise ValueError('benchmark requires pilot')
    audit=json.loads((path/'audit.json').read_text())
    if not audit['valid']:raise ValueError('pilot audit failed')
    jobs=json.loads((path/'jobs.json').read_text())
    if any(v!='complete' for v in jobs.values()):raise ValueError('pilot incomplete')
    times=[json.loads((path/'traces'/(key+'.json')).read_text())['timing'] for key in jobs]
    med={k:float(np.median([t[k] for t in times])) for k in times[0]}
    source=[json.loads(p.read_text())['timing'] for p in (path/'models').glob('*.json')]
    source_medians={k:float(np.median([t[k] for t in source])) for k in source[0]}
    analysis=analyze(path);nt=len(target['kernels'])*target['replicates']*len(target['conditions'])*target['episodes'];nf=len(target['kernels'])*target['replicates']
    ratios=dict(generation=(target['source_train']+target['source_cal']+target['source_iid'])/(c['source_train']+c['source_cal']+c['source_iid']),estimator_fit=target['source_train']/c['source_train'],source_calibration=target['source_cal']/c['source_cal'],iid=target['source_iid']/c['source_iid'])
    source_projection=sum(source_medians[k]*ratios[k]*nf for k in ratios)
    aggregation=json.loads((path/'timing.json').read_text())['aggregation_seconds']
    components=dict(source_generation_and_fitting=source_projection,batched_prediction=med['prediction']*nt,allocation=med['allocation']*nt,online_fitting=med['online_fitting']*nt,trace_artifact_writing=med['artifact']*nt,episode_total=med['total']*nt,aggregate_writing=aggregation*nt/len(jobs),analysis=analysis['analysis_seconds']*nt/len(jobs),audit=audit['seconds']*nt/len(jobs))
    total=source_projection+components['episode_total']+components['aggregate_writing']+components['analysis']+components['audit']
    report=dict(pilot_episode_median_seconds=med,pilot_source_block_median_seconds=source_medians,pilot_analysis_seconds=analysis['analysis_seconds'],pilot_audit_seconds=audit['seconds'],target_jobs=nt,target_predictor_fits=nf,source_size_ratios=ratios,projected_component_seconds=components,estimated_seconds=total,range_seconds=[.7*total,1.8*total],note='Serial measured projection, not a wall-time promise. Allocation/online/artifact component times are contained in episode_total. Source fitting is conservatively scaled by row count. Full audit and analysis are included; plotting time is not included.')
    atomic_json(path/'benchmark.json',report);write_report(path);return report

def freeze(config,output):
    from .experiment import job_id
    c=load(config)
    if c['phase']!='final':raise ValueError('freeze requires final config')
    run=Path(output);prerequisites={}
    for name in ('smoke','pilot'):
        other=run.parent/name;verify(other)
        audit=json.loads((other/'audit.json').read_text())
        if not audit['valid']:raise ValueError(f'{name} audit must pass before freeze')
        prerequisites[name+'_audit']=file_sha(other/'audit.json')
    prerequisites['pilot_benchmark']=file_sha(run.parent/'pilot'/'benchmark.json')
    verification=run.parent/'verification.txt'
    if not verification.exists() or '\nOK' not in verification.read_text():raise ValueError('save passing unittest output as runs/verification.txt before freeze')
    prerequisites['unit_tests']=file_sha(verification)
    m=initialize(run,c,freeze=True)
    if not (run/'jobs.json').exists():atomic_json(run/'jobs.json',{job_id(k,r,cond,e):'pending' for k in c['kernels'] for r in range(c['replicates']) for cond in c['conditions'] for e in range(c['episodes'])})
    atomic_json(run/'preflight_evidence.json',prerequisites);atomic_bytes(run/'verification.txt',verification.read_bytes());return m

def main():
    p=argparse.ArgumentParser(prog='python -m cal_mrta',description='Fixed version-1 CPU-only calibration and task allocation study')
    sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate-config',help='Validate the full fixed protocol');v.add_argument('--config',required=True)
    r=sub.add_parser('run',help='Execute or resume source fits and episode work units');r.add_argument('--config');r.add_argument('--output');r.add_argument('--frozen-run');r.add_argument('--resume',action='store_true')
    a=sub.add_parser('audit',help='Verify committed artifacts and deterministic replay');a.add_argument('--run',required=True)
    i=sub.add_parser('verify-run',help='Read-only archived artifact checks; requires output_checksums.json');i.add_argument('--run',required=True);i.add_argument('--checksum-index-sha256',help='Optional previously recorded SHA-256 of output_checksums.json')
    b=sub.add_parser('benchmark',help='Extrapolate measured pilot runtime');b.add_argument('--run',required=True);b.add_argument('--target-config',required=True)
    f=sub.add_parser('freeze',help='Freeze final source, configuration, dependencies and prerequisite evidence');f.add_argument('--config',required=True);f.add_argument('--output',required=True)
    n=sub.add_parser('analyze',help='Aggregate paired outer-block effects');n.add_argument('--run',required=True);n.add_argument('--partial',action='store_true')
    q=sub.add_parser('plot',help='Export prespecified PNG and PDF figures');q.add_argument('--run',required=True)
    args=p.parse_args()
    if args.command=='validate-config':load(args.config);out=dict(valid=True,config=args.config)
    elif args.command=='run':
        from .experiment import run
        if bool(args.frozen_run)==bool(args.config and args.output):p.error('use --frozen-run or --config with --output')
        out=run(args.config,args.output,args.frozen_run,args.resume)
    elif args.command=='audit':
        from .audit import audit
        out=audit(args.run)
    elif args.command=='verify-run':
        from .integrity import verify_run
        out=verify_run(args.run,args.checksum_index_sha256)
    elif args.command=='benchmark':out=benchmark(args.run,args.target_config)
    elif args.command=='freeze':out=freeze(args.config,args.output)
    elif args.command=='analyze':
        from .analysis import analyze
        out=analyze(args.run,args.partial)
    else:
        from .plots import plot
        out=plot(args.run)
    print(json.dumps(out,indent=2,allow_nan=False))
    if args.command=='audit' and not out['valid']:raise SystemExit(1)
if __name__=='__main__':main()
