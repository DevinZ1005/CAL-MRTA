"""Six prespecified figure groups; every exported plot is PNG and PDF."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .artifacts import verify,atomic_json
from .seeds import rng

POLICIES=('raw','source_platt','rolling_offset','rolling_platt','oracle')
LABELS=('Raw','Source','Offset','Platt','Oracle')
COLORS=('#666666','#0072B2','#E69F00','#009E73','#CC79A7')

def save(fig,base):
    fig.savefig(str(base)+'.png',dpi=150,bbox_inches='tight')
    fig.savefig(str(base)+'.pdf',bbox_inches='tight');plt.close(fig)

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def plot(run):
    run=Path(run);_,c=verify(run);summary=json.loads((run/'summary.json').read_text())
    if summary['partial']:raise ValueError('partial summaries cannot produce final figures')
    folder=run/'figures';folder.mkdir(exist_ok=True);generated=[]
    def emit(fig,name):save(fig,folder/name);generated.append(name)
    estimates=read_csv(run/'policy_estimates.csv')
    for k in c['kernels']:
        n=len(c['conditions']);cols=min(3,n);rows=(n+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,3.6*rows),squeeze=False,sharey=True,layout='constrained')
        for ax,cond in zip(axes.flat,c['conditions']):
            for pi,p in enumerate(POLICIES):
                row=next(r for r in estimates if r['kernel']==k and r['condition']==cond and r['policy']==p and r['period']=='post' and r['metric']=='realized')
                val=float(row['mean']);lo=float(row['ci_low']);hi=float(row['ci_high'])
                ax.errorbar(pi,val,yerr=[[max(0,val-lo)],[max(0,hi-val)]],fmt='o',capsize=3,color=COLORS[pi])
            ax.set(title=cond,xticks=range(5),xticklabels=LABELS,ylabel='Net utility / round (40–159)');ax.axhline(0,color='.85',lw=.8)
        for ax in list(axes.flat)[n:]:ax.set_visible(False)
        fig.suptitle(f'Kernel {k}: post-change utility and 95% outer-block bootstrap intervals')
        emit(fig,'01_utility_'+k)
    primary=summary['primary'];vals=list(primary['blocks'].values());labels=[f'Block {i}' for i in primary['blocks']]
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ax.scatter(vals,range(len(vals)),color=COLORS[3]);y=len(vals)+.5;mean=primary['mean'];lo,hi=primary['ci95']
    ax.errorbar(mean,y,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='D',capsize=5,color='black');ax.axvline(0,color='.6',ls='--');ax.set(yticks=list(range(len(vals)))+[y],yticklabels=labels+['Mean, 95% CI'],xlabel='Rolling Platt − source Platt: net utility / round',title='Primary contrast: kernel A, mixed_070, rounds 40–159');ax.invert_yaxis()
    emit(fig,'02_primary_blocks')
    with np.load(run/'analysis_arrays.npz') as z:trajectories=z['mean_trajectories'];metric_names=list(z['metrics'])
    for ki,k in enumerate(c['kernels']):
        for metric,title in (('realized','Realized net utility / round'),('regret','Expected regret / round')):
            n=len(c['conditions']);cols=min(3,n);rows=(n+cols-1)//cols
            fig,axes=plt.subplots(rows,cols,figsize=(5*cols,3.5*rows),squeeze=False,sharex=True,sharey=True,layout='constrained')
            for ci,(ax,cond) in enumerate(zip(axes.flat,c['conditions'])):
                for pi,p in enumerate(POLICIES):ax.plot(range(160),trajectories[ki,ci,pi,:,metric_names.index(metric)],color=COLORS[pi],label=LABELS[pi],lw=1)
                ax.axvline(40,color='black',ls='--',lw=.8);ax.set(title=cond,xlabel='Dispatch round',ylabel=title)
            for ax in list(axes.flat)[n:]:ax.set_visible(False)
            axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle(f'Kernel {k}: {title.lower()} (unsmoothed means)')
            emit(fig,f'03_{metric}_{k}')
    bins=read_csv(run/'calibration_bins.csv')
    def reliability(ax,count_ax,k,pop,period,task_type):
        ax.plot([0,1],[0,1],color='.6',ls='--',lw=.8)
        for pi,p in enumerate(POLICIES):
            data=sorted([b for b in bins if b['kernel']==k and b['condition']=='mixed_070' and b['policy']==p and b['population']==pop and b['period']==period and int(b['task_type'])==task_type],key=lambda b:int(b['bin']))
            counts=np.array([int(b['count']) for b in data]);centers=(np.arange(10)+.5)/10
            nonempty=[b for b in data if int(b['count'])>0]
            ax.scatter([float(b['sum_q'])/int(b['count']) for b in nonempty],[float(b['sum_y'])/int(b['count']) for b in nonempty],s=18,marker='o',color=COLORS[pi],label=f'{LABELS[pi]} (n={counts.sum():,})')
            count_ax.plot(centers,counts,color=COLORS[pi],marker='.',lw=.8,label=LABELS[pi])
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed success fraction')
        count_ax.set(xlim=(0,1),xlabel='Fixed probability bin center',ylabel='Pair count');count_ax.set_yscale('symlog',linthresh=1);count_ax.set_ylim(bottom=0);count_ax.grid(axis='y',alpha=.2)
    for k in c['kernels']:
        fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
        for i,pop in enumerate(('candidate','attempted')):
            for j,period in enumerate(('before','after')):
                ax,count_ax=axes[i,j*2:j*2+2];reliability(ax,count_ax,k,pop,period,-1)
                ax.set_title(f'{pop}: {period} ({"0–39" if period=="before" else "40–159"})');count_ax.set_title('Counts, including sparse/empty bins');ax.legend(fontsize=7)
        fig.suptitle(f'Kernel {k}, mixed_070: reliability and bin counts; candidate outcomes are privileged diagnostics')
        emit(fig,'04_reliability_'+k)
        fig,axes=plt.subplots(4,5,figsize=(22,14),layout='constrained')
        for ti,task_name in enumerate(c['simulator']['task_names']):
            for i,pop in enumerate(('candidate','attempted')):
                reliability(axes[i*2,ti],axes[i*2+1,ti],k,pop,'after',ti)
                axes[i*2,ti].set_title(f'{task_name}: {pop}, rounds 40–159');axes[i*2,ti].legend(fontsize=7)
        fig.suptitle(f'Kernel {k}, mixed_070: task-type diagnostics (no subgroup calibration guarantee)')
        emit(fig,'04_task_types_'+k)
    for ki,k in enumerate(c['kernels']):
        n=len(c['conditions']);cols=min(3,n);rows=(n+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,3.6*rows),squeeze=False,sharey=True,layout='constrained')
        for ci,(ax,cond) in enumerate(zip(axes.flat,c['conditions'])):
            successes=trajectories[ki,ci,:,40:,metric_names.index('successes')].mean(axis=1);failures=trajectories[ki,ci,:,40:,metric_names.index('failures')].mean(axis=1)
            ax.bar(range(5),successes,color='#009E73',label='Successes');ax.bar(range(5),failures,bottom=successes,color='#D55E00',label='Failures');ax.scatter(range(5),successes+failures,color='black',s=12,label='Attempts')
            ax.set(title=cond,xticks=range(5),xticklabels=LABELS,ylim=(0,5),ylabel='Tasks / round (40–159)')
        for ax in list(axes.flat)[n:]:ax.set_visible(False)
        axes[0,0].legend(fontsize=8);fig.suptitle(f'Kernel {k}: assignment attempts, successes and failures')
        emit(fig,'05_activity_'+k)
    if (run/'distortion.csv').exists():
        data=read_csv(run/'distortion.csv');fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ki,k in enumerate(c['kernels']):
            for ax,metric in zip(axes,('regret','attempts','disagreement')):
                means=[];lows=[];highs=[]
                for b in c['distortion_biases']:
                    values=np.array([np.mean([float(row[metric]) for row in data if row['kernel']==k and int(row['replicate'])==r and float(row['bias'])==b]) for r in range(c['replicates'])])
                    ix=rng('bootstrap',k,0,'bootstrap',episode=1).integers(0,len(values),(c['bootstrap_resamples'],len(values)))
                    lo,hi=np.quantile(values[ix].mean(axis=1),[.025,.975]);means.append(values.mean());lows.append(lo);highs.append(hi)
                ax.plot(c['distortion_biases'],means,marker='o',label='Kernel '+k);ax.fill_between(c['distortion_biases'],lows,highs,alpha=.15);ax.set(xlabel='Imposed logit bias (synthetic)',ylabel={'regret':'Expected regret / batch','attempts':'Attempts / batch','disagreement':'Robot assignment differences / batch'}[metric]);ax.axvline(0,color='.7',ls='--')
        axes[0].legend();fig.suptitle('Controlled distortion diagnostic; shaded bands are outer-block 95% intervals')
        emit(fig,'06_distortion')
    atomic_json(run/'figures_manifest.json',dict(groups=6 if (run/'distortion.csv').exists() else 5,figures=generated,formats=['png','pdf'],note='Smoke/pilot omit the final-only distortion diagnostic.'))
    return dict(figures=str(folder),count=len(generated))
