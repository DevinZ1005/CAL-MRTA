"""Generate a factual report from committed run evidence."""
import json,re
from pathlib import Path
from .artifacts import atomic_bytes

def write_report(run):
    run=Path(run);c=json.loads((run/'resolved_config.json').read_text());m=json.loads((run/'manifest.json').read_text())
    def read(name):return json.loads((run/name).read_text()) if (run/name).exists() else None
    summary=read('summary.json');audit=read('audit.json');benchmark=read('benchmark.json')
    text=f'''# CAL-MRTA execution report — {c['phase']}

This is an AI-assisted software and simulation analysis artifact for student and mentor review. It is not a student-authored competition submission. The complete supplied protocol and assistance history are in the source snapshot.

## Implementation and protocol

Five heterogeneous robots choose among eight tasks in each of 160 externally reset dispatch batches. Conditions change at round 40. Positions, batteries and candidate tasks are exogenous. The classifier has exactly 31 public features. The normal policy interface receives a current DecisionInput and only that policy's selected-pair raw scores and outcomes. Source datasets are disjoint from final evaluation; all condition comparisons share exogenous seeds inside an outer replicate. Oracle truth is supplied only through the privileged engine/evaluator path.

Source models use the fixed HistGradientBoostingClassifier parameters; source and online Platt calibration use bounded, regularized L-BFGS-B. Online buffers contain up to 250 actual selected-pair outcomes, with the first refit after 50 and subsequent refits after at least 25 new outcomes at round end. All algorithms, simulator constants and reporting windows are in resolved_config.json. No search or favorable-effect stopping rule is implemented.

The environment is {m['versions']['python']} on {m['platform']}. Exact dependencies are in manifest.json and requirements-lock.txt. Source hash: `{m['source_hash']}`. Configuration hash: `{m['config_hash']}`. Final freeze status: {m['frozen']}.

## Verification and completed work

'''
    if (run/'verification.txt').exists():
        receipt=(run/'verification.txt').read_text();match=re.search(r'Ran (\d+) tests',receipt)
        if match:text+=f"Unit tests: {match.group(1)}; passing receipt: {'yes' if '\nOK' in receipt else 'no'}. Full output: verification.txt.\n\n"
    if audit:text+=f"Audit valid: **{audit['valid']}**. Audited episodes: {audit['audited_jobs']}. Unresolved fitting failures: {audit['fit_failures']}. Detailed checks and replay evidence: audit.json.\n\n"
    else:text+='Audit has not yet been run. Completion is not claimed.\n\n'
    if summary:
        p=summary['primary'];text+=f"Completed shared episodes: {summary['completed_jobs']}; predictor fits: {summary['predictor_fits']}; candidate-pair predictions: {summary['candidate_predictions']:,}; policy-round assignment solves: {summary['assignment_solves']:,}. Online refits: {summary['online_refits']:,}; online bound hits: {summary['online_bound_hits']:,}; source bound hits: {summary['source_bound_hits']}.\n\n"
        if 'distortion' in summary:text+=f"Separate distortion diagnostic: {summary['distortion']['shared_batches']:,} shared single-round batches and {summary['distortion']['assignment_solves']:,} assignment solves.\n\n"
        text+='## Prespecified primary contrast\n\n'
        if c['phase']!='final':text+='**Execution diagnostic only:** this smoke/pilot sample is not the final confirmatory experiment.\n\n'
        text+=f"Kernel A, mixed severity .70, rolling Platt minus source Platt, mean realized net utility per round over rounds 40–159: **{p['mean']:.6f}**. The 95% percentile bootstrap interval is **[{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}]**, width **{p['width']:.6f}**, using {p['n_blocks']} independent outer blocks and {p['bootstrap_resamples']:,} block resamples.\n\n"
        text+='Outer block differences: '+', '.join(f'{key}: {value:.6f}' for key,value in p['blocks'].items())+'.\n\n'
        text+='Every condition, method, and secondary contrast is retained in policy_estimates.csv and secondary_effects.csv. Probability metrics, sample counts and task-type breakdowns are in probability_metrics.csv and calibration_bins.csv. Brier and ECE are diagnostics; probability estimation MSE is explicitly distinct from calibration. The oracle gap includes hidden information and estimation limitations.\n\n'
        text+=f"Measured execution command wall time: {summary['run_timing']['last_run_seconds']:.2f} seconds. Source processing time: {summary['source_timing_seconds']['total']:.2f} seconds. Aggregation writing: {summary['run_timing']['aggregation_seconds']:.2f} seconds. Audit: {audit['seconds']:.2f} seconds.\n\n" if audit else ''
        times=summary['episode_timing_seconds'];text+=f"Total measured episode processing time: {times['total']:.2f} seconds; batched prediction: {times['prediction']:.2f}; allocation: {times['allocation']:.2f}; online fitting: {times['online_fitting']:.2f}; trace writing: {times['artifact']:.2f}. Analysis: {summary['analysis_seconds']:.2f} seconds. Source fitting and generation times are saved per source block. These component timers overlap only where explicitly nested (online fitting inside online observe and episode total).\n\n"
    else:text+='Analysis is pending. No study result is claimed.\n\n'
    if benchmark:text+='Pilot runtime projection is saved in benchmark.json, including component measurements and uncertainty range.\n\n'
    text+='''## Artifacts and replay

Each trace is a compressed array archive with explicit policy, round and robot axes. It contains assignments, saved raw and calibrated selected-pair probabilities, outcomes, per-robot and per-round utilities, parameters, task identifiers, and candidate/attempted sufficient statistics including task types. Full candidate raw and policy scores are saved for the first episode of each condition/replicate. The corresponding true probabilities and potential-outcome uniforms are in the separate truth directory. Regeneration identifiers use the manifest seed tuple and stable IDs.

Each work unit commits its trace and metadata hashes atomically. Resume checks code, configuration, feature schema, dependencies, source data, models, source snapshot and completed trace integrity. Incomplete work units can be regenerated with the same seeds, with prior failure/interruption status retained in failure_log.jsonl. Saved predictor pickle files are trusted local artifacts and must not be loaded from untrusted sources.

## Limitations

All success-model coefficients are modeling assumptions. This study has no navigation, coupled motion, deadlines, retries, battery depletion or multi-robot coalitions. It supports conclusions about this repeated centralized assignment simulation only. Online labels are selection dependent; attempt counts, abstention, task coverage and candidate-versus-attempted probability diagnostics must be considered together. Ten outer blocks can give imprecise intervals, and secondary comparisons do not establish multiple independent discoveries. The oracle's expected utility is an upper bound for each exogenous batch; its realized utility can be lower through chance.

Reliability panels use fixed bins and show counts, including sparse bins. They do not establish subgroup calibration guarantees. The distortion diagnostic imposes a known score bias and does not show that real robots or the learned predictor have that same error. A null or negative primary effect is a completed research result when the full protocol and audits are complete.
'''
    atomic_bytes(run/'execution_report.md',text.encode())
