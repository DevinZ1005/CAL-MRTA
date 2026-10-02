"""Strict, fully resolved version-1 protocol. No parameter search is permitted."""
import json
import math
from pathlib import Path
from .predictor import PARAMS
from .simulator import CONDITIONS, ARCHETYPES, TEMPLATES, DIRECTIONS
from .calibration import BOUNDS
from .features import SCHEMA
from .seeds import MASTER, PHASE, KERNEL, STREAM

BASE = dict(
    master_seed=MASTER, rounds=160, shift_round=40, robots=5, tasks=8,
    source_train=40000, source_cal=10000, source_iid=10000, replicates=10,
    episodes=10, kernels=["A", "B"], conditions=list(CONDITIONS),
    predictor={**PARAMS, "random_state": "source-training-seed"},
    source_lambda=.001, online_lambda=.01, online_minimum=50,
    online_interval=25, online_capacity=250, a_bounds=list(BOUNDS[0]),
    b_bounds=list(BOUNDS[1]), bootstrap_resamples=20000,
    distortion_batches=100, distortion_biases=[-1, -.5, 0, .5, 1],
    blas_threads=1, workers=1,
    simulator=dict(
        capability_order=["mobility", "sensing", "payload", "manipulation", "communication"],
        archetype_names=["Scout", "Inspector", "Carrier", "Manipulator", "Generalist"],
        archetypes=ARCHETYPES.tolist(), nominal_noise_sd=.03, nominal_clip=[.02,.98],
        sensitivity_uniform=[.85,1.15], reliability_normal=[0,.15],
        battery_uniform=[.6,1.0], position_uniform=[0,1],
        task_names=["Reconnaissance","Inspection","Transport","Manipulation","Mixed"],
        task_type_probabilities=[.2]*5, templates=TEMPLATES.tolist(), requirement_noise_sd=.05,
        requirement_clip=[.02,.98], difficulty_uniform=[0,1], reward_uniform=[2,5],
        failure_penalty_uniform=[1,4], distance_normalizer="sqrt(2)",
        execution_cost=[.15,.60,.10], directions={k:v.tolist() for k,v in DIRECTIONS.items()},
        baseline_direction="mixed", baseline_severity=.05, changed_severities=[.35,.70],
        battery_factor=[.75,.25], battery_capability_indices=[0,2,3],
        telemetry_noise_sd=[.03,.05,.05,.05], telemetry_clip=[0,1],
        model_A=dict(intercept=1.2,margin=3,deficit=-8,distance=-.8,battery=.8,
                     battery_center=.5,difficulty=-.5,weight_offset=.05,deficit_power=2),
        model_B=dict(intercept=1.5,bottleneck=5,distance=-.6,difficulty=-.4,
                     critical_count=2,tie_break="lower capability index"),
        probability_clip=[.02,.98], outcome_rule="independent uniform < p_true",
        source_direction_probabilities=[.25]*4, source_severity_uniform=[0,.15],
    ),
    optimizer=dict(method="L-BFGS-B",maxiter=100,ftol=1e-10,gtol=1e-7,
                   analytic_gradient=True,source_reference=[1,0],online_reference="source fit",
                   initialization="reference",raw_clip=[1e-6,1-1e-6]),
    assignment=dict(policies=["raw","source_platt","rolling_offset","rolling_platt","oracle"],
                    dummy_columns=5,dummy_utility=0,nonpositive_mask=-1e6,tie_epsilon=0),
    reporting=dict(primary_kernel="A",primary_condition="mixed_070",
                   primary_contrast=["rolling_platt","source_platt"],primary_rounds=[40,160],
                   secondary_early_rounds=[40,80],secondary_late_rounds=[120,160],
                   reliability_before=[0,40],reliability_after=[40,160],bins=10,
                   bin_edges=[i/10 for i in range(11)],bootstrap_interval=[.025,.975],
                   independent_unit="outer replicate",bootstrap_method="percentile",
                   tolerance=1e-9,feature_schema=SCHEMA),
    seed_protocol=dict(tuple=["master_seed","phase_id","kernel_id","replicate_id","stream_id","episode_id"],
                       phases=PHASE,kernels=KERNEL,streams=STREAM,condition_in_seed=False),
)

def for_phase(phase):
    # JSON round trip makes nested structures independent.
    c=json.loads(json.dumps(BASE)); c["phase"]=phase
    if phase=="smoke": c.update(source_train=2000,source_cal=1000,source_iid=1000,replicates=1,episodes=1,conditions=["no_change","mixed_070"])
    elif phase=="pilot": c.update(source_train=5000,source_cal=2000,source_iid=2000,replicates=2,episodes=3,conditions=["no_change","terrain_070","mixed_070"])
    elif phase!="final": raise ValueError("unknown phase")
    return c

def _compare(actual,expected,path="config"):
    if type(actual) is not type(expected): raise ValueError(f"{path}: invalid type")
    if isinstance(actual,dict):
        if set(actual)!=set(expected): raise ValueError(f"{path}: missing={set(expected)-set(actual)}, unknown={set(actual)-set(expected)}")
        for key in expected: _compare(actual[key],expected[key],f"{path}.{key}")
    elif isinstance(actual,list):
        if len(actual)!=len(expected): raise ValueError(f"{path}: length")
        for i,(a,b) in enumerate(zip(actual,expected)): _compare(a,b,f"{path}[{i}]")
    elif actual!=expected: raise ValueError(f"{path}: differs from fixed version-1 protocol")

def validate(c):
    if not isinstance(c,dict) or c.get("phase") not in ("smoke","pilot","final"): raise ValueError("phase")
    _compare(c,for_phase(c["phase"]))
    ids=list(STREAM.values())
    if len(ids)!=len(set(ids)): raise ValueError("duplicate stream IDs")
    if len(c["conditions"])!=len(set(c["conditions"])): raise ValueError("duplicate conditions")
    if len(SCHEMA)!=31 or len(set(SCHEMA))!=31: raise ValueError("feature schema")
    if not 0<c["online_interval"]<=c["online_minimum"]<=c["online_capacity"]: raise ValueError("buffer schedule")
    return c

def load(path):
    def reject_constant(value): raise ValueError(f"nonfinite JSON constant {value}")
    return validate(json.loads(Path(path).read_text(),parse_constant=reject_constant))
