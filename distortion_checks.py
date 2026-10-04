"""Validate saved paired distortion batches before using their results."""
import csv
import json
import math
from pathlib import Path

from .artifacts import file_sha


def verify_distortion(run, config):
    """Return the manifest and rows, accepting both legacy and independent solves."""
    run = Path(run)
    manifest = json.loads((run / 'distortion_manifest.json').read_text())
    path = run / 'distortion.csv'
    if file_sha(path) != manifest['file_hash']:
        raise ValueError('corrupt distortion diagnostic')
    expected = {(k, r, batch, float(bias)) for k in config['kernels']
                for r in range(config['replicates'])
                for batch in range(config['distortion_batches'])
                for bias in config['distortion_biases']}
    batches = len(config['kernels']) * config['replicates'] * config['distortion_batches']
    method = manifest.get('zero_bias_method', 'reuse_oracle')
    if method not in ('reuse_oracle', 'independent_solve'):
        raise ValueError('unknown distortion zero-bias method')
    bias_solves = len(expected) - (batches if method == 'reuse_oracle' else 0)
    counts = dict(rows=len(expected), shared_batches=batches,
                  assignment_solves=batches + bias_solves, phase=5)
    if method == 'independent_solve':
        counts.update(oracle_solves=batches, bias_solves=bias_solves)
    if any(manifest.get(key) != value for key, value in counts.items()):
        raise ValueError('inconsistent distortion workload')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    optima = {}
    for row in rows:
        key = (row['kernel'], int(row['replicate']), int(row['batch']), float(row['bias']))
        if key not in expected or key in seen:
            raise ValueError('duplicate or unexpected distortion row')
        seen.add(key)
        value, regret = float(row['expected']), float(row['regret'])
        attempts, disagreement = int(row['attempts']), int(row['disagreement'])
        if not all(math.isfinite(x) for x in (value, regret)) or regret < 0:
            raise ValueError('invalid distortion utility or regret')
        if not 0 <= attempts <= config['robots'] or not 0 <= disagreement <= config['robots']:
            raise ValueError('invalid distortion activity count')
        optimum = value + regret
        if not math.isfinite(optimum) or not math.isclose(
                optimum, optima.setdefault(key[:3], optimum), rel_tol=0, abs_tol=1e-9):
            raise ValueError('inconsistent paired distortion optimum')
        # Tied optima can select different assignments after floating-point round trips.
        if key[3] == 0 and (regret > 1e-9 or (method == 'reuse_oracle' and disagreement != 0)):
            raise ValueError('zero distortion oracle agreement')
    if seen != expected:
        raise ValueError('incomplete distortion matrix')
    return manifest, rows
