"""Read-only archive checks; never replay experiments or load model pickles."""
import json
from pathlib import Path

from .artifacts import canonical, file_sha, inventory, sha, versions


def verify_run(run, checksum_index_sha256=None):
    """Check a complete archive against its saved indexes, independent of live code.

    This checks local record consistency, not scientific validity or authenticity.
    An optional externally recorded index digest anchors the checksum index itself.
    Execution/resume must continue to use artifacts.verify's strict drift checks.
    """
    run = Path(run).resolve()
    index_path = run / 'output_checksums.json'
    index_hash = file_sha(index_path)
    if checksum_index_sha256 is not None and index_hash != checksum_index_sha256:
        raise ValueError('corrupt output checksum index')
    checked = {}

    def check(name, digest):
        relative = Path(name)
        path = run / relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(run):
            raise ValueError(f'artifact path escapes run: {name}')
        if not path.is_file() or file_sha(path) != digest:
            raise ValueError(f'corrupt or missing artifact: {name}')
        checked[relative.as_posix()] = digest

    def read(name):
        return json.loads((run / name).read_text())

    outputs = read('output_checksums.json')
    required = {'manifest.json', 'resolved_config.json', 'jobs.json',
                'job_hashes.json', 'source_data_manifest.json'}
    if not required.issubset(outputs):
        raise ValueError('incomplete output checksum index')
    for name, digest in outputs.items():
        check(name, digest)
    manifest, config = read('manifest.json'), read('resolved_config.json')
    saved_inventory = manifest['source_inventory']
    if sha(canonical(saved_inventory)) != manifest['source_hash']:
        raise ValueError('corrupt source inventory')
    if sha(canonical(config)) != manifest['config_hash']:
        raise ValueError('corrupt saved configuration')
    if sha(canonical(config['reporting']['feature_schema'])) != manifest['schema_hash']:
        raise ValueError('corrupt saved feature schema')
    for name, digest in saved_inventory.items():
        # Validate before prefixing so absolute inventory paths cannot escape the snapshot.
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'invalid source inventory path: {name}')
        check('source_snapshot/' + name, digest)
    jobs, hashes = read('jobs.json'), read('job_hashes.json')
    expected = {f'{k}_{r:02d}_{condition}_{ep:02d}'
                for k in config['kernels'] for r in range(config['replicates'])
                for condition in config['conditions'] for ep in range(config['episodes'])}
    if set(jobs) != expected or set(hashes) != expected or set(jobs.values()) != {'complete'}:
        raise ValueError('incomplete archived job matrix')
    for key, item in hashes.items():
        check(f'traces/{key}.npz', item['trace'])
        check(f'traces/{key}.json', item['metadata'])
        if item.get('truth'):
            check(f'truth/{key}.npz', item['truth'])
        elif key.endswith('_00'):
            raise ValueError(f'missing first-episode truth hash: {key}')
    expected_sources = {(k, r) for k in config['kernels'] for r in range(config['replicates'])}
    seen = set()
    for block in read('source_data_manifest.json')['blocks']:
        meta = block['metadata']
        key = (meta['kernel'], meta['replicate'])
        if key not in expected_sources or key in seen:
            raise ValueError('duplicate or unexpected source block')
        seen.add(key)
        stem = f'{key[0]}_{key[1]:02d}'
        check(f'models/{stem}.json', block['metadata_hash'])
        if read(f'models/{stem}.json') != meta:
            raise ValueError('source metadata differs from manifest')
        check(f'models/{stem}.pkl', meta['model_hash'])
        if set(meta['splits']) != {'train', 'cal', 'iid'}:
            raise ValueError('incomplete source splits')
        for split, info in meta['splits'].items():
            check(f'data/{stem}_{split}.npz', info['file_hash'])
    if seen != expected_sources:
        raise ValueError('incomplete source block matrix')
    files = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
    unindexed = files - set(checked) - {'output_checksums.json'}
    if unindexed:
        raise ValueError(f'unindexed archive files: {sorted(unindexed)}')
    current = inventory()
    changed = sorted(name for name in set(saved_inventory) | set(current)
                     if saved_inventory.get(name) != current.get(name))
    return dict(valid=True, mode='read-only artifact integrity; no replay',
                verified_files=len(checked), checksum_index_sha256=index_hash,
                completed_jobs=len(jobs), source_blocks=len(seen),
                working_source_matches=not changed, working_source_changes=changed,
                environment_matches=versions() == manifest['versions'])
