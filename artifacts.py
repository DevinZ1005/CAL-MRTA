"""Atomic local artifacts with source, configuration, schema and dependency checks."""
import csv,gzip,hashlib,json,os,pickle,platform,shutil,tempfile,sys
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from importlib.metadata import version
from .seeds import PHASE,KERNEL,STREAM
from .features import SCHEMA
PACKAGE=Path(__file__).resolve().parents[2]

def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(data);f.flush();os.fsync(f.fileno());temp=f.name
    os.replace(temp,path)
def atomic_json(path,obj):atomic_bytes(path,json.dumps(obj,indent=2,sort_keys=True,allow_nan=False).encode())
def atomic_pickle(path,obj):atomic_bytes(path,pickle.dumps(obj,protocol=pickle.HIGHEST_PROTOCOL))
def versions():
    packages=[line.split('==')[0] for line in (PACKAGE/'requirements-lock.txt').read_text().splitlines() if '==' in line]
    return dict(python=platform.python_version(),packages={p:version(p) for p in packages})
def source_files():
    return sorted([*PACKAGE.glob('src/cal_mrta/*.py'),*PACKAGE.glob('configs/*.json'),*PACKAGE.glob('tests/*.py'),*PACKAGE.glob('*.md'),PACKAGE/'requirements-lock.txt',PACKAGE/'pyproject.toml'])
def inventory():return {str(p.relative_to(PACKAGE)):file_sha(p) for p in source_files()}
def source_hash():return sha(canonical(inventory()))
def initialize(run,c,freeze=False):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    manifest_file=run/'manifest.json'
    current=dict(config_hash=sha(canonical(c)),source_hash=source_hash(),schema_hash=sha(canonical(SCHEMA)),versions=versions())
    if manifest_file.exists():
        saved,_=verify(run)
        if saved['config_hash']!=current['config_hash']:raise ValueError('resume drift: config_hash')
        if freeze and not saved.get('frozen'):raise ValueError('run already initialized without freeze')
        return saved
    snapshot=run/'source_snapshot'
    for p in source_files():
        target=snapshot/p.relative_to(PACKAGE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    manifest=dict(current,phase=c['phase'],frozen=bool(freeze),seed_map=dict(phases=PHASE,kernels=KERNEL,streams=STREAM),seed_tuple=['master','phase','kernel','replicate','stream','episode'],platform=platform.platform(),interpreter=sys.executable,created_utc=datetime.now(timezone.utc).isoformat(),source_inventory=inventory())
    atomic_json(run/'resolved_config.json',c);atomic_json(manifest_file,manifest)
    for name in ('data','models','traces','truth','figures'):(run/name).mkdir(exist_ok=True)
    atomic_json(run/'job_hashes.json',{})
    if (run.parent/'verification.txt').exists():atomic_bytes(run/'verification.txt',(run.parent/'verification.txt').read_bytes())
    return manifest

def verify(run):
    from .config import validate
    run=Path(run);m=json.loads((run/'manifest.json').read_text());c=validate(json.loads((run/'resolved_config.json').read_text()))
    for k,v in dict(config_hash=sha(canonical(c)),source_hash=source_hash(),schema_hash=sha(canonical(SCHEMA)),versions=versions()).items():
        if m[k]!=v:raise ValueError(f'run drift: {k}')
    for name,digest in m['source_inventory'].items():
        if file_sha(run/'source_snapshot'/name)!=digest:raise ValueError(f'corrupt source snapshot: {name}')
    return m,c

@contextmanager
def atomic_csv(path,fieldnames):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent);os.close(fd)
    opener=gzip.open if str(path).endswith('.gz') else open
    try:
        with opener(temp,'wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='raise');writer.writeheader();yield writer
        with open(temp,'rb') as f:os.fsync(f.fileno())
        os.replace(temp,path)
    except BaseException:
        Path(temp).unlink(missing_ok=True);raise

def write_csv(path,rows,fieldnames):
    with atomic_csv(path,fieldnames) as writer:writer.writerows(rows)


def verify_aggregation(run):
    run=Path(run)
    expected=json.loads((run/'aggregation_manifest.json').read_text())
    required={'episode_metrics.csv','round_metrics.csv.gz','fit_events.csv.gz','calibration_bins.csv'}
    if set(expected)!=required:raise ValueError('incomplete aggregation manifest')
    for name,digest in expected.items():
        if file_sha(run/name)!=digest:raise ValueError(f'corrupt aggregated artifact: {name}')
