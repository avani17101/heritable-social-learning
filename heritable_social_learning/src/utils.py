"""Configuration, independent seed namespaces, and content-addressed provenance."""
import argparse
import hashlib
import json
import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ['individual', 'fixed', 'best_fixed', 'evolved', 'shuffled']


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seed_for(seed, namespace, index=0):
    """Stable seed substreams, independent of Python's process-randomized hash."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{namespace}:{index}'.encode()).digest()[:4], 'little')


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    required = ['population_size','generations','learning_episodes','evaluation_episodes','test_episodes','episode_length']
    if any(not isinstance(config[k], int) or config[k] < 1 for k in required):
        raise ValueError('Population sizes and episode counts must be positive integers')
    if config['population_size'] < 2 or not 0 < config['elite_fraction'] <= 1:
        raise ValueError('Need at least two agents and a valid elite fraction')
    if not 0 < config['learning_rate'] or config['mutation_scale'] < 0:
        raise ValueError('Invalid learning rate or mutation scale')
    for split in ['train','validation','test']:
        if any(not .35 <= s <= .95 or not .25 < p < 1 for p,s in config[split]):
            raise ValueError('Main environments require private>.25 and social in [.35,.95]')
    if set(map(tuple,config['train'])) & set(map(tuple,config['test'])):
        raise ValueError('Training and test environments overlap')
    return config


def cli(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default=str(ROOT/'configs/debug.yaml'))
    parser.add_argument('--condition', default='all')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--fresh', action='store_true', help='Overwrite matching output; otherwise refuse existing runs')
    return parser


def source_files():
    paths = []
    for pattern in ['src/*.py','scripts/*.py','tests/*.py','configs/*.yaml','*.yaml','*.md','requirements*.txt','Makefile']:
        paths.extend(ROOT.glob(pattern))
    # Reports change after analysis and are not part of executable provenance.
    return {str(p.relative_to(ROOT)):p.read_text() for p in sorted(set(paths))
            if p.name not in ['claude_report_to_avani.md']}


def provenance(config, output):
    output.mkdir(parents=True, exist_ok=True)
    files = source_files()
    source_hash = canonical_hash(files)
    config_hash = canonical_hash(config)
    try:
        workspace_sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
        workspace_dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except subprocess.CalledProcessError:
        workspace_sha, workspace_dirty = None, True
    archive = output/'provenance'
    archive.mkdir(exist_ok=True)
    # A clean, isolated Git snapshot avoids committing the user's workspace.
    # Fixed metadata makes the snapshot commit reproducible from its contents.
    bundle = archive/f'{source_hash}.bundle'
    if not bundle.exists():
        with tempfile.TemporaryDirectory(prefix='social-learning-source-') as temporary:
            directory=Path(temporary)
            for name,content in files.items():
                dest=directory/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(content)
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=directory,stderr=subprocess.DEVNULL,
                    env={**os.environ,'GIT_AUTHOR_NAME':'Research source snapshot','GIT_AUTHOR_EMAIL':'snapshot@localhost',
                         'GIT_COMMITTER_NAME':'Research source snapshot','GIT_COMMITTER_EMAIL':'snapshot@localhost',
                         'GIT_AUTHOR_DATE':'2026-09-19T00:00:00+00:00','GIT_COMMITTER_DATE':'2026-09-19T00:00:00+00:00'},text=True).strip()
            git('init','-q')
            git('add','.')
            git('-c','commit.gpgsign=false','commit','-qm',f'Frozen experimental source {source_hash}')
            git('bundle','create',str(bundle.resolve()),'HEAD')
    sha = subprocess.check_output(['git','bundle','list-heads',str(bundle)],text=True).split()[0]
    manifest = dict(source_hash=source_hash,config_hash=config_hash,git_sha=sha,git_dirty=False,
                    workspace_git_sha=workspace_sha,workspace_git_dirty=workspace_dirty,
                    config=config, files=files, torch_version=torch.__version__, numpy_version=np.__version__)
    path = archive/f'{source_hash}_{config_hash}.json'
    if not path.exists():
        path.write_text(json.dumps(manifest,indent=2))
    return dict(source_hash=source_hash,config_hash=config_hash,git_sha=sha,git_dirty=False)


def save_frame(rows, path, metadata):
    frame = pd.DataFrame(rows)
    for k,v in metadata.items():
        frame[k] = v
    frame.to_parquet(path,index=False)


def paired_data(seed, namespace, config, regimes, groups=1, episodes=None):
    from .environment import sample
    data = sample(seed_for(seed,namespace),episodes or config['learning_episodes'],
                  config['episode_length'],config['population_size'],regimes)
    return tile_data(data,groups)


def tile_data(data, groups):
    return {k: np.concatenate([v]*groups,axis=-1) if v.ndim==3 else v.copy() for k,v in data.items()}
