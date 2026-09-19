"""Baseline freezing and complete reproducible runs."""
import itertools
import json
import time
import numpy as np
import torch
from .agent import Population, lifetime
from .genome import encode
from .evolution import evolve
from .evaluation import evaluate
from .utils import (CONDITIONS, canonical_hash, paired_data, provenance,
                    save_frame, seed_for, setup, source_files)

TUNING_PROTOCOL = 'independent_screen_v1'


def tune(config, output, fresh=False):
    setup()
    output.mkdir(parents=True,exist_ok=True)
    target=output/'baseline.json'
    if target.exists() and not fresh:
        raise FileExistsError(f'{target} exists; use --fresh to explicitly replace it')
    meta=provenance(config,output)
    candidates=list(itertools.product([.2,.5,.8],repeat=3))
    records=[]
    def screen(indices,split):
        scores=[]
        n=config['population_size']
        for seed in range(1000,1005):
            genomes=np.concatenate([np.tile(encode(candidates[i]),(n,1)) for i in indices])
            pop=Population(genomes,seed_for(seed,'tune_birth'),config)
            with torch.no_grad():
                base=pop.weights[:n].clone()
                for j in range(len(indices)):
                    pop.weights[j*n:(j+1)*n].copy_(base)
            train=paired_data(seed,'tune_train',config,config['train'],len(indices))
            lifetime(pop,train)
            # Preserve learning/validation streams; give training-distribution
            # screening a separate trajectory stream from lifetime learning.
            evaluation_namespace = 'tune_screen' if split == 'train' else 'tune_validation'
            data=paired_data(seed,evaluation_namespace,config,config[split],len(indices),config['evaluation_episodes'])
            acc,_=lifetime(pop,data,learn=False)
            means=acc.mean(0).reshape(len(indices),n).mean(1)
            scores.append(means)
            for index,score in zip(indices,means):
                records.append(dict(seed=seed,split=split,candidate=index,alpha=candidates[index][0],
                                    beta=candidates[index][1],gamma=candidates[index][2],fitness=float(score),
                                    learning_seed=seed_for(seed,'tune_train'),
                                    evaluation_seed=seed_for(seed,evaluation_namespace)))
        return np.mean(scores,axis=0)
    train_scores=screen(list(range(len(candidates))),'train')
    shortlist=np.argsort(-train_scores,kind='stable')[:9]
    validation_scores=screen(shortlist,'validation')
    winner=int(shortlist[np.argmax(validation_scores)])
    result=dict(parameters=list(candidates[winner]),candidate=winner,seeds=list(range(1000,1005)),
                tuning_protocol=TUNING_PROTOCOL,
                candidates=len(candidates),shortlisted=9,training_score=float(train_scores[winner]),
                validation_score=float(max(validation_scores)),**meta)
    result['freeze_hash']=canonical_hash(result)
    target.write_text(json.dumps(result,indent=2))
    save_frame(records,output/'tuning.parquet',meta)
    return result


def load_baseline(config, output):
    path=output/'baseline.json'
    if not path.exists():
        raise FileNotFoundError('Tune and freeze the baseline before running final experiments')
    data=json.loads(path.read_text())
    frozen_hash=data.pop('freeze_hash')
    if canonical_hash(data)!=frozen_hash or data['config_hash']!=canonical_hash(config):
        raise ValueError('Baseline hash/config mismatch; retune explicitly in a separate output directory')
    if data.get('tuning_protocol') != TUNING_PROTOCOL:
        raise ValueError('Baseline predates independent screening; retune before running experiments')
    data['freeze_hash']=frozen_hash
    return data


def run(config,seed,condition,output,fresh=False):
    setup()
    if condition not in CONDITIONS+['all','specialist_private','specialist_social']:
        raise ValueError(f'Unknown condition: {condition}')
    frozen=load_baseline(config,output)
    run_dir=output/f'seed_{seed:03d}_{condition}'
    if run_dir.exists() and any(run_dir.iterdir()) and not fresh:
        raise FileExistsError(f'{run_dir} exists; use --fresh or choose another output')
    run_dir.mkdir(parents=True,exist_ok=True)
    # Completion marker is written last; incomplete runs are never silently analyzed.
    (run_dir/'complete.json').unlink(missing_ok=True)
    meta=provenance(config,output)
    meta['baseline_hash']=frozen['freeze_hash']
    start=time.perf_counter()
    champions,genomes,curves,lineage=evolve(config,seed,frozen['parameters'],None if condition=='all' else [condition])
    save_frame(genomes,run_dir/'genomes.parquet',meta)
    save_frame(curves,run_dir/'evolution_episodes.parquet',meta)
    if lineage:
        save_frame(lineage,run_dir/'lineage.parquet',meta)
    if any(not k.startswith('specialist') for k in champions):
        episodes,summary,diagnostics,transfer=evaluate(config,seed,champions)
        save_frame(episodes,run_dir/'test_episodes.parquet',meta)
        save_frame(summary,run_dir/'summary.parquet',meta)
        save_frame(diagnostics,run_dir/'diagnostics.parquet',meta)
        save_frame(transfer,run_dir/'transfer_genomes.parquet',meta)
    result=dict(seed=seed,condition=condition,champions=champions,config=config,
                elapsed_seconds=time.perf_counter()-start,**meta)
    if canonical_hash(source_files()) != meta['source_hash']:
        raise RuntimeError('Source changed during execution; run is incomplete and must be rerun')
    (run_dir/'complete.json').write_text(json.dumps(result,indent=2))
    return result
