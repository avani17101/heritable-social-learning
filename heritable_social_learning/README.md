# Heritable social-learning priors

A small CPU-first PyTorch experiment testing whether selection discovers a useful
inherited learning strategy. A three-logit genome controls the relative strength
of private and peer teaching signals and selectivity toward a noisy peer cue.
Every birth initializes a new 13×4 neural policy. **Only genomes are inherited.**

The complete experimental brief is in `../codex_instructions.md`; mathematical
choices, hypotheses and falsifiers were registered in [DESIGN.md](DESIGN.md).
The exact environment check passes before neural experiments: combining cued
signals reaches 80.208% Bayes accuracy when private-only and blind-copy both reach
70%. All observations remain compatible with every hidden state.

## Setup

From the workspace root:

```bash
source .venv/bin/activate
# For a new checkout, first create a Python 3.12+ venv and install:
python -m pip install -r heritable_social_learning/requirements.txt
cd heritable_social_learning
```

The workspace's `.venv` already contains the required packages. CPU is the default;
set `device: auto` to select CUDA if available. Paired comparisons require the same
device; bitwise cross-platform/CUDA reproducibility is not promised.

## Run

```bash
python scripts/analytical_check.py
python scripts/smoke_test.py
python scripts/tune_baseline.py --config configs/debug.yaml
python scripts/run_experiment.py --config configs/debug.yaml --all-seeds
python scripts/analyze_results.py --config configs/debug.yaml

# Full pre-registered experiment: 20 seeds, 32 agents, 50 generations.
python scripts/tune_baseline.py --config configs/final.yaml
python scripts/run_experiment.py --config configs/final.yaml --all-seeds
python scripts/check_provenance.py --config configs/final.yaml
python scripts/analyze_results.py --config configs/final.yaml
```

Existing outputs are refused by default to prevent accidental pooling or reuse.
`--fresh` explicitly replaces a matching run. Analysis demands all configured
seeds, the same source version/config, and a valid frozen baseline. To continue
an interrupted sweep, run each missing seed explicitly; do not overwrite completed
runs just to resume. Incomplete run folders have no `complete.json` marker and
require `--fresh` to rerun.

```bash
python scripts/run_experiment.py --condition evolved --seed 0 --config configs/debug.yaml
python scripts/run_transfer.py --condition all --seed 0 --config configs/final.yaml
```

`--condition all` batches the five main arms plus two specialization arms, then
evaluates random, ablated and scrambled controls. Individual condition runs are
useful for development; the preregistered report analyzes complete `all` runs.
Every experiment CLI accepts `--seed`, `--config`, `--condition`, and `--output`.
Tuning deliberately uses registered seeds 1000–1004 regardless of `--seed`;
its `--condition` is also ignored because it searches the fixed family. Smoke
tests ignore experiment parameters. The analytical gate is deterministic.

## Read the outputs

- [Final report](results/RESULTS.md), generated only after the full study.
- `results/final/baseline.json` and `tuning.parquet`: frozen train/validation search.
- `results/final/seed_*/genomes.parquet`: every genotype, decoded prior and fitness.
- `evolution_episodes.parquet`: each agent's training and held-out episode scores.
- `lineage.parquet`: nominal parent, actual genome donor and shuffled offspring slot.
- `test_episodes.parquet`: held-out newborn learning, transfer, shift and controls.
- `summary.parquet`, `diagnostics.parquet`, `transfer_genomes.parquet`: metrics,
  controlled conflict probes and the exact genomes used for transfer.
- `complete.json`: configuration, chosen genomes and timing; written last.
- `results/final/provenance/`: complete content-addressed source/config snapshots.
- Each source snapshot also has a clean Git bundle, independently verifiable
  without committing the user's working repository.
- `figures/final/`: eight required figures plus behavioral probes, each PNG/PDF.

Every numeric source row includes seed, source hash, config hash and Git metadata.
Task trajectories regenerate from stable seed namespaces and the saved source;
environment parameters and model hyperparameters are in each run's configuration.
Raw network weights are intentionally not persisted across generations.
`make check-provenance` checks all raw rows and immutable snapshots.

## What the experiment can and cannot establish

The primary comparison is evolved vs a fixed strategy selected on separate
training/validation data. The test grid is never used for tuning or selection.
Statistics resample independent evolutionary runs, not individual agents.
Nulls, failed controls and right-censored adaptation are reported.

**The requested lineage shuffle is not a causal inheritance knockout.** Permuting
genomes among independent, exchangeable newborns preserves the inherited genome
distribution. The code performs the requested permutation and records it, but H3
cannot prove lineage dependence in this design. This limitation is registered in
advance; a favorable noise fluctuation will not be presented as inheritance proof.

Peers are exogenous noisy demonstrators, and the neural policy has limited memory.
Its evidence-weighting update is a generalized teaching rule, not exact Bayesian
inference. Absolute alpha/beta scale is redundant after loss normalization.
This is neither cultural evolution nor inherited knowledge. Stage 2 is not built.

## Code map

`environment.py` generates noisy tasks and exact enumeration; `genome.py` contains
decoding/reproduction; `learning_rule.py` defines the normalized loss; `agent.py`
contains the tiny policy and lifetime learning; `evolution.py` performs births,
held-out selection and mutation; `evaluation.py` runs fresh newborn controls;
`experiment.py` freezes baselines and saves runs; `statistics.py` implements
seed-level bootstrap summaries using SciPy; `utils.py` handles configuration,
seed streams and provenance. There are no large MARL or neuroevolution frameworks.
