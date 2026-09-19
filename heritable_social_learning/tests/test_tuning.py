"""Regression tests for the actual baseline-search sampling and freeze path."""
import numpy as np
import pytest
import json
from src import experiment
from src.utils import ROOT, canonical_hash, load_config, seed_for, setup


def test_tuning_evaluates_independent_trajectories_with_candidate_pairing(tmp_path, monkeypatch):
    setup()
    config = {**load_config(ROOT / 'configs/debug.yaml'), 'population_size': 2,
              'learning_episodes': 2, 'evaluation_episodes': 2, 'episode_length': 3}
    calls = []
    original = experiment.paired_data

    def record(seed, namespace, config, regimes, groups=1, episodes=None):
        data = original(seed, namespace, config, regimes, groups, episodes)
        calls.append((seed, namespace, data))
        # Every candidate sees identical tasks; independence is between phases.
        for value in data.values():
            if value.ndim == 3:
                for group in range(1, groups):
                    assert np.array_equal(value[..., :2], value[..., group*2:(group+1)*2])
        return data

    monkeypatch.setattr(experiment, 'paired_data', record)
    monkeypatch.setattr(experiment, 'provenance', lambda config, output: {})
    experiment.tune(config, tmp_path)
    assert len(calls) == 20  # learning/evaluation × five seeds × two screening stages
    for learning, evaluation in zip(calls[::2], calls[1::2]):
        learn_seed, learn_namespace, learn_data = learning
        eval_seed, eval_namespace, eval_data = evaluation
        assert seed_for(learn_seed, learn_namespace) != seed_for(eval_seed, eval_namespace)
        assert not all(np.array_equal(learn_data[key], eval_data[key])
                       for key in ['z', 'private', 'social', 'feedback'])


def test_legacy_baseline_cannot_be_reused_after_screening_fix(tmp_path):
    config = load_config(ROOT / 'configs/debug.yaml')
    baseline = {'parameters': [.2, .8, .2], 'config_hash': canonical_hash(config)}
    baseline['freeze_hash'] = canonical_hash(baseline)
    (tmp_path / 'baseline.json').write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match='predates independent screening'):
        experiment.load_baseline(config, tmp_path)
