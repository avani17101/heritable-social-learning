"""Fresh newborn generalization, transfer, shift and behavioral probes."""
import numpy as np
import torch
from .agent import Population, lifetime, probes
from .environment import sample
from .utils import seed_for


def adaptation_lag(accuracy, shift):
    """First trailing 3-episode post-shift mean >= .9 × last-5 pre-shift mean.

    Return NaN if right-censored, rather than dropping unsuccessful recovery.
    """
    target = .9*np.mean(accuracy[max(0,shift-5):shift])
    for end in range(shift+2,len(accuracy)):
        if np.mean(accuracy[end-2:end+1]) >= target:
            return float(end-shift+1)
    return float('nan')


def evaluate(config, seed, champions):
    n, steps, episodes = config['population_size'], config['episode_length'], config['test_episodes']
    names = [k for k in champions if not k.startswith('specialist')]
    if 'evolved' in champions:
        names += ['random','ablated','scrambled']
    genomes = {}
    for name in names:
        if name=='random':
            genomes[name] = np.random.default_rng(seed_for(seed,'transfer_random')).normal(size=(n,3))
        elif name=='scrambled':
            rng = np.random.default_rng(seed_for(seed,'scramble'))
            # Uniform over all six permutations, including identity; log realized genomes.
            genomes[name] = np.array([rng.permutation(champions['evolved']) for _ in range(n)])
        else:
            genomes[name] = np.tile(champions['evolved' if name=='ablated' else name],(n,1))
    regimes = [(f'test_{p:.2f}_{s:.2f}',[[p,s]]) for p,s in config['test']]
    regimes += [('shift',None),('no_social',[[.7,.25]]),('perfect_social',[[.7,.99]])]
    rows, summaries, diagnostics, transfer_genomes = [], [], [], []
    for name in names:
        for agent,g in enumerate(genomes[name]):
            transfer_genomes.append(dict(seed=seed,condition=name,agent=agent,g_alpha=g[0],g_beta=g[1],g_gamma=g[2]))
    # One vectorized population across regimes and conditions; independent network weights.
    combined = np.concatenate([genomes[name] for _ in regimes for name in names])
    population = Population(combined,seed_for(seed,'test_birth'),config)
    with torch.no_grad():
        base = population.weights[:n].clone()
        for block in range(len(regimes)*len(names)):
            population.weights[block*n:(block+1)*n].copy_(base)
            if names[block%len(names)] in ['individual','ablated']:
                population.parameters_decoded[block*n:(block+1)*n,1] = 0
    blocks=[]
    for regime_id, (label, rs) in enumerate(regimes):
        if label=='shift':
            first = sample(seed_for(seed,'test_tasks',regime_id),episodes//2,steps,n,[[.8,.6]])
            second = sample(seed_for(seed,'shift_tasks'),episodes-episodes//2,steps,n,[[.6,.8]])
            data = {k:np.concatenate([first[k],second[k]],axis=0) for k in first}
        else:
            data = sample(seed_for(seed,'test_tasks',regime_id),episodes,steps,n,rs)
        blocks.extend([data]*len(names))
    data = {k:np.concatenate([b[k] for b in blocks],axis=-1) if blocks[0][k].ndim==3 else blocks[0][k]
            for k in blocks[0]}
    # Birth score is pre-update, across the complete first episode's independent inputs.
    birth_population = Population(combined,seed_for(seed,'test_birth'),config)
    birth_population.load_state_dict(population.state_dict())
    birth,_ = lifetime(birth_population,{k:v[:1] for k,v in data.items()},learn=False)
    accuracy,reward = lifetime(population,data)
    for block,(label,rs) in enumerate((r for r in regimes for _ in names)):
        name = names[block%len(names)]
        sl = slice(block*n,(block+1)*n)
        a, rew = accuracy[:,sl], reward[:,sl]
        per_episode = a.mean(1)
        regret = steps*(1-a)
        for ep in range(episodes):
            for agent in range(n):
                rows.append(dict(seed=seed,condition=name,regime=label,episode=ep,agent=agent,
                                 accuracy=float(a[ep,agent]),reward=float(rew[ep,agent]),
                                 regret=float(regret[ep,agent]),expected_noisy_regret=float(.8*regret[ep,agent])))
        record=dict(seed=seed,condition=name,regime=label,regret=float(regret.sum(0).mean()),
                    birth_accuracy=float(birth[0,sl].mean()),final_accuracy=float(a[-min(10,episodes):].mean()),
                    adaptation_lag=adaptation_lag(per_episode,episodes//2) if label=='shift' else float('nan'))
        for fraction in [.1,.25,.5]:
            stop=max(1,int(episodes*fraction))
            record[f'early_{int(fraction*100)}']=float(a[:stop].mean())
            record[f'regret_{int(fraction*100)}']=float(regret[:stop].sum(0).mean())
        summaries.append(record)
        # Probe only this condition/regime's networks, without any extra learning.
        small=Population(genomes[name],seed_for(seed,'test_birth'),config,beta_zero=name in ['individual','ablated'])
        with torch.no_grad():
            small.weights.copy_(population.weights[sl])
        diagnostics.append(dict(seed=seed,condition=name,regime=label,**probes(small)))
    return rows,summaries,diagnostics,transfer_genomes
