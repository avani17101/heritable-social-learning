"""Selection receives held-out scores; birth receives only genome arrays."""
import numpy as np
import torch
from .agent import Population, lifetime
from .environment import sample
from .genome import decode, encode, reproduce
from .utils import CONDITIONS, seed_for


def evolve(config, seed, frozen, conditions=None):
    names = conditions or CONDITIONS + ['specialist_private','specialist_social']
    n, steps = config['population_size'], config['episode_length']
    initial = np.random.default_rng(seed_for(seed,'initial_genomes')).normal(size=(n,3))
    genomes, rngs = {}, {}
    for name in names:
        parameters = frozen if name=='best_fixed' else [.5,.5,.5]
        genomes[name] = (initial.copy() if name in ['evolved','shuffled','specialist_private','specialist_social']
                         else np.tile(encode(parameters),(n,1)))
        rngs[name] = np.random.default_rng(seed_for(seed,'mutation'))
    logs, curves, lineage, champions = [], [], [], {}
    for generation in range(config['generations']):
        combined = np.concatenate([genomes[name] for name in names])
        population = Population(combined,seed_for(seed,'birth',generation),config)
        # Pair exact initial weights by slot across every condition.
        with torch.no_grad():
            base = population.weights[:n].clone()
            for j,name in enumerate(names):
                population.weights[j*n:(j+1)*n].copy_(base)
                if name=='individual':
                    population.parameters_decoded[j*n:(j+1)*n,1]=0
        blocks = []
        for name in names:
            regimes = ([[.8,.6]] if name=='specialist_private' else [[.6,.8]]
                       if name=='specialist_social' else config['train'])
            blocks.append(sample(seed_for(seed,'learning',generation),config['learning_episodes'],steps,n,regimes))
        training = {k:np.concatenate([b[k] for b in blocks],axis=-1) if blocks[0][k].ndim==3 else blocks[0][k]
                    for k in blocks[0]}
        train_accuracy, train_rewards = lifetime(population,training)
        blocks = []
        for name in names:
            regimes = ([[.8,.6]] if name=='specialist_private' else [[.6,.8]]
                       if name=='specialist_social' else config['train'])
            blocks.append(sample(seed_for(seed,'fitness',generation),config['evaluation_episodes'],steps,n,regimes))
        evaluation = {k:np.concatenate([b[k] for b in blocks],axis=-1) if blocks[0][k].ndim==3 else blocks[0][k]
                      for k in blocks[0]}
        accuracy, rewards = lifetime(population,evaluation,learn=False)
        fitness = accuracy.mean(0)
        for j,name in enumerate(names):
            sl = slice(j*n,(j+1)*n)
            f = fitness[sl]
            decoded = decode(genomes[name])
            if name=='individual':
                decoded[:,1]=0
            for agent in range(n):
                logs.append(dict(seed=seed,condition=name,generation=generation,agent=agent,
                                 fitness=float(f[agent]),alpha=decoded[agent,0],beta=decoded[agent,1],gamma=decoded[agent,2],
                                 g_alpha=genomes[name][agent,0],g_beta=genomes[name][agent,1],g_gamma=genomes[name][agent,2]))
            for phase, acc, rew in [('learning',train_accuracy,train_rewards),('fitness',accuracy,rewards)]:
                for ep in range(len(acc)):
                    for agent in range(n):
                        curves.append(dict(seed=seed,condition=name,generation=generation,phase=phase,episode=ep,agent=agent,
                                           accuracy=float(acc[ep,j*n+agent]),reward=float(rew[ep,j*n+agent])))
            champions[name] = genomes[name][np.argmax(f)].tolist()
            if generation+1 < config['generations'] and name in ['evolved','shuffled','specialist_private','specialist_social']:
                child, nominal, donor, assignment = reproduce(genomes[name],f,rngs[name],config['elite_fraction'],
                                                              config['mutation_scale'],name=='shuffled')
                for slot in range(n):
                    lineage.append(dict(seed=seed,condition=name,generation=generation+1,slot=slot,
                                        nominal_parent=int(nominal[slot]),genetic_donor=int(donor[slot]),
                                        source_offspring_slot=int(assignment[slot])))
                genomes[name] = child
    return champions, logs, curves, lineage
