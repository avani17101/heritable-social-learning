"""Only genome arrays, never agents, enter reproduction."""
import numpy as np
from scipy.special import expit, logit


def decode(genomes):
    return expit(genomes)


def encode(parameters):
    return logit(np.clip(parameters, 1e-6, 1-1e-6))


def reproduce(genomes, fitness, rng, elite_fraction=.25, mutation_scale=.25, shuffle=False):
    n = len(genomes)
    elite = np.argsort(fitness, kind='stable')[-max(1, int(n*elite_fraction)):]
    parents = rng.choice(elite, size=n)
    children = genomes[parents].copy() + rng.normal(0, mutation_scale, (n,3))
    assignment = np.arange(n)
    if shuffle:
        if n < 2:
            raise ValueError('Shuffling requires at least two newborns')
        while np.any(assignment == np.arange(n)):
            assignment = rng.permutation(n)
    return children[assignment].copy(), parents, parents[assignment], assignment
