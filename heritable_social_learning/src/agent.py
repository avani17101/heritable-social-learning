"""A vectorized population of independent tiny linear neural policies."""
import numpy as np
import torch
from torch import nn
from .genome import decode
from .learning_rule import evidence_loss


class Population(nn.Module):
    def __init__(self, genomes, seed, config, beta_zero=False):
        super().__init__()
        requested = config.get('device', 'cpu')
        device = 'cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested)
        self.genomes = np.array(genomes, copy=True)
        self.register_buffer('parameters_decoded', torch.tensor(decode(genomes), dtype=torch.float32, device=device))
        if beta_zero:
            self.parameters_decoded[:,1] = 0
        generator = torch.Generator(device=device).manual_seed(seed)
        self.weights = nn.Parameter(torch.randn((len(genomes),13,4), generator=generator, device=device) * config['init_std'])
        self.optimizer = torch.optim.SGD([self.weights], lr=config['learning_rate'])

    def features(self, private, social, cue, available):
        onehot = torch.nn.functional.one_hot
        use_social = available & (self.parameters_decoded[:,1] != 0)
        peer = onehot(social,4).float() * use_social[...,None]
        return torch.cat([onehot(private,4).float(), peer * (~cue)[...,None],
                          peer * cue[...,None], torch.ones_like(private[...,None], dtype=torch.float32)], -1)

    def forward(self, private, social, cue, available):
        return torch.einsum('...ni,nij->...nj', self.features(private,social,cue,available), self.weights)

    def learn(self, logits, private, feedback, social, cue, available):
        self.optimizer.zero_grad(set_to_none=True)
        # Sum, not mean: each learner receives its own full-sized gradient.
        evidence_loss(logits,private,feedback,social,cue,available,self.parameters_decoded).sum().backward()
        self.optimizer.step()


def lifetime(population, data, learn=True):
    device = population.weights.device
    tensors = {k:torch.as_tensor(v,device=device) for k,v in data.items() if k in
               ('private','social','cue','available','feedback','z','reward_flip')}
    episodes, steps, n = data['z'].shape
    accuracy = np.empty((episodes,n))
    rewards = np.empty_like(accuracy)
    for e in range(episodes):
        correct, observed = [], []
        for t in range(steps):
            d = {k:v[e,t] for k,v in tensors.items()}
            with torch.set_grad_enabled(learn):
                logits = population(d['private'],d['social'],d['cue'],d['available'])
                hit = logits.argmax(-1) == d['z']
                correct.append(hit.detach())
                observed.append(torch.logical_xor(hit,d['reward_flip']).detach())
                if learn:
                    population.learn(logits,d['private'],d['feedback'],d['social'],d['cue'],d['available'])
        accuracy[e] = torch.stack(correct).float().mean(0).cpu().numpy()
        rewards[e] = torch.stack(observed).float().mean(0).cpu().numpy()
    return accuracy, rewards


def probes(population):
    """Average action probabilities on all 12 conflicting signals per cue."""
    n, device = len(population.genomes), population.weights.device
    out = {}
    with torch.no_grad():
        for cue in [False,True]:
            private, social = zip(*[(i,j) for i in range(4) for j in range(4) if i!=j])
            p = torch.tensor(private,device=device)[:,None].expand(-1,n)
            s = torch.tensor(social,device=device)[:,None].expand(-1,n)
            c = torch.full_like(p,cue,dtype=torch.bool)
            probs = population(p,s,c,torch.ones_like(c)).softmax(-1)
            label = 'high' if cue else 'low'
            out['private_'+label] = probs.gather(-1,p[...,None]).mean().item()
            out['social_'+label] = probs.gather(-1,s[...,None]).mean().item()
    out['reject_low'] = 1-out['social_low']
    out['selectivity'] = out['social_high']-out['social_low']
    return out
