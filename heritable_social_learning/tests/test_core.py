import numpy as np
import pytest
import torch
from src.agent import Population,lifetime
from src.environment import analytical,sample,cue_reliability
from src.genome import decode,encode,reproduce
from src.learning_rule import evidence_loss,social_weight
from src.evaluation import adaptation_lag
from src.utils import ROOT,load_config,setup


@pytest.fixture
def config():
    setup()
    return load_config(ROOT/'configs/debug.yaml')


def test_hidden_state_distribution():
    data=sample(1,20000,2,1,[[.7,.7]])
    counts=np.bincount(data['z'][:,0,0],minlength=4)/20000
    assert np.all(np.abs(counts-.25)<.015)
    assert np.array_equal(data['z'][:,0],data['z'][:,1])


def test_private_reliability():
    d=sample(2,3000,10,4,[[.7,.7]])
    assert abs(np.mean(d['private']==d['z'])-.7)<.01


def test_social_reliability_and_delay():
    d=sample(3,3000,10,4,[[.7,.7]])
    assert abs(np.mean(d['peer']==d['z'])-.7)<.01
    assert np.array_equal(d['social'][:,1:],d['peer'][:,:-1])
    assert not d['available'][:,0].any()
    for cue in [False,True]:
        mask=d['peer_cue']==cue
        actual=(d['peer']==d['z'])[mask].mean()
        assert abs(actual-cue_reliability(.7,cue))<.01


def test_no_oracle_and_analytical_advantage():
    a=analytical()
    assert a['min_joint']>0
    assert a['optimal']==pytest.approx(.8020833333333333)
    assert a['optimal']>a['private'] and a['optimal']>a['blind_copy']
    assert analytical(.85,.5)['follow_peer_high']==0
    assert analytical(.5,.85)['follow_peer_low']==1


def test_empirical_bayes_advantage():
    d=sample(9,30000,2,1,[[.7,.7]])
    p,s,c,z=[d[k][:,1,0] for k in ['private','social','cue','z']]
    action=np.where(c,s,p)
    assert (action==z).mean()>max((p==z).mean(),(s==z).mean())+.08


def test_beta_zero_removes_all_social_information(config):
    genomes=np.zeros((8,3))
    a,b=Population(genomes,1,config,True),Population(genomes,1,config,True)
    data=sample(2,4,10,8,[[.7,.7]])
    changed={k:v.copy() for k,v in data.items()}
    changed['social']=(changed['social']+1)%4
    changed['cue']=~changed['cue']
    out_a=lifetime(a,data)
    out_b=lifetime(b,changed)
    assert torch.equal(a.weights,b.weights)
    assert np.array_equal(out_a[0],out_b[0])


def test_gamma_changes_relative_selectivity():
    cue=torch.tensor([False,True])
    available=torch.ones(2,dtype=torch.bool)
    low=social_weight(torch.tensor(.2),cue,available)
    high=social_weight(torch.tensor(.8),cue,available)
    assert high[1]/high[0]>low[1]/low[0]


def test_genome_changes_gradient_direction_not_only_rate():
    gradients=[]
    for pars in [[.8,.2,.5],[.2,.8,.5]]:
        logits=torch.zeros((1,4),requires_grad=True)
        evidence_loss(logits,torch.tensor([0]),torch.tensor([0]),torch.tensor([1]),
                      torch.tensor([True]),torch.tensor([True]),torch.tensor([pars])).sum().backward()
        gradients.append(logits.grad[0])
    assert not torch.allclose(gradients[0]/gradients[0].norm(),gradients[1]/gradients[1].norm())


def test_children_have_fresh_weights_and_only_genome(config):
    g=np.zeros((8,3))
    parent=Population(g,1,config)
    lifetime(parent,sample(4,2,10,8,[[.7,.7]]))
    children,*_=reproduce(g,np.arange(8),np.random.default_rng(1),mutation_scale=0)
    child=Population(children,2,config)
    assert np.array_equal(child.genomes,g)
    assert not torch.equal(parent.weights,child.weights)
    assert not child.optimizer.state
    assert not hasattr(child,'memory')
    parent.weights.data.fill_(100)
    assert child.weights.max()<1
    children.fill(12)
    assert np.all(child.genomes==0)


def test_shuffling_preserves_multiset_destroys_assignments():
    g=np.arange(24).reshape(8,3).astype(float)
    a,*_=reproduce(g,np.arange(8),np.random.default_rng(3))
    b,nominal,donor,assignment=reproduce(g,np.arange(8),np.random.default_rng(3),shuffle=True)
    assert sorted(map(tuple,a))==sorted(map(tuple,b))
    assert np.all(assignment!=np.arange(8))
    assert np.array_equal(donor,nominal[assignment])


def test_reproducible(config):
    g=np.zeros((8,3))
    a,b=Population(g,11,config),Population(g,11,config)
    x=lifetime(a,sample(3,3,10,8,[[.7,.7]]))
    y=lifetime(b,sample(3,3,10,8,[[.7,.7]]))
    assert np.array_equal(x[0],y[0])
    assert torch.equal(a.weights,b.weights)


def test_evaluation_frozen(config):
    pop=Population(np.zeros((8,3)),1,config)
    before=pop.weights.detach().clone()
    lifetime(pop,sample(2,2,10,8,[[.7,.7]]),learn=False)
    assert torch.equal(before,pop.weights)


def test_no_hidden_state_in_learning(config):
    a,b=Population(np.zeros((8,3)),1,config),Population(np.zeros((8,3)),1,config)
    data=sample(2,2,10,8,[[.7,.7]])
    changed={k:v.copy() for k,v in data.items()}
    changed['z']=(changed['z']+1)%4
    changed['reward_flip']=~changed['reward_flip']
    lifetime(a,data)
    lifetime(b,changed)
    assert torch.equal(a.weights,b.weights)


def test_adaptation_censoring():
    assert np.isnan(adaptation_lag(np.array([1.]*5+[.1]*5),5))
    assert adaptation_lag(np.ones(10),5)==3


def test_genome_encode_roundtrip():
    x=np.array([.2,.5,.8])
    assert np.allclose(decode(encode(x)),x)


def test_no_social_and_perfect_social_channels():
    for reliability in [.25,.99]:
        d=sample(4,3000,10,4,[[.7,reliability]])
        assert abs(np.mean(d['peer']==d['z'])-reliability)<.01


def test_population_gradient_independent_of_batch_size(config):
    a=Population(np.zeros((1,3)),1,config)
    b=Population(np.zeros((8,3)),1,config)
    with torch.no_grad():
        b.weights.copy_(a.weights.expand_as(b.weights))
    one=sample(5,2,10,1,[[.7,.7]])
    many={k:np.repeat(v,8,axis=-1) if v.ndim==3 else v for k,v in one.items()}
    lifetime(a,one)
    lifetime(b,many)
    assert torch.allclose(a.weights[0],b.weights[0],atol=1e-7)


def test_complete_evolution_and_transfer_reproducible(config):
    from src.evolution import evolve
    from src.evaluation import evaluate
    c={**config,'population_size':4,'generations':2,'learning_episodes':2,
       'evaluation_episodes':2,'test_episodes':4,'episode_length':3}
    a=evolve(c,4,[.5,.5,.5])
    b=evolve(c,4,[.5,.5,.5])
    assert a==b
    x=evaluate(c,4,a[0])
    y=evaluate(c,4,b[0])
    assert x[0]==y[0] and x[2]==y[2] and x[3]==y[3]
    # Birth forward behavior does not depend on a non-ablated inherited genome.
    born={r['condition']:r['birth_accuracy'] for r in x[1] if r['regime']=='test_0.60_0.60'}
    assert born['evolved']==born['random']==born['fixed']==born['best_fixed']


def test_batched_permutation_equivariance(config):
    g=np.random.default_rng(2).normal(size=(8,3))
    order=np.array([3,0,6,2,7,1,4,5])
    a,b=Population(g,1,config),Population(g[order],1,config)
    with torch.no_grad():
        b.weights.copy_(a.weights[order])
    data=sample(3,3,10,8,[[.7,.7]])
    permuted={k:v[...,order] if v.ndim==3 else v for k,v in data.items()}
    before,_=lifetime(a,data)
    after,_=lifetime(b,permuted)
    assert np.array_equal(before[:,order],after)
    assert torch.allclose(a.weights[order],b.weights,atol=1e-7)
