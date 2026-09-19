"""Noisy categorical evidence; latent state is constant for ten steps."""
import itertools
import numpy as np

K = 4


def signal(rng, z, reliability):
    wrong = (z + rng.integers(1, K, size=z.shape)) % K
    return np.where(rng.random(z.shape) < reliability, z, wrong)


def cue_reliability(p_social, cue):
    """P(peer correct | noisy high/low cue), by Bayes' rule."""
    mix = (p_social - .35) / .6
    high = np.where(cue, .85, .15)
    low = 1 - high
    posterior = mix * high / (mix * high + (1 - mix) * low)
    return .35 + .6 * posterior


def sample(seed, episodes, steps, batch, regimes):
    rng = np.random.default_rng(seed)
    regime = np.asarray(regimes)[rng.integers(len(regimes), size=episodes)]
    pp, ps = regime[:, 0, None, None], regime[:, 1, None, None]
    z = np.broadcast_to(rng.integers(K, size=(episodes, 1, batch)), (episodes, steps, batch)).copy()
    private = signal(rng, z, pp)
    # Outside the mixture range, controls use a constant reliability and an uninformative cue.
    mixture = (ps >= .35) & (ps <= .95)
    high = rng.random(z.shape) < np.clip((ps - .35) / .6, 0, 1)
    reliability = np.where(mixture, np.where(high, .95, .35), ps)
    peer = signal(rng, z, reliability)
    cue = np.logical_xor(high, rng.random(z.shape) > .85)
    cue = np.where(mixture, cue, rng.random(z.shape) < .5)
    social = np.zeros_like(peer)
    social[:, 1:] = peer[:, :-1]
    delayed_cue = np.zeros_like(cue)
    delayed_cue[:, 1:] = cue[:, :-1]
    available = np.ones_like(z, dtype=bool)
    available[:, 0] = False
    return dict(z=z, private=private, social=social, cue=delayed_cue,
                available=available, feedback=signal(rng, z, .65),
                reward_flip=rng.random(z.shape) < .1, regimes=regime,
                peer=peer, peer_cue=cue)


def analytical(pp=.7, ps=.7):
    """Exact enumeration for one private + one cued peer observation.

    Bayes accuracy = sum_observation max_z P(z, observation).
    Every joint probability is positive: no observation identifies z.
    """
    mix = (ps - .35) / .6
    accuracy = 0.
    minimum = 1.
    follows_social = []
    for private, peer, cue in itertools.product(range(K), range(K), (False, True)):
        joint = []
        for z in range(K):
            p_private = pp if private == z else (1 - pp) / 3
            p_peer_cue = sum(
                m * (r if peer == z else (1-r)/3) * (.85 if h == cue else .15)
                for h, m, r in [(True, mix, .95), (False, 1-mix, .35)])
            joint.append(.25 * p_private * p_peer_cue)
        accuracy += max(joint)
        minimum = min(minimum, min(joint))
        if private != peer:
            follows_social.append((cue, int(np.argmax(joint)) == peer))
    return dict(private=pp, blind_copy=ps, optimal=accuracy, min_joint=minimum,
                follow_peer_low=float(np.mean([v for c,v in follows_social if not c])),
                follow_peer_high=float(np.mean([v for c,v in follows_social if c])))
