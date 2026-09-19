"""Seed-level paired bootstrap statistics; agents are never independent replicates."""
import numpy as np
from scipy.stats import bootstrap


def describe(values, confidence=.95, seed=173):
    """Percentile CI of the mean, resampling independent evolutionary seeds."""
    values=np.asarray(values,dtype=float)
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent seed-level observations')
    result=bootstrap((values,),np.mean,confidence_level=confidence,method='percentile',
                     n_resamples=10000,rng=np.random.default_rng(seed))
    std=float(values.std(ddof=1))
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),std=std,
                ci_low=float(result.confidence_interval.low),ci_high=float(result.confidence_interval.high))


def paired(treatment,control,confidence=.95):
    """Positive control-minus-treatment = treatment has less regret."""
    t,c=np.asarray(treatment),np.asarray(control)
    if t.shape!=c.shape:
        raise ValueError('Paired observations must align by seed')
    delta=c-t
    result=describe(delta,confidence)
    result['effect_dz']=float(delta.mean()/delta.std(ddof=1)) if delta.std(ddof=1)>0 else None
    result['positive_seeds']=int(np.sum(delta>0))
    result['negative_seeds']=int(np.sum(delta<0))
    result['tied_seeds']=int(np.sum(delta==0))
    return result


def verdict(result):
    if result['ci_low']>0:
        return 'SUPPORTED'
    if result['ci_high']<=0:
        return 'NOT SUPPORTED'
    return 'INCONCLUSIVE'
