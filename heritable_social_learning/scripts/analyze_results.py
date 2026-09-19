"""Generate all planned plots and an honest hypothesis-by-hypothesis report."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,cli,load_config
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.statistics import describe,paired,verdict
from check_provenance import check

COLORS={'evolved':'#146b69','best_fixed':'#bd7728','individual':'#697581',
        'fixed':'#769ac2','shuffled':'#a26078','random':'#8757a5','ablated':'#b55847','scrambled':'#708742'}


def interval(matrix):
    """Pointwise 95% bootstrap bands over seeds, retaining each seed's whole curve."""
    from scipy.stats import bootstrap
    a=np.asarray(matrix)
    result=bootstrap((a,),np.mean,axis=0,vectorized=True,method='percentile',n_resamples=2000,
                     rng=np.random.default_rng(41))
    return a.mean(0),result.confidence_interval.low,result.confidence_interval.high


def line(ax,frame,x,y,label,color=None):
    matrix=frame.pivot(index='seed',columns=x,values=y).sort_index(axis=1)
    mean,lo,hi=interval(matrix.values)
    ax.plot(matrix.columns,mean,label=label,color=color)
    ax.fill_between(matrix.columns,lo,hi,alpha=.15,color=color)


def save(fig,figdir,name):
    fig.text(.01,.005,'Source: saved Parquet runs · seed-level pointwise 95% bootstrap intervals',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.03,1,1))
    fig.savefig(figdir/f'{name}.png',dpi=180)
    fig.savefig(figdir/f'{name}.pdf')
    plt.close(fig)


def analyze(config,output):
    check(output)
    completed=sorted(output.glob('seed_*_all/complete.json'))
    if len(completed)!=config['seeds']:
        raise ValueError(f'Expected {config["seeds"]} complete runs; found {len(completed)}. No silent exclusions.')
    run_info=[json.loads(p.read_text()) for p in completed]
    assert sorted(r['seed'] for r in run_info)==list(range(config['seeds']))
    assert len({r['source_hash'] for r in run_info})==1,'Do not pool different source versions'
    assert all(r['config']==config for r in run_info)
    def read(name):
        return pd.concat([pd.read_parquet(p.parent/f'{name}.parquet') for p in completed],ignore_index=True)
    summary,episodes,genomes,diagnostics=read('summary'),read('test_episodes'),read('genomes'),read('diagnostics')
    baseline=json.loads((output/'baseline.json').read_text())
    test=summary[summary.regime.str.startswith('test_')]
    primary=test.groupby(['seed','condition']).mean(numeric_only=True).reset_index()
    pivot=primary.pivot(index='seed',columns='condition',values='regret').sort_index()
    stats={name:describe(pivot[name]) for name in pivot}
    comparisons={}
    for hypothesis,control,metric in [('H1','individual','regret'),('H2','best_fixed','regret'),('H4','random','regret_25')]:
        p=primary.pivot(index='seed',columns='condition',values=metric).sort_index()
        normal=paired(p.evolved,p[control])
        simultaneous=paired(p.evolved,p[control],1-.05/3)
        comparisons[hypothesis]=dict(control=control,metric=metric,**normal,
                                     simultaneous_ci=[simultaneous['ci_low'],simultaneous['ci_high']],
                                     verdict=verdict(simultaneous))
    comparisons['H3']=dict(**paired(pivot.evolved,pivot.shuffled),verdict='INCONCLUSIVE',
                          reason='Genome permutation among exchangeable newborns does not remove genetic inheritance.')
    specialists=[]
    for run in run_info:
        from src.genome import decode
        row={'seed':run['seed']}
        for name in ['specialist_private','specialist_social']:
            a,b,g=decode(run['champions'][name])
            row[name]=float(b*(.85**(4*g)+.15**(4*g))/(4*a))
        specialists.append(row)
    spec=pd.DataFrame(specialists)
    # Expected direction is larger social weight under the social-favoring regime.
    h5=paired(spec.specialist_private,spec.specialist_social)
    comparisons['H5']=dict(**h5,verdict=verdict(h5),exploratory=True)
    analysis=output/'analysis'
    analysis.mkdir(exist_ok=True)
    spec.to_csv(analysis/'specialization.csv',index=False)
    primary.to_csv(analysis/'seed_metrics.csv',index=False)
    (analysis/'statistics.json').write_text(json.dumps(dict(conditions=stats,hypotheses=comparisons),indent=2))
    figdir=ROOT/'figures'/output.name
    figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.figsize':(9,5),'axes.titleweight':'bold'})
    main=['individual','fixed','best_fixed','evolved','shuffled']
    fig,ax=plt.subplots()
    fitness=genomes.groupby(['seed','condition','generation']).fitness.mean().reset_index()
    for name in main:
        line(ax,fitness[fitness.condition==name],'generation','fitness',name,COLORS[name])
    ax.set(title='Held-out fitness during evolution',xlabel='Generation',ylabel='Mean correctness (fraction)')
    ax.legend(ncol=3)
    save(fig,figdir,'01_fitness')
    learning=episodes[episodes.regime.str.startswith('test_')].groupby(['seed','condition','episode']).regret.mean().reset_index()
    learning['cumulative_regret']=learning.groupby(['seed','condition']).regret.cumsum()
    fig,ax=plt.subplots()
    for name in main:
        line(ax,learning[learning.condition==name],'episode','cumulative_regret',name,COLORS[name])
    ax.set(title='Newborn cumulative regret on held-out environments',xlabel='Lifetime episode (zero-indexed)',ylabel='Cumulative correctness regret (actions)')
    ax.legend()
    save(fig,figdir,'02_regret')
    shifted=episodes[episodes.regime=='shift'].groupby(['seed','condition','episode']).accuracy.mean().reset_index()
    fig,ax=plt.subplots()
    for name in main:
        line(ax,shifted[shifted.condition==name],'episode','accuracy',name,COLORS[name])
    ax.axvline(config['test_episodes']//2-.5,color='black',linestyle='--',label='Reliability shift')
    ax.set(title='Adaptation after an unannounced reliability shift',xlabel='Lifetime episode (zero-indexed)',ylabel='Correctness (fraction)')
    ax.legend(ncol=3)
    save(fig,figdir,'03_shift')
    mean_genomes=genomes[genomes.condition=='evolved'].groupby(['seed','generation'])[['alpha','beta','gamma']].mean().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,key in zip(axes,['alpha','beta','gamma']):
        line(ax,mean_genomes,'generation',key,key,COLORS['evolved'])
        ax.set(title=key,xlabel='Generation',ylabel='Mean decoded parameter',ylim=(0,1))
    fig.suptitle('Evolution of the learning prior')
    save(fig,figdir,'04_genome_trajectory')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    bins=np.linspace(0,1,11)
    for ax,key in zip(axes,['alpha','beta','gamma']):
        for generation,label,color in [(0,'Initial','#697581'),(config['generations']-1,'Final','#146b69')]:
            hist=[]
            for seed in range(config['seeds']):
                values=genomes[(genomes.seed==seed)&(genomes.condition=='evolved')&(genomes.generation==generation)][key]
                hist.append(np.histogram(values,bins=bins)[0]/len(values))
            mean,lo,hi=interval(np.asarray(hist))
            x=(bins[:-1]+bins[1:])/2
            ax.plot(x,mean,label=label,color=color)
            ax.fill_between(x,lo,hi,color=color,alpha=.15)
        ax.set(title=key,xlabel='Decoded parameter bin',ylabel='Population fraction')
    axes[0].legend()
    fig.suptitle('Beginning and end genome distributions')
    save(fig,figdir,'05_genome_distribution')
    fig,ax=plt.subplots()
    delta=pivot.shuffled-pivot.evolved
    ax.axhline(0,color='gray',linewidth=1)
    ax.scatter(delta.index,delta,color=COLORS['evolved'])
    h3=comparisons['H3']
    ax.axhspan(h3['ci_low'],h3['ci_high'],alpha=.15,color=COLORS['evolved'],label='95% CI of mean difference')
    ax.axhline(h3['mean'],color=COLORS['evolved'],label='Mean difference')
    ax.set(title='Lineage shuffle: population permutation negative control',xlabel='Independent evolutionary seed',ylabel='Shuffled − evolved regret (actions)')
    ax.legend()
    save(fig,figdir,'06_lineage')
    fig,ax=plt.subplots()
    for name in ['evolved','random','best_fixed','ablated']:
        line(ax,learning[learning.condition==name],'episode','cumulative_regret',name,COLORS[name])
    ax.axvline(max(1,int(config['test_episodes']*.25))-1,color='gray',linestyle='--',label='Primary transfer window')
    ax.set(title='Frozen-genome transfer into identically initialized newborns',xlabel='Lifetime episode (zero-indexed)',ylabel='Cumulative correctness regret (actions)')
    ax.legend()
    save(fig,figdir,'07_transfer')
    fig,ax=plt.subplots(figsize=(11,5))
    labels=sorted(test.regime.unique())
    for offset,name in enumerate(['evolved','best_fixed','individual']):
        means=[]; lower=[]; upper=[]
        for label in labels:
            result=describe(test[(test.condition==name)&(test.regime==label)].regret)
            means.append(result['mean']); lower.append(result['mean']-result['ci_low']); upper.append(result['ci_high']-result['mean'])
        ax.errorbar(np.arange(len(labels))+(offset-1)*.18,means,yerr=[lower,upper],fmt='o',capsize=3,label=name,color=COLORS[name])
    ax.set_xticks(range(len(labels)),[s.replace('test_','').replace('_',' / ') for s in labels])
    ax.set(title='Generalization across unseen reliability pairs',xlabel='Private / social reliability',ylabel='Lifetime cumulative regret (actions)')
    ax.legend()
    save(fig,figdir,'08_generalization')
    # Supplementary behavioral diagnostics and all requested controls.
    probe=diagnostics[diagnostics.regime.str.startswith('test_')].groupby(['seed','condition']).mean(numeric_only=True).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['social_high','social_low']):
        for x,name in enumerate(main):
            s=describe(probe[probe.condition==name][metric])
            ax.errorbar(x,s['mean'],yerr=[[s['mean']-s['ci_low']],[s['ci_high']-s['mean']]],fmt='o',color=COLORS[name],capsize=4)
        ax.set_xticks(range(len(main)),main,rotation=20)
        ax.set(title=metric.replace('_',' '),xlabel='Condition',ylabel='Probability of following peer on conflicts',ylim=(0,1))
    save(fig,figdir,'09_behavior')
    report=[]
    report.append('# Results: heritable social-learning strategy\n')
    report.append(f"Completed {config['seeds']} independent evolutionary seeds; no failed or excluded seeds. "
                  f"H2 (evolved vs tuned fixed): **{comparisons['H2']['verdict']}**. "
                  'This is a minimal causal learning-prior prototype, not evidence of cumulative culture.\n')
    report.append('## 1. Experimental question\nDoes selecting a three-dimensional inherited learning prior improve learning in fresh neural agents?\n')
    report.append('## 2. Exact environment\nFour equiprobable latent states, constant over ten steps; noisy private and delayed peer signals. '
                  'Peer reliability is a .35/.95 mixture with an 85%-accurate type cue. Independent 65%-reliable private feedback arrives after acting. '
                  'No hidden state enters the update. Analytical Bayesian accuracy at (.7,.7) is 80.208%, versus 70% for either source alone. '
                  'The neural learner has no history accumulator and is not the analytical Bayes observer.\n')
    report.append('## 3. Conditions\nIndividual-only; fixed (.5,.5,.5); train/validation-tuned best-fixed; evolved; lineage-shuffled. '
                  'Additional random, evolved beta=0, dimension-scrambled, no-social, near-perfect-social, and two specialization regimes.\n')
    report.append('## 4. Hyperparameters\n```json\n'+json.dumps(config,indent=2)+'\n```\n')
    report.append(f"Frozen baseline: {baseline['parameters']}; validation fitness {baseline['validation_score']:.4f}. "
                  '27 candidates screened on training, nine on validation, each over five independent tuning seeds. '
                  'Evolution evaluates 32 candidates per generation; search costs are not matched. See baseline.json and tuning.parquet.\n')
    report.append(f"## 5. Number of seeds\n{config['seeds']} paired evolutionary seeds (0–{config['seeds']-1}); agents are averaged within seeds. "
                  'Tuning seeds are 1000–1004. All test environments and weights are paired across conditions.\n')
    report.append('## 6. Primary metric\nCumulative correctness regret against an omniscient state oracle; lower is better. '
                  'Expected regret for the noisy scalar reward is 0.8 times correctness regret. '
                  'Primary H4 uses the first 25% of the newborn lifetime.\n')
    report.append('## 7. Results\nSeed-level cumulative regret, averaged across nine held-out environments:\n')
    for name in main+['random','ablated','scrambled']:
        s=stats[name]
        report.append(f"- **{name}**: mean {s['mean']:.3f}, median {s['median']:.3f}, SD {s['std']:.3f}, 95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}].")
    report.append('\nEarly learning and newborn measurements (accuracy):\n')
    for name in main+['random']:
        s=primary[primary.condition==name].mean(numeric_only=True)
        report.append(f"- {name}: birth {s.birth_accuracy:.3f}; first 10% {s.early_10:.3f}; 25% {s.early_25:.3f}; 50% {s.early_50:.3f}; final ten episodes {s.final_accuracy:.3f}.")
    report.append('\n## 8. Statistical analysis\nPaired bootstrap with 10,000 resamples at evolutionary-seed level. '
                  'Positive differences favor evolved. H1/H2/H4 verdicts use Bonferroni 98.333% intervals; 95% intervals are also reported.\n')
    for key in ['H1','H2','H4']:
        s=comparisons[key]
        lo,hi=s['simultaneous_ci']
        dz='undefined' if s['effect_dz'] is None else f"{s['effect_dz']:.3f}"
        report.append(f"- {key}, evolved vs {s['control']}: difference {s['mean']:.3f}; median {s['median']:.3f}; SD {s['std']:.3f}; "
                      f"95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]; simultaneous CI [{lo:.3f}, {hi:.3f}]; paired dz {dz}; "
                      f"positive/negative/tied seeds {s['positive_seeds']}/{s['negative_seeds']}/{s['tied_seeds']}; **{s['verdict']}**.")
    report.append('\n## 9. Genome evolution\nPopulation mean decoded parameters at the first and last generations:\n')
    for gen in [0,config['generations']-1]:
        values=mean_genomes[mean_genomes.generation==gen]
        parts=[]
        for key in ['alpha','beta','gamma']:
            s=describe(values[key]);parts.append(f"{key}={s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
        report.append(f"- Generation {gen}: "+'; '.join(parts)+'.')
    report.append(f"\nH5 specialization: social-favoring minus private-favoring effective social weight = {h5['mean']:.3f}, "
                  f"95% CI [{h5['ci_low']:.3f}, {h5['ci_high']:.3f}]; **{comparisons['H5']['verdict']}** (exploratory). "
                  'Only coefficient ratios are identifiable after loss normalization; absolute alpha/beta movements are not separate mechanisms.\n')
    report.append('## 10. Transfer results\nAll test metrics above use newborn networks and frozen selected genomes. '
                  'Evolved and random arms start with identical neural weights and receive identical trajectories. '
                  'Figure 7 shows the full learning curve. This measures an inherited learning bias, not inherited knowledge.\n')
    report.append(f"## 11. Lineage-shuffle results\nShuffled minus evolved regret {h3['mean']:.3f}, 95% CI [{h3['ci_low']:.3f}, {h3['ci_high']:.3f}]. "
                  '**H3: INCONCLUSIVE** as a causal inheritance claim. A derangement changes every offspring slot assignment, '
                  'but preserves inherited genomes. Identical-population permutation cannot distinguish lineage continuity from population-level selection.\n')
    report.append('## 12. Failure modes and controls\n')
    for regime in ['no_social','perfect_social']:
        part=summary[summary.regime==regime].pivot(index='seed',columns='condition',values='regret')
        s=paired(part.evolved,part.individual)
        report.append(f"- {regime}: individual minus evolved regret {s['mean']:.3f}, 95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}].")
    for control in ['ablated','scrambled']:
        s=paired(pivot.evolved,pivot[control])
        report.append(f"- {control} minus evolved regret {s['mean']:.3f}, 95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}].")
    report.append('\nShift recovery and secondary contrast:\n')
    shifted_summary=summary[summary.regime=='shift']
    for name in main:
        lag=shifted_summary[shifted_summary.condition==name].adaptation_lag
        uncensored=lag.dropna()
        median='not reached' if len(uncensored)==0 else f'{uncensored.median():.1f}'
        report.append(f"- {name}: {lag.isna().sum()}/{len(lag)} right-censored; median among observed recoveries {median} episodes.")
    shift_pivot=shifted_summary.pivot(index='seed',columns='condition',values='regret')
    shift_effect=paired(shift_pivot.evolved,shift_pivot.best_fixed)
    report.append(f"- Shift best-fixed minus evolved regret {shift_effect['mean']:.3f}, 95% CI [{shift_effect['ci_low']:.3f}, {shift_effect['ci_high']:.3f}] (secondary).")
    report.append('\nBehavioral conflict probes (mean action probabilities, after learning):\n')
    for name in main:
        s=probe[probe.condition==name].mean(numeric_only=True)
        report.append(f"- {name}: follow private at low/high cue {s.private_low:.3f}/{s.private_high:.3f}; "
                      f"follow peer at low/high cue {s.social_low:.3f}/{s.social_high:.3f}; reject low-cue peer {s.reject_low:.3f}.")
    report.append('\nA three-parameter search can be matched by a strong fixed strategy; selection noise and coarse baseline search both matter. '
                  'Peers are exogenous; the population does not socially interact. Learning uses a noisy teaching signal and imitation, '
                  'not reward-driven exploration. Beta=0 removes both social input and social updates, so its ablation is a channel-level intervention. '
                  'The gamma cue gate is hand-designed; evolution selects its degree. '
                  'The memory-limited policy can ignore repeated-state information that a Bayesian history model would exploit. '
                  'The adaptation threshold can be easy when pre-shift performance is low. No endpoint was changed after inspecting final outcomes.\n')
    report.append('## 13. Interpretation\n'+('Evolution outperformed the tested fixed search under this distribution. '
                  if comparisons['H2']['verdict']=='SUPPORTED' else 'The primary comparison does not establish an advantage over the tuned fixed strategy. ')+
                  'A useful evolved bias relative to random genomes would establish only that selection found a better prior in this small family. '
                  'Neither a positive transfer result nor lineage shuffling demonstrates cultural transmission.\n')
    report.append('## 14. Next experiment\nUse a denser fixed-strategy search and a Bayesian-history reference under the same learning budget; '
                  'replace the lineage permutation with an explicitly randomized parent-genome transmission intervention if a lineage claim is needed. '
                  'Do not proceed to a richer developmental genome until the current control limitations are resolved.\n')
    report.append('## Provenance and reuse\nNumPy RNG/arrays, PyTorch autograd/SGD, SciPy bootstrap/sigmoid, Pandas/PyArrow Parquet and Matplotlib plotting were reused. '
                  'Only environment, reproduction and task-specific metrics are custom. Each source row links to a config hash and full source snapshot. '
                  f"Source hash: `{run_info[0]['source_hash']}`. Clean snapshot Git SHA: `{run_info[0]['git_sha']}`. "
                  'An isolated Git bundle stores the exact executed source without committing the working project. '
                  'Raw rows are in seed folders; derived seed metrics/statistics are in analysis/.\n')
    report.append('## Figures\n')
    for file in sorted(figdir.glob('*.png')):
        report.append(f"![{file.stem}](../../figures/{output.name}/{file.name})\n")
    report.append('## Final hypothesis classifications\n')
    for key in ['H1','H2','H3','H4','H5']:
        report.append(f"- {key}: **{comparisons[key]['verdict']}**"+(' (exploratory)' if key=='H5' else ''))
    result='\n'.join(report)+'\n'
    (output/'RESULTS.md').write_text(result)
    if output.name=='final':
        (ROOT/'results/RESULTS.md').write_text(result.replace('../../figures/','../figures/'))
    print(json.dumps(comparisons,indent=2))
    return comparisons


if __name__=='__main__':
    args=cli(__doc__).parse_args()
    config=load_config(args.config)
    analyze(config,args.output or ROOT/'results'/Path(args.config).stem)
