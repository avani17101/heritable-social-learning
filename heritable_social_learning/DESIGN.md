# Pre-registration — 2026-09-19, before neural experiments

## Question and scope
Can selection of three inherited learning coefficients improve newborn learning?
This is a heritable social-learning strategy, not inherited knowledge, biological
genes, cumulative culture, or a rich developmental genome. Only three floats cross
the generation boundary. All network weights are initialized independently at birth.

## Exact environment
Four equally likely states, one fixed state per ten-step episode. Private signals
have reliability p_private; mistakes are uniform over the other three states.
Peers are exogenous independent demonstrators, not members of the evolving population.
Their reliability is .95 or .35, mixed to achieve p_social. A binary reliability
cue identifies the peer type with probability .85. The learner sees the previous
step's peer signal and cue; neither is available at step zero. There are no peer
rewards. All finite observation strings retain positive likelihood for every state.
An independent .65-reliable categorical outcome arrives after acting, as private
feedback. The task score is correctness; observed scalar reward is correctness
flipped with probability .1. It is recorded, while learning uses the categorical
feedback. True state and correctness are accessible only to the evaluator.

The first analytical check enumerates all joint observations, demonstrating
.802083 Bayesian accuracy versus .7 for private-only and blind copying. It also
checks a reversal of the optimal conflict rule. This one-step comparison does
not claim optimality for the neural lifetime task or incorporate post-action feedback.

## Agent and learning rule
Policy: a tiny linear softmax network with 13 inputs and four outputs, freshly
initialized N(0,.05²) at birth. Inputs are one-hot current private signal, one-hot
previous peer signal in separate high/low cue blocks, and a constant. Weights
persist across lifetime episodes, but no episodic state or replay buffer is kept.
This deliberately memory-limited learner does not accumulate the full history.
Genomes do not appear in the policy's forward pass except for the exact beta=0
social-channel ablation, which removes all social features and updates.

G is three real logits, decoded with sigmoid to alpha, beta, gamma. Exact beta=0
is an intervention after decoding. With cue score q=.85 (high) or .15 (low),
w=q^(4 gamma), masked to zero when the delayed observation is absent.
L_individual = CE(private_signal) + CE(noisy_feedback).
L_social = CE(delayed_peer_signal).
L = [alpha L_individual + beta w L_social] / [2 alpha + beta w].
The normalized loss is a convex mixture of categorical evidence targets;
it is a generalized evidence-weighting rule, not an exact Bayesian posterior.
Gamma changes the high/low trust ratio (.85/.15)^(4 gamma), not just a uniform
learning-rate multiplier. Alpha/beta scale is non-identifiable after normalization;
their ratio and gamma are the meaningful quantities. This limitation will be
reported rather than interpreted as independent evolution of three mechanisms.
SGD has fixed learning rate .15, no momentum, and one update after every action.
Vectorization batches independent learners without averaging their gradients together.
Identical initialization and task seeds pair all intervention arms.

## Evolution and baseline
Final: 32 agents, 50 generations, elite fraction .25, Gaussian mutation SD .25.
Each generation gets fresh networks, 24 learning episodes, and 24 held-out episodes
without weight updates. Selection uses held-out accuracy. All conditions have
the same per-generation lifetime and evaluation budget. Fixed and individual-only
conditions repeat independently born populations over the same generations.
Evolution training grid: {.65,.75}². Validation grid: {.625,.725,.775}².
Test grid: {.60,.70,.80}². Separate seed namespaces ensure no trajectory reuse.
27 fixed strategies from {.2,.5,.8}³ are screened on training environments;
the best nine are ranked on validation with seeds 1000–1004 and frozen before
any final test. Individual-only uses alpha=.5, beta=0, gamma=.5.
Final champion is the best held-out fitness member of the last generation,
never a genome picked by final test performance. Baseline search budget and
evolution budget are reported separately; this is not an equal-search-cost claim.

## Lineage control: identifiability warning
The requested control permutes mutated selected genomes among newborn slots,
preserving the distribution, mutation and budgets at that reproduction event.
A derangement ensures every assignment changes. Nominal parent IDs and actual
genetic donor IDs are both logged. This destroys slot correspondence, but does
NOT remove genetic inheritance from the population. With exchangeable newborns
and no other inherited state, permutations leave the population law unchanged.
Thus H3 is structurally unidentifiable as an inheritance-causality hypothesis.
We implement and report the contrast as a negative control, never treat a chance
positive difference as proof of lineage dependence. Subsequent distributions
can diverge because slot-specific training and initialization noise differs.

## Hypotheses and decision rules
The independent unit is an evolutionary run: seeds 0–19, none excluded.
Primary metric: cumulative correctness regret sum_t(1 - 1[action=z]) in a
fresh 40-episode newborn lifetime, averaged over 32 newborns and the nine test
environments. This is regret relative to an omniscient state oracle, not a
Bayes observer; irreducible observation noise contributes to it. Expected noisy
reward regret equals .8 times this quantity and is logged separately.
Positive paired difference means the named treatment has lower regret.
Use scipy paired percentile bootstrap, 10,000 seed-level resamples, 95% CI,
mean/median/sample SD and paired standardized effect (mean difference / SD).
H1, H2, H4 are planned confirmatory comparisons; Bonferroni simultaneous
98.333% intervals also reported and used for their verdicts. A lower bound >0
supports the claim, an upper bound <=0 is NOT SUPPORTED; overlap is INCONCLUSIVE.
No per-environment significance fishing. Nulls and adverse results stay in the report.

- H1: evolved social learner vs individual-only on test cumulative regret.
- H2: evolved vs frozen best-fixed on test cumulative regret. The same contrast
  under a shift is secondary; it cannot rescue a negative primary result.
- H3: evolved vs lineage-shuffled; INCONCLUSIVE by structural non-identifiability.
- H4: evolved vs random genomes, first 25% newborn cumulative regret, paired
  identical network initialization, environment and learning budget.
- H5: two additional independent selection regimes (.80,.60) and (.60,.80),
  paired seeds, predict higher effective social/private weight in the latter.
  Effective weight = beta * mean_q(q^(4 gamma)) / (2 alpha), under a balanced
  diagnostic cue distribution. Bootstrap paired difference; exploratory 95% CI.
  Test-grid generalization is descriptive and includes all nine environments.

## Secondary metrics and controls
Early accuracy at 10%,25%,50%; pre-update birth accuracy; final ten-episode
accuracy; full cumulative regret; post-shift curve. Shift halfway through a
40-episode lifetime: (.80,.60) to (.60,.80), no phase flag. Adaptation lag is the
first post-shift trailing three-episode mean reaching 90% of the last five
pre-shift episode mean. Right-censor if not reached; report censored counts,
do not replace them with successful recoveries. A low pre-shift score can make
this threshold easy, so also report the full curve and regret.
Controlled probes use all conflicting private/peer pairs at both cues and
report mean policy probability following each source and rejection of low-cue
peers. Probe order does not update networks. Random genome, evolved beta=0,
random dimension permutation, no-social (p=.25) and near-perfect-social (p=.99)
controls all use fresh networks. Control environments have uninformative cues.
These are diagnostics, not guarantees of desirable outcomes.

## Reuse and provenance
NumPy: RNG, enumeration and array operations. PyTorch: softmax, autograd, SGD.
SciPy: sigmoid/logit and bootstrap. Pandas/PyArrow: Parquet. Matplotlib: figures.
Custom code only implements this environment, genome selection and domain metrics.
Every result row carries seed, config hash and source hash. There is initially no
Git repository, so git_sha is null and an immutable content-hashed source snapshot
is saved; no fabricated commit identifier. Provenance checker verifies snapshots,
configuration hashes, baseline freeze and absence of duplicate observation keys.

## Deviations (append only)
None at registration. The analytical environment gate passed before neural code.

- 2026-09-19, before final runs: A workspace Git repository became available after
  initial inspection. Provenance now creates a clean isolated Git snapshot/bundle
  of the exact source files; every row references that commit as well as content
  and configuration hashes. This avoids attributing uncommitted code to a base
  commit and does not commit or modify the user's working repository.
- 2026-09-19, debug validation: fixed a test-only NumPy `.fill_` typo. No change
  to the registered environment, model, comparisons, seeds or decision rules.
