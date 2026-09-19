# Future research

The present prototype finds useful transfer relative to random priors. It has
not established an advantage over the tested tuned fixed strategy. The tasks
below strengthen that comparison before expanding the mechanism. These are
proposed experiments, not results or changes to the completed protocol.

## 1. Make the strongest comparison fair

- [ ] **Match the outer search budget.** Give evolutionary search and fixed-prior
  search equal numbers of agent-lifetimes, learning updates and held-out
  evaluations. Report candidate diversity and replications per candidate as well
  as total compute. Repeating a fixed condition without using those observations
  to select its prior does not count as additional tuning.
- [ ] **Repeat baseline tuning for each independent outer seed.** Use nested
  train/validation/test splits and report variability from both search methods.
  The current comparison conditions on one shared baseline search.
- [ ] **Search the same effective parameter domain.** Compare evolution against
  random search and a denser fixed search over evidence ratios and selectivity.
  The current {.2,.5,.8} grid does not cover the extreme ratios evolution finds.
- [ ] **Align selection with the primary question.** Select both methods on
  cumulative lifetime regret if the claim concerns learning speed. Retain
  post-learning accuracy as a separate endpoint; currently selection optimizes
  that endpoint while the main test measures lifetime regret.
- [ ] **Run a new confirmation on untouched data.** Register the revised protocol,
  thresholds and fresh seed sets before running it. Preserve the original and
  corrected studies; do not call another pass over their test seeds independent
  confirmation.

**Decision criterion:** assess the registered paired regret difference against
the strongest equally budgeted baseline. Report superiority, a null or a reversal
without using selected environments to rescue the primary claim. Claiming
equivalence would require a separately registered equivalence margin and test.

## 2. Identify what the prior is doing

- [ ] **Address parameter non-identifiability.** Because the loss is normalized,
  the common scale of alpha and beta cancels. Analyze beta/alpha and gamma, or
  reparameterize to two identifiable quantities before interpreting their roles.
- [ ] **Separate social input from social teaching.** Add controls that retain
  peer features while removing the social loss, and vice versa. The current
  beta=0 intervention removes both and therefore measures a combined channel
  effect.
- [ ] **Check learning-rate sensitivity symmetrically.** Use the same learning-rate
  search and initialization distribution for each arm. If a claimed advantage
  vanishes under matched retuning, report it as such.
- [ ] **Add an information-matched Bayesian reference.** Use the same observation
  timing and available history. Keep this distinct from the omniscient-state
  oracle currently used to define regret. Measure the gap attributable to the
  learner rather than irreducible signal noise.

**Decision criterion:** a proposed explanation must predict an intervention on
behavioral probes and held-out learning curves, beyond a change in raw reward.

## 3. Test when social trust becomes a liability

- [ ] **Train and evaluate with uninformative peers.** Include unreliable-channel
  regimes in a newly registered training mixture, then test new instances.
  The current evolved prior is harmed by the out-of-distribution no-social control.
- [ ] **Break cue calibration deliberately.** Change how well reliability cues
  predict peer quality, independently of average peer reliability. Measure whether
  selectivity follows useful evidence or an obsolete cue association.
- [ ] **Vary environmental persistence and shift frequency.** Test slow drift,
  abrupt shifts and shorter episodes without changing endpoints after inspection.
- [ ] **Measure recovery and failure together.** Keep full post-shift curves and
  right-censored recovery counts. Add a registered absolute performance target
  so poor pre-shift performance cannot make recovery appear artificially easy.

**Decision criterion:** useful social trust should improve learning where peers
help, while limiting the cost of misleading or uninformative peers. Specify the
acceptable cost before examining the new test results.

## 4. Intervene on actual transmission

- [ ] **Add a no-selection control.** Choose parents uniformly rather than by
  fitness while retaining mutation, birth initialization and learning budgets.
  This measures the contribution of selection to finding a useful prior.
- [ ] **Interrupt genome transmission explicitly.** Compare selected-parent
  offspring with newborns whose genomes are freshly sampled from a registered
  reference distribution, keeping initialization and evaluation noise paired.
  This removes selected information; it changes the genome distribution and
  must be interpreted accordingly.
- [ ] **Retire lineage permutation as a causal inheritance test.** Preserve it
  as an implementation negative control. With exchangeable newborns and no other
  inherited state, reassigning the same genomes cannot identify a special effect
  of lineage labels.

**Decision criterion:** identify exactly which information the intervention
removes. Distinguish selection, transmission of a useful prior, and continuity of
a named lineage; the current design does not make these interchangeable claims.

## 5. Expand only after the controls hold

- [ ] **Introduce endogenous peers.** Let demonstrators learn and adapt, then
  separate benefits of learner selectivity from changes in demonstrator quality.
- [ ] **Try a richer developmental genome.** Add dimensions or a decoder only
  alongside a capacity-matched fixed baseline and the genome-only birth invariant.
- [ ] **Study cultural transmission separately.** Define a social channel through
  which acquired information persists across generations, and intervene on that
  channel independently of genome transmission.
- [ ] **Define a cumulative-culture test before using the term.** Register a
  measure of improvement retained and extended across social transmissions,
  together with controls for individual learning, genetic selection and repeated
  rediscovery.

The criterion for progressing is stronger evidence about a specific mechanism,
not a larger simulation or a positive result alone.
