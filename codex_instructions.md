# Codex Task: Build a Minimal PoC for Heritable Social-Learning Priors

Build a small, rigorous, reproducible research prototype testing the following hypothesis:

> **Can evolution discover a heritable learning prior that determines how newborn agents balance individual information and social information, such that this inherited prior improves adaptation across changing environments?**

The goal is **not** to demonstrate full cultural evolution yet. This is a first causal PoC for the narrower claim:

**genome → learning/developmental prior → lifetime learning → selection → next-generation genome**

The genome must be the **only information transmitted between generations**. Agents must NOT inherit neural-network weights, hidden states, memories, optimizer state, or experience.

Do not overclaim this as biological genes, culture, cumulative culture, or human-like evolution. Call it a **heritable genome / heritable social-learning strategy**.

---

# 1. Scientific motivation

Relevant literature establishes several neighboring ideas:

* Evolution can select biases that affect lifetime learning (Baldwin-effect / evolution-of-learning literature).
* Neuroevolution can use compact indirect representations to generate neural-network structure/weights (NEAT/HyperNEAT).
* Social-learning strategies can emerge through optimization, including balancing individual and social information.
* Artificial agents can exhibit cultural transmission through imitation.

This PoC asks a more specific question:

> If a population repeatedly faces learning problems in which both private information and social information are useful but noisy, can evolution discover a **heritable strategy for deciding how much to trust each information source**?

Relevant references to record in `LITERATURE.md`:

1. Ha & Jeong, 2023, *Social learning spontaneously emerges by searching optimal heuristics with deep reinforcement learning*, ICML/PMLR.
   https://proceedings.mlr.press/v202/ha23a.html

2. Pawar et al., 2023, *Learning few-shot imitation as cultural transmission*, Nature Communications.
   https://doi.org/10.1038/s41467-023-42875-2

3. Gruau & Whitley, 1993, *Adding Learning to the Cellular Development of Neural Networks: Evolution and the Baldwin Effect*.

4. Stanley et al., HyperNEAT / indirect encoding literature.

5. Uber Research PyTorch-NEAT:
   https://github.com/uber-research/PyTorch-NEAT

6. ABrain / ES-HyperNEAT:
   https://github.com/kgd-al/abrain

The implementation should be original and minimal rather than importing these frameworks.

---

# 2. Important conceptual distinction

The first prototype is NOT:

`genome → neural-network weights`

and it is NOT:

`parent weights → child weights`

Instead:

`genome → social-learning prior → individual learning`

For this PoC, use a deliberately tiny genome:

$$
G=(g_\alpha,g_\beta,g_\gamma)
$$

which decodes to:

$$
\alpha,\beta,\gamma=\sigma(G)
$$

where:

* `alpha`: strength of individual evidence
* `beta`: strength of social evidence
* `gamma`: trust/selectivity parameter for social evidence

This is intentionally simple.

Later work can replace this with a richer latent genome and a decoder that generates neural initializations or learning rules. Do NOT implement that complexity in this PoC.

---

# 3. Critical environment design

Do NOT use an environment where observing a successful peer directly reveals the hidden answer.

The earlier design

`reward = 1 iff both agents chose the hidden target`

combined with observing `(peer_action, peer_reward)` creates a potentially trivial social-learning signal.

Instead implement a **noisy hidden-state social-learning task**.

## Environment

Use:

* `K = 4` possible latent states/actions.
* Each episode samples a hidden state:

$$
z\in\{0,1,2,3\}
$$

uniformly.

Each agent receives a private noisy signal about `z`.

For example:

```text
P(private_signal = z) = p_private
P(private_signal = another state) = 1-p_private
```

with:

```text
p_private = 0.7
```

The peer independently receives another noisy signal.

The agents then choose actions.

The important property is:

> Neither the agent's own signal nor the peer's signal is perfectly reliable.

Social information should therefore be useful but not always correct.

---

# 4. Make social information genuinely informative but non-trivial

The peer should provide **evidence**, not an oracle.

Do NOT give the learner direct access to:

```text
peer_action + peer_reward
```

if that combination reveals the latent state.

Instead expose a compact noisy social observation such as:

```text
peer_signal
peer_confidence / reliability cue
```

or, preferably for the first implementation:

```text
peer_action
peer_observed_outcome
```

where the outcome is itself noisy and does NOT uniquely reveal `z`.

Document exactly why the resulting observation is not sufficient to identify `z`.

The environment should satisfy:

1. private information has positive value;
2. social information has positive value;
3. blindly copying the peer is suboptimal;
4. ignoring the peer is also suboptimal;
5. the optimal weighting of private vs. social information changes when information reliability changes.

Add an assertion/test demonstrating these properties empirically.

---

# 5. Episode structure

Use a small repeated interaction rather than a single isolated timestep.

For example:

```text
episode length = 10
```

At each timestep:

1. latent state `z` exists;
2. each agent receives a noisy private observation;
3. agents observe limited social information from the previous timestep;
4. agents select actions;
5. agents receive noisy reward/evidence;
6. agents update their policy.

This makes social learning meaningful while remaining extremely small.

Do not use PPO, MAPPO, SMAC, Waterworld, LLMs, transformers, or large MARL frameworks.

Use a tiny PyTorch policy and a simple learning rule.

---

# 6. Agent learning

Each newborn receives:

* a random neural network initialization;
* its inherited genome `G`.

The neural-network weights are freshly initialized every generation.

The genome determines how the agent updates from:

### Individual evidence

$$
L_{\text{individual}}
$$

### Social evidence

$$
L_{\text{social}}
$$

Use:

$$
L =
\alpha L_{\text{individual}}
+
\beta w_{\text{social}}L_{\text{social}}
$$

where:

$$
w_{\text{social}}=f(\gamma,\text{peer reliability})
$$

The precise implementation should be chosen so that `gamma` genuinely controls trust/selectivity rather than being an arbitrary multiplier.

Prefer a principled Bayesian/evidence-weighting interpretation where possible.

Document the mathematical interpretation in `DESIGN.md`.

---

# 7. Avoid a fake genome effect

A major requirement:

The genome must affect **learning behavior**, not simply provide a reward multiplier.

Bad:

```python
loss = alpha * loss
```

if this merely changes the effective learning rate.

Better:

```text
alpha controls reliance on private evidence
beta controls reliance on social evidence
gamma controls the conditions under which social evidence is trusted
```

The resulting behavior should be measurable.

For example, after training, quantify:

* probability of following private evidence;
* probability of following peer evidence;
* probability of rejecting unreliable peer evidence.

Add behavioral diagnostics that estimate these quantities from controlled probe episodes.

---

# 8. Population and evolution

Use:

```text
population_size = 32
elite_fraction = 0.25
generations = 50–100
```

Each genome corresponds to a newborn agent.

For every generation:

### Step 1 — Birth

Create newborn agents.

For every newborn:

```text
new random neural-network weights
inherited genome
empty memory
empty optimizer state
```

### Step 2 — Lifetime learning

Agents interact for a fixed number of episodes.

### Step 3 — Evaluation

Evaluate agents on fresh episodes.

Fitness should be based on **held-out evaluation episodes**, not the same trajectories used for learning.

### Step 4 — Selection

Select the highest-fitness genomes.

### Step 5 — Reproduction

Create children by copying elite genomes and adding Gaussian mutation:

$$
G_{child}=G_{parent}+\epsilon
$$

with:

$$
\epsilon\sim N(0,\sigma^2)
$$

Use configurable mutation scale.

### Step 6 — Destroy the parents

Do not carry neural weights forward.

This is essential.

---

# 9. Experimental conditions

Implement at least these five conditions.

## A. Individual-only

Set:

```text
beta = 0
```

Agents cannot use social learning.

This tests whether social information provides any advantage.

---

## B. Fixed social-learning strategy

Use one fixed:

```text
(alpha, beta, gamma)
```

strategy.

This establishes a simple non-evolving social learner.

---

## C. Best fixed strategy

Do NOT merely hand-pick this.

Perform a small grid/random search over fixed `(alpha,beta,gamma)` values on the **training environments**.

Choose the best fixed strategy.

Then freeze it.

Evaluate it on independent test environments/seeds.

This is the strongest important baseline.

The evolved system must beat this baseline to support the stronger claim.

---

## D. Evolved genome

Allow `(alpha,beta,gamma)` to evolve.

This is the primary condition.

---

## E. Lineage-shuffled control

This is critical.

Run the same evolutionary process as D, but destroy parent-offspring correspondence.

After selection/reproduction, randomly shuffle genome assignments among newborn agents.

Preserve:

* same genome distribution;
* same population size;
* same mutation;
* same training budget.

Destroy:

* lineage continuity.

This tests whether the benefit depends on actual inheritance rather than merely discovering a good population distribution.

---

# 10. Add a frozen-genome transfer test

After evolution converges:

1. take the best evolved genome `G*`;
2. create newborn agents with fresh random neural networks;
3. compare them against newborn agents with random genomes;
4. give both identical lifetime learning budgets.

Measure performance during early learning.

This directly tests:

> Does the inherited genome give a newborn a useful learning bias before it has accumulated experience?

Do not interpret this as inherited knowledge. It is inherited **learning bias**.

---

# 11. Environment generalization

Do not evaluate only on the exact environment used for evolution.

Create separate training and test distributions.

For example:

### Training

```text
private reliability ∈ {0.65, 0.75}
social reliability ∈ {0.65, 0.75}
```

### Test

```text
private reliability ∈ {0.60, 0.70, 0.80}
social reliability ∈ {0.60, 0.70, 0.80}
```

Include combinations not seen during evolution.

The important question is whether evolution discovers a generally useful learning strategy rather than memorizing one environment.

---

# 12. Environment shifts

Include a second evaluation regime where information reliability changes.

For example:

```text
phase 1:
private information reliable
social information moderately reliable

phase 2:
private information less reliable
social information more reliable
```

The agents are not told explicitly that the phase changed.

Measure adaptation after the shift.

This is particularly important because a hereditary learning prior is interesting if it helps newborns/adults adapt to changing information regimes.

---

# 13. Primary hypotheses

Pre-register these in `DESIGN.md`.

### H1 — Social-learning advantage

A social learner should outperform the individual-only baseline when peer information is sufficiently reliable.

### H2 — Evolution of learning strategy

The evolved condition should outperform the strongest fixed strategy on some held-out environments or adaptation regimes.

Do NOT assume this will happen.

This is the important empirical question.

### H3 — Heritable advantage

The evolved lineage condition should outperform the lineage-shuffled condition if the benefit genuinely depends on intergenerational inheritance.

### H4 — Newborn transfer

A newborn carrying the evolved genome should learn faster than a newborn carrying a random genome despite having identical randomly initialized neural weights.

### H5 — Environmental specialization/generalization

Evolution should alter `(alpha,beta,gamma)` as the distribution of private/social reliability changes.

---

# 14. Primary metrics

Use these as the main metrics.

## 1. Cumulative regret

$$
R_T=\sum_{t=1}^{T}(r^*_t-r_t)
$$

where `r*` is the oracle reward.

This should be the primary metric.

## 2. Early-life learning

Performance during the first:

```text
10%, 25%, 50%
```

of the lifetime learning budget.

This tests whether the inherited prior actually helps learning.

## 3. Adaptation lag

After an environment shift:

```text
number of episodes required to recover 90% of pre-shift performance
```

## 4. Final lifetime performance

Performance after the full learning budget.

## 5. Newborn performance

Performance of agents immediately after birth / before substantial learning.

## 6. Behavioral social-learning diagnostics

Estimate:

```text
social reliance
private reliance
peer-trust selectivity
```

from controlled probe trials.

---

# 15. Evolutionary diagnostics

Log the genome every generation.

Plot:

```text
alpha vs generation
beta vs generation
gamma vs generation
```

Also plot their population distributions.

Important question:

> Does evolution actually discover social learning?

For example, if `beta → 0` consistently, that is a scientifically meaningful result.

Do NOT modify the environment until you obtain the desired result.

---

# 16. Statistical methodology

The independent experimental unit should be the **evolutionary run**, not individual agents.

Use at least:

```text
20 independent seeds
```

for the final experiment.

Use:

```text
3–5 seeds
```

for debugging.

For every seed, evaluate every condition on the same evaluation task seeds where appropriate.

Report:

* mean;
* median;
* standard deviation;
* 95% bootstrap confidence interval;
* effect size.

Use paired comparisons where the same environment/evaluation seeds are shared.

Do not run a large collection of statistical tests and only report significant ones.

Define the primary metric before running the final experiment.

---

# 17. Critical controls

Implement these sanity checks.

## Control 1 — Random genome

Randomly generated genomes should not systematically outperform the evolved genome.

## Control 2 — Genome ablation

Set:

```text
beta = 0
```

for an evolved genome.

This tests whether the discovered social component matters.

## Control 3 — Genome scrambling

Take an evolved genome and randomly permute its three dimensions.

Performance should change if the dimensions encode meaningful roles.

## Control 4 — No-social environment

If peer information contains no useful information, social learning should not systematically help.

## Control 5 — Perfect-social environment

If social information becomes highly reliable, social learning should become more valuable.

These controls demonstrate that the implementation actually measures the intended phenomenon.

---

# 18. Do not use weight inheritance

For the main experiment:

```text
parent network weights → child network weights
```

must NEVER happen.

Also do not inherit:

```text
optimizer state
replay buffer
hidden state
episode memory
training history
```

The only inherited object is:

```text
genome G
```

This is essential to the scientific interpretation.

---

# 19. Code structure

Create:

```text
heritable_social_learning/
│
├── README.md
├── DESIGN.md
├── LITERATURE.md
├── requirements.txt
├── config.yaml
│
├── src/
│   ├── environment.py
│   ├── agent.py
│   ├── genome.py
│   ├── learning_rule.py
│   ├── evolution.py
│   ├── experiment.py
│   ├── evaluation.py
│   ├── statistics.py
│   └── utils.py
│
├── scripts/
│   ├── smoke_test.py
│   ├── tune_baseline.py
│   ├── run_experiment.py
│   ├── run_transfer.py
│   └── analyze_results.py
│
├── configs/
│   ├── debug.yaml
│   └── final.yaml
│
├── results/
│
└── figures/
```

Keep the implementation small and readable.

Do not add unnecessary abstractions.

---

# 20. Reproducibility

Every experiment must accept:

```bash
--seed
--config
--condition
```

Example:

```bash
python scripts/run_experiment.py \
    --condition evolved \
    --seed 0 \
    --config configs/debug.yaml
```

Use deterministic seeds where practical.

Save:

```text
config
seed
genomes
fitness
episode rewards
evaluation rewards
environment parameters
model hyperparameters
```

for every run.

---

# 21. Smoke tests

Before running experiments, implement tests confirming:

### Test 1

Environment produces the correct hidden state distribution.

### Test 2

Private evidence has the expected reliability.

### Test 3

Social evidence has the expected reliability.

### Test 4

No information channel is an oracle.

### Test 5

Setting `beta=0` eliminates social learning.

### Test 6

Changing `gamma` changes social selectivity.

### Test 7

Children have different neural weights from parents.

### Test 8

Children inherit genome but nothing else.

### Test 9

Lineage shuffling actually destroys parent-offspring correspondence.

### Test 10

Same seed produces reproducible results.

---

# 22. Baseline tuning protocol

This is important.

Do not tune the fixed baseline on the test environments.

Split:

```text
training environments
validation environments
test environments
```

Use training/validation to select the best fixed strategy.

Freeze it.

Only then evaluate all final conditions on test environments.

Otherwise the comparison against the fixed baseline is weak.

---

# 23. Analysis plots

Produce at minimum:

### Figure 1

Fitness vs generation.

### Figure 2

Cumulative regret vs lifetime episodes.

### Figure 3

Post-shift adaptation curves.

### Figure 4

Evolved alpha/beta/gamma vs generation.

### Figure 5

Genome distribution at beginning vs end.

### Figure 6

Evolved vs shuffled lineage.

### Figure 7

Newborn transfer: evolved genome vs random genome.

### Figure 8

Generalization across private/social reliability.

All plots should show uncertainty across independent evolutionary seeds.

---

# 24. Final analysis

Generate:

```text
results/RESULTS.md
```

with:

1. Experimental question
2. Exact environment
3. Conditions
4. Hyperparameters
5. Number of seeds
6. Primary metric
7. Results
8. Statistical analysis
9. Genome evolution
10. Transfer results
11. Lineage-shuffle results
12. Failure modes
13. Interpretation
14. Next experiment

At the end explicitly classify the result as:

```text
SUPPORTED
NOT SUPPORTED
INCONCLUSIVE
```

for each hypothesis.

Do not manipulate the experiment to force support.

---

# 25. Interpretation rules

Be extremely careful about claims.

If evolved > fixed:

You may conclude:

> Evolution discovered a social-learning parameterization that outperformed the tested fixed strategies under the tested distribution.

Do NOT conclude:

> Evolution discovered human-like intelligence.

If evolved > shuffled:

You may conclude:

> The performance advantage depends on preserving genome-to-offspring lineage information under this experimental setup.

Do NOT conclude:

> This proves genetic inheritance causes culture.

If newborn evolved-genome agents learn faster:

You may conclude:

> The genome provides a useful inherited learning bias.

Do NOT call it inherited knowledge.

If beta evolves toward zero:

That is a legitimate result. It means social learning was not favored under those conditions.

---

# 26. Important conceptual limitation

The three-dimensional genome is deliberately simple.

This experiment therefore demonstrates:

> **evolution of a heritable social-learning strategy**

not yet:

> **evolution of a rich developmental genome**

and definitely not yet:

> **cumulative culture**

Treat those as later stages.

---

# 27. Follow-up if the PoC works

Only after the above experiment is convincing, build Stage 2.

Replace:

```text
G = [alpha,beta,gamma]
```

with:

```text
G ∈ R^d
```

and introduce a decoder:

```text
genome → developmental/social-learning parameters
```

Potentially:

```text
genome
   ↓
developmental decoder
   ↓
initial neural parameters
   +
learning-rule parameters
   +
social-learning biases
```

This is where indirect encodings / HyperNEAT-like approaches become relevant.

The longer-term architecture is:

```text
             genome
                ↓
       developmental prior
                ↓
          newborn agent
                ↓
       individual learning
                ↓
         social learning
                ↓
            behavior
                ↓
            fitness
                ↓
            selection
                ↓
       next-generation genome
```

Later, introduce explicit cultural transmission:

```text
genetic inheritance
        +
cultural inheritance
        +
individual learning
```

and eventually test genuine cumulative culture / ratcheting.

Do not implement those extensions now.

---

# 28. Most important implementation principle

Before writing the full experiment, first implement a **tiny analytical version** of the environment and demonstrate:

```text
private-only < optimal private+social
blind-copy < optimal private+social
```

under at least one reliability regime.

Then demonstrate that the optimal social/private weighting changes when reliability changes.

Only after that should you add neural networks and evolution.

This prevents the evolutionary experiment from becoming an elaborate experiment around a broken environment.

---

# 29. Deliverable

When finished, provide:

```text
1. Working code
2. Unit tests
3. README.md
4. DESIGN.md
5. LITERATURE.md
6. Configurations
7. Smoke-test output
8. Baseline-tuning output
9. Final experimental results
10. Figures
11. RESULTS.md
```

The code should run on CPU.

GPU support is optional but should work automatically if CUDA is available.

Keep the initial implementation small enough that the entire experiment can be understood by reading the repository.

Most importantly:

**Do not optimize for obtaining a positive result. Optimize for making the experiment capable of falsifying the hypothesis.**
