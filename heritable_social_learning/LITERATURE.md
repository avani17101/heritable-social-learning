# Literature and conceptual boundaries

Checked against publisher pages and project repositories on 2026-09-19. This
prototype is original, small code; none of these neuroevolution frameworks is
imported. Neighboring results motivate the question rather than establish its answer.

1. **Ha, S. & Jeong, H. (2023).** *Social learning spontaneously emerges by
   searching optimal heuristics with deep reinforcement learning.* ICML,
   PMLR 202:12319–12338. [Publisher](https://proceedings.mlr.press/v202/ha23a.html).
   Optimizes social-learning heuristics in a cooperative setting. Relevant to
   balancing individual and social information; distinct from this genome-only
   transmission experiment and its fresh newborn networks.

2. **Bhoopchand, A. et al. (2023).** *Learning few-shot imitation as cultural
   transmission.* Nature Communications 14, 7536.
   [Publisher](https://www.nature.com/articles/s41467-023-42875-2).
   The supplied brief labels this “Pawar et al.”; Julia Pawar is a coauthor,
   but Bhoopchand is the first author. The paper studies few-shot imitation in
   embodied artificial agents. This prototype has neither embodied imitation
   nor cross-generation cultural transmission, so it does not reproduce that claim.

3. **Gruau, F. & Whitley, D. (1993).** *Adding Learning to the Cellular
   Development of Neural Networks: Evolution and the Baldwin Effect.*
   Evolutionary Computation 1(3):213–233.
   [DOI](https://doi.org/10.1162/evco.1993.1.3.213).
   Historical neighboring work on evolution, development and lifetime learning.
   The publisher DOI was inaccessible to the browsing tool; bibliographic metadata
   was cross-checked, but no unverified specific experimental claim is used here.

4. **Stanley, K. O., D’Ambrosio, D. B. & Gauci, J. (2009).** *A Hypercube-Based
   Encoding for Evolving Large-Scale Neural Networks.* Artificial Life
   15(2):185–212. [DOI](https://doi.org/10.1162/artl.2009.15.2.15202),
   [author manuscript](https://www.ncheney.com/teaching/robotics_readings/AHypercubeBasedIndirectEncodingForEvolvingLargeScaleNeuralNetworks%28StanleyDAmbrosioGauci2009%29.pdf).
   Indirect representations motivate possible later developmental encodings.
   The present genome controls learning coefficients, not neural weights/topology.

5. **Uber Research, PyTorch-NEAT.**
   [Repository](https://github.com/uber-research/PyTorch-NEAT).
   Implementation reference for neuroevolution; inspected as context only.

6. **ABrain / ES-HyperNEAT.** [Repository](https://github.com/kgd-al/abrain).
   An implementation of indirect neuroevolution using C++ computations and a
   Python interface. Context for possible later work; not a dependency here.

## Reuse audit

Numerical operations use NumPy, PyTorch, SciPy, Pandas/PyArrow and Matplotlib.
The custom code is confined to the noisy social task, evidence-target mixture,
selection/mutation orchestration and scientifically defined diagnostics. Exact
enumeration pins the environment; tests verify gradient direction, missing social
channels, no state leakage, clean births and population permutation invariance.
