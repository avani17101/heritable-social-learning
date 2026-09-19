# Research status

## State
Complete: working CPU PyTorch prototype, 21 passing tests, corrected frozen baseline,
20 independent seeds × 32 agents × 50 generations, nine PNG/PDF figures and
results/RESULTS.md. Analytical gate: .802083 combined Bayes accuracy vs .7 for
either source. Seed-zero transfer replay matches every saved metric exactly.
The baseline screening split is fixed and the entire 20-seed study has been
rerun. A separate attributed blog, root README, explanatory SVG/PNG diagram,
and prioritized future-research checklist are ready for the repository.

## Results vs pre-registration
H1 supported: evolved saves 32.684 regret actions vs individual-only (95% CI
31.979–33.343). H2 inconclusive: best-fixed minus evolved regret -0.365 (95% CI
-0.982–0.191); evolved mean 88.198 vs best-fixed 87.834 over 400 actions.
H4 supported: evolved saves 1.295 early-life regret actions vs random genomes
(95% CI 1.191–1.406). H1/H2/H4 verdicts use registered Bonferroni intervals.
H5 specialization supported exploratorily. H3 remains structurally inconclusive.
All 20 seeds retained. Final baseline and runs share one clean Git snapshot.

## Blockers and doubts
Lineage shuffling among exchangeable newborns preserves genetic transmission;
it cannot test the proposed lineage-causal claim. Alpha/beta absolute scale is
redundant under the normalized loss. The tuned fixed prior matches evolution
within uncertainty. In the no-social control, the evolved learner incurs about
68.253 more regret actions than individual-only: its inherited trust does not
robustly reject an uninformative channel outside its training distribution.
No evidence justifies moving to Stage 2 or claiming cumulative culture.

The initial fairness audit found that screening reused `tune_train` trajectories.
This is corrected using a separate `tune_screen` stream, guarded by a regression
test that fails on the original implementation. Legacy baseline freezes are now
rejected. The corrected screen selected the same prior (.2,.8,.2); all 20 rerun
seeds reproduce the original experimental metrics exactly. The original study is
preserved under results/archive_screen_reuse/. Validation and test were already
independent. Reusing the inspected test set makes this a correction, not a new
independent confirmation. Matching search effort, domain, replication and
selection objective remains future work in docs/RESEARCH_ROADMAP.md.

## Decisions needed
None to finish Stage 1. Future work: denser fixed search, Bayesian-history
reference, and a true randomized transmission intervention for a lineage claim.

## Change log
- 2026-09-19: Read supplied brief; registered analytical-first implementation.
- 2026-09-19: Verified noisy-cue analytical gate and created PyTorch prototype.
- 2026-09-19: Corrected one test's NumPy `.fill_` typo to `.fill`; no model change.
- 2026-09-19: Passed 19 tests; froze baseline (.2,.8,.2) using training/validation.
- 2026-09-19: Completed and analyzed all 20 final seeds with no exclusions.
- 2026-09-19: Verified exact independent replay of all seed-zero transfer outputs.
- 2026-09-19: Fairness audit identified baseline screening trajectory reuse;
  disclosed in the blog draft and marked the original comparison preliminary.
- 2026-09-19: Drafted blog/draft.md and blog/launch-post.md to introduce the idea;
  neither artifact has been published or sent externally.
- 2026-09-19: Fixed screening, added independent-stream and legacy-freeze tests,
  retuned the baseline and reran all 20 seeds. Verified exact metric agreement
  and transfer replay; 13,691,508 source rows passed provenance verification.
- 2026-09-19: Revised the blog using the user's own Substack posts as a writing
  reference, explicitly credited OpenAI Codex generation, and added an editorial
  diagram and prioritized research checklist. User authorized committing/pushing.
- 2026-09-19: Set repository identity to avani17101 and prepared existing main
  history with that author/committer; retained the original history in a local
  backup branch.
