# Next experiment (deliberate): lock down the factoredness metric

Decision: **metric-first.** Every conclusion in this project has flipped when we
changed metrics (weight→neuron in the toy; the UFR gate exposed sparsity
confounds; the transformer proxy may be wrong too). Before any more
phenomenon-hunting we establish a metric we can trust, by forcing every candidate
through the same validation gauntlet and keeping only survivors.

## The problem, stated precisely

A factoredness metric must satisfy properties that our metrics have failed one by
one:
- **P1 monotone on ground truth.** Ranks a known-UFR net above a known-FER net.
  (toy weight-space metrics FAILED this — ranked evolved genome as more fractured.)
- **P2 capacity/width invariant.** Not driven by param count, sparsity, or width.
  (weight-space sensitivity FAILED; neuron-TV and effrank-frac were built to pass.)
- **P3 measurement-mode invariant.** Same verdict regardless of causal/bidir eval,
  batch, layer choice within reason. (exp2 dual-mode check tests this.)
- **P4 agreement.** Cheap proxy agrees with the expensive gold standard (SAE
  monosemanticity). If they disagree, at least one is wrong — investigate which.
- **P5 sensitivity.** Actually moves when the representation demonstrably changes
  (e.g. early vs late checkpoint), so it's not just measuring noise.

## Ground-truth pairs (the gauntlet)

We need KNOWN factored/fractured contrasts to validate against. Available:
1. **evolved Picbreeder genome vs FER-paper SGD genome** (the gold UFR/FER pair;
   image domain; already loaded). Any image-domain metric must pass this.
2. **synthetic superposition** (k latents in d dims, controllable) — we already
   used this to validate the transformer effrank proxy reads ~k/d. Gives a metric
   a *known* answer to reproduce.
3. **early vs late transformer checkpoint** (P5 sensitivity; from exp2 logs — we
   saw mlp_effrank move 0.03→0.35 over training, so the metric is at least alive).
4. **SAE monosemanticity on matched-loss arms** (exp2 is producing this now) — the
   gold standard the cheap proxy must agree with (P4).

## Candidate metrics to run through the gauntlet

Transformer-domain (the regime we now care about):
- **A. effrank_frac** (current proxy) — width-normalized correlation-matrix
  effective rank. Cheap. Validated on synthetic. Unknown: P4 vs SAE.
- **B. token participation ratio** (current diagnostic) — features per token.
- **C. SAE monosemanticity proxy** (exp2) — mono_proxy + dead_frac + fvu. Gold-ish
  but has its own confounds (SAE hyperparams, dict size, k).
- **D. NEW — feature interference / off-diagonal Gram** of SAE decoder columns:
  how non-orthogonal are the learned features? (superposition = overcomplete,
  interfering dictionary). Directly operationalizes "superposition".
- **E. NEW — probing-based**: can a linear probe read a known concept from few
  dimensions (factored) vs needing many (distributed)? Requires labeled concepts.

## The experiment

1. **Agreement matrix (P4).** exp2 gives SAE metrics for flat_ar/staged/
   staged_strong (seed 0). Compare arm-ranking under effrank_frac vs SAE
   mono_proxy vs decoder-interference. Do they agree on the ordering? Build the
   correlation across arms×layers.
2. **Gauntlet table.** For each candidate metric, mark pass/fail on P1–P5 using
   the ground-truth pairs above. Keep only metrics passing P1–P4.
3. **Pick the survivor**, re-run the objective-lever verdict (flat vs staged vs
   staged_strong) under the survivor metric, and THAT becomes the trustworthy
   answer to "does the objective move factoredness at transformer scale".

## What exp2 (running now) already contributes

- SAE mono_proxy / dead_frac / fvu for 3 causal arms → seeds the P4 agreement test.
- dual-mode effrank (causal vs native) → P3 for the proxy.
- staged_strong arm → whether a stronger lineage changes the verdict (feeds the
  final re-run under the survivor metric).

## Open design choices to settle when exp2 lands

- If SAE and proxy AGREE (both say "objective doesn't move it"): metric is trusted,
  the negative result is real → pivot to Reason-A (outer-loop search) with
  confidence.
- If they DISAGREE: the metric was the bottleneck → the disagreement itself is the
  finding; dig into which is right using the decoder-interference metric (D) as a
  tiebreaker, since it most directly measures superposition.
- Layer choice: validate the metric at several layers (early/mid/late), not just
  L4 — factoredness may be layer-dependent.
