# Anti-collapse optimizer: make factoredness an EXPLICIT objective (design principle #2)

We found Muon raises effective rank *implicitly* (depth-compounded spectral
updates). The design frontier: make it EXPLICIT — train with `loss − λ·richness`
so the optimizer prizes keeping the representation navigable, not just reaching
low loss. This is the inner-loop analog of novelty-search in weight/feature space.

## The circularity trap (read first)

If we reward effrank and then VALIDATE on effrank, we've proven nothing — of
course optimizing X raises X. So the design is:
- **Objective term**: a cheap differentiable richness surrogate R (below).
- **Validation MUST be a DIFFERENT, downstream property**: RECOMBINABILITY
  (novelty-vs-coherence frontier, from recombinability.py) — the thing
  open-endedness actually needs. Also report loss (must stay matched) and a
  DIFFERENT rank metric than the one optimized (e.g. optimize MLP-hidden rank,
  measure residual-stream rank + SAE monosemanticity) to check it's not just
  gaming one layer's statistic.
- **The real question**: does explicitly pushing richness buy MORE recombinability
  per unit loss than Muon's implicit version — or does it just inflate the metric
  without the downstream benefit? Either answer is informative.

## The richness regularizer R (differentiable, cheap)

Effective rank via SVD every step is expensive and the SVD-of-entropy gradient is
fiddly. Cheap differentiable surrogates that push the activation covariance toward
isotropic (= high effective rank):

  R_logdet = log det( C + εI )      where C = normalized activation covariance
             (maximizing log-det spreads variance across all directions; this is
             the classic "volume" / diversity term, differentiable, no SVD needed
             beyond a cholesky/logdet). Collapse → tiny eigenvalues → very negative
             logdet, so maximizing R_logdet fights collapse.

  R_offdiag = −|| offdiag(Corr) ||_F   (penalize feature correlation directly;
             cheapest; pushes toward decorrelated features = higher rank).

Start with R_offdiag (cheapest, most stable), offer R_logdet as the principled
version. Apply on the SAME mid-layer we measure elsewhere. Total objective:
    L_total = CE_loss − λ · R           (λ small; sweep λ ∈ {0, 1e-3, 1e-2, 1e-1})

## Arms (all AdamW base, matched-loss where possible)

- **adamw**            : baseline (λ=0).
- **adamw+anticollapse**: AdamW + λ·R, λ swept.
- **muon**             : the implicit-richness reference (our measured winner).
- (stretch) **muon+anticollapse**: does explicit stack on top of implicit?

Matched loss caveat: the regularizer changes the objective, so "final loss" is
the CE part only; we compare recombinability at matched CE by tuning steps/λ so CE
lands in the same band as the baselines.

## Predictions / decision tree

1. adamw+anticollapse raises effrank toward Muon AND raises recombinability at
   matched CE → **explicit anti-collapse works; we can engineer factoredness
   without Muon's specific geometry.** Strong result — a designed "open-endedness
   optimizer" ingredient.
2. It raises effrank but NOT recombinability → effrank was a red herring / gameable;
   the recombinability that matters comes from Muon's geometry specifically, not
   from rank per se. Important negative — refines the mechanism.
3. It hurts CE too much to match → the richness/loss tradeoff is real and steep;
   quantify the frontier (how much recombinability per unit CE sacrificed).

## Why this is the right "what do we do about it"

It directly tests whether the lever we found (richness/anti-collapse) is
CAUSAL for open-endedness-relevant recombinability, by installing it explicitly
and cheaply in ANY optimizer — decoupling "the property" from "Muon the method."
If it works, the open-endedness-aware optimizer is just: your optimizer of choice
+ an anti-collapse term, validated on recombinability not loss. That's a concrete,
buildable artifact, and the natural sequel to the recombinability result.

## Implementation notes

- add `--anticollapse {none,offdiag,logdet}` and `--acl_lambda` to train.py.
- compute R on the captured mid-layer activations already available via
  model.captured(); add −λ·R to loss before backward. Keep it cheap (subsample
  tokens for the covariance, like the metric does).
- log CE separately from total loss so matched-CE comparison is possible.
- validate with recombinability.py + effrank on a DIFFERENT layer (anti-circular).
