# Recombinability: does Muon's higher effrank actually buy open-endedness stepping stones?

The load-bearing unmeasured link. We have: Muon → higher effective rank (measured,
robust). We have ASSUMED: higher effrank → more recombinable/factored → better
stepping stones for an outer curiosity loop (Stanley's claim, never measured here).
This experiment tests that link directly, on the matched-loss AdamW vs Muon
checkpoints we already trained.

## What "recombinable stepping stone" means, operationally

Stanley: a factored representation gives clean, semantically-meaningful directions
you can perturb/recombine to get NEW, COHERENT artifacts (novelty search / QD
finds viable variations). A fractured one: perturbing yields garbage, so the outer
loop stalls. So recombinability = **when you move in representation space, do you
get outputs that are (a) DISTINCT from the original and each other [novelty], and
(b) still COHERENT/on-manifold [viability]?** Open-endedness needs BOTH: distinct
but broken = noise; coherent but identical = no exploration.

## The measurement (checkpoint-only, no retraining)

For each matched-loss checkpoint (adamw/muon, seeds 0-2):
1. Pick a set of probe prompts; run the model, capture a hidden layer's activation
   for the last token (the "representation" we recombine).
2. **Perturb / interpolate** in that representation space:
   - directional perturbation: add noise along random OR top principal directions,
     scaled to a fixed fraction of the activation norm;
   - interpolation: mix two prompts' representations at alpha in [0,1].
3. Patch the perturbed representation back in (activation patching at that layer)
   and free-run the model to produce output text.
4. Score the resulting outputs on the two axes:
   - **novelty/diversity**: distinct-ness of outputs (e.g. mean pairwise distance
     in an embedding, or distinct n-gram / type-token ratio across the set);
   - **coherence/viability**: is the output still well-formed? Proxy without human
     labels: the model's OWN likelihood of its output (low perplexity under the
     same model = on-manifold) or a simple fluency proxy (repetition rate, valid
     token fraction). Prefer perplexity under a *held-out* clean model if available.
5. The open-endedness-relevant quantity is the **Pareto frontier of
   novelty-vs-coherence**: a recombinable rep gives high novelty at matched
   coherence (or high coherence at matched novelty). Compare Muon vs AdamW frontiers.

## Predictions

- If the effrank→recombinability link is real: **Muon checkpoints give MORE
  distinct outputs at the SAME coherence** (frontier shifted up/right) — because
  more independent feature directions => perturbing one yields a coherent variation
  instead of breaking the whole output.
- Null / reversed: effrank does NOT buy recombinability in this setup → the
  open-endedness claim stays a story, and we say so. (This is the honest test.)

## Controls (so it's not confounded)

- **Matched loss** already holds (that's why we use these checkpoints).
- **Matched perturbation magnitude**: perturb to the same fraction of activation
  norm for both optimizers (Muon/Adam activations may differ in scale — normalize).
- **Same layer, same prompts, same seeds** for the perturbation noise.
- **Coherence baseline**: report unperturbed coherence too, so we measure the
  DROP in coherence per unit novelty gained, not absolute values.
- Confound to watch: higher effrank could just mean higher activation entropy that
  mechanically inflates "diversity" without real coherent novelty. The coherence
  axis is exactly what guards against this — diversity that comes with a coherence
  collapse is not recombinability.

## Scope / smallest-useful-version

Start minimal on the existing 30M matched-loss checkpoints (adamw/muon seed 0):
- 1 hidden layer (mid), ~50 probe prompts, random-direction perturbation at 3
  magnitudes, ~8 samples each.
- novelty = mean pairwise output distance (token-level); coherence = mean output
  perplexity under the SAME model (self-consistency proxy).
- If signal appears, expand to seeds 1-2 and principal-direction / interpolation
  perturbations + the 124M checkpoint.

## What this would let us claim

- Positive: "Muon's higher effective rank is not just a metric — at matched loss it
  yields representations that are measurably more recombinable (more distinct
  coherent variations under perturbation), i.e. better stepping stones for an outer
  open-ended loop." This is the real bridge from optimizer geometry to open-endedness.
- Negative: "effrank moves but recombinability does not — the open-endedness link
  is not supported in this setup," which is just as valuable to know.
