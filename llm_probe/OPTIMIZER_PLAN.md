# Optimizer at scale: does the update rule move factoredness in a real transformer?

Closes the biggest hole in the project: "optimizer geometry doesn't move
factoredness" is solid in the CPPN toy but only ASSERTED at transformer scale.
We test it directly, with the now-validated metric and a design that engages the
one paper that disagrees with us.

## The paper we're stress-testing (arXiv:2606.09658)

Claim: **Muon yields higher effective rank of hidden states than Adam/SGD**, and
this = more transferable/robust features. Effective rank is on our exact axis
(it's our validated factoredness metric). Key gap in their abstract: **no mention
of controlling for final loss** when comparing optimizers.

## The sharp, falsifiable contrast

Compare AdamW vs Muon vs plain SGD, **at MATCHED final validation loss**, measure
mid-layer effrank (validated on the width gate to track superposition correctly).

- If Muon shows higher effrank **even at matched loss** → they're right, and our
  "optimizer doesn't move factoredness" is WRONG at scale. We update.
- If the effrank gap **shrinks/vanishes once loss is matched** (i.e. Muon's edge
  was just "it optimizes better / reaches lower loss / different point on the same
  loss-effrank curve") → their effect is confounded with fit quality, and our
  claim stands: the update rule is a within-basin lever, not a factoredness lever.

This is the honest experiment: it can prove us wrong, and it isolates the exact
thing (matched-loss) their abstract doesn't mention.

## Design

- Model: same 30M arch as the objective experiment (d=384, 6 layers), fixed.
- Optimizers: **AdamW** (baseline), **Muon** (spectral/orthogonalized updates on
  2D matrices, AdamW on the rest — the standard Muon recipe), **SGD+momentum**.
- Matched loss: train all to the SAME target val loss. Two ways, do both:
  1. same steps, report the loss each reaches + effrank (loss-vs-effrank scatter);
  2. early-stop each optimizer when it hits a common loss threshold, compare
     effrank at that matched loss.
- Seeds: 3 each. Metric: mid-layer mlp_effrank + resid_effrank (validated), plus
  the width-gate sanity that effrank tracks superposition.
- Capability control: report ar_ce for all; only compare effrank among runs within
  a small ar_ce band.

## What would change our minds

- Muon effrank > AdamW effrank at matched loss, across seeds, robustly → we were
  wrong at scale; optimizer IS a factoredness lever; write that.
- No gap at matched loss → we were right; the toy generalizes; the Muon-features
  effect is a fit-quality / conditioning effect, not a superposition one.

## Then, to make it "not preliminary"

If the matched-loss contrast is clean, scale the ONE dimension that matters for
credibility: (a) a second model size (e.g. 100M) to show it's not a 30M artifact,
and (b) a second dataset. Same design. That turns "suggestive" into "holds across
scale and data".

## What we learned from reading the FULL paper (arXiv:2606.09658, HTML, not abstract)

Confirmed our design is well-aimed and found our differentiated contribution:
- **Their effrank == ours**: eRank(Z) = exp(-Σ qᵢ log qᵢ), qᵢ = σᵢ²/Σσ², on the
  hidden-state matrix (features stacked over examples). Same quantity. NOTE: they
  report RAW eRank (Adam 11.12, Muon 16.00 on GPT-2); we report eRank/width. To be
  directly comparable we should ALSO log raw eRank, not just the fraction.
- **They matched BUDGET, not final loss, empirically.** Their matched-*loss* claim
  is only THEORETICAL (1-layer, "first time each reaches loss ≤ ε", Thm 5.2/5.3).
  => **Our empirical matched-loss result is the novel piece** — the empirical
  version of what they only prove in a 1-layer toy. Keep matched-loss central.
- **Their main setting is GPT-2 124M (12 layer), scale-up GPT-2-Medium 354M, on
  FineWeb.** So our scale-up should target ~124M to be directly comparable to
  their headline, and ideally use a FineWeb slice as the "second dataset."
- **No public code** — build on our own stack.
- Their reported gap: GPT-2 Muon eRank 16.0 vs Adam 11.1 (~44% higher), depth-
  averaged. Our 30M matched-loss gap was +40% — strikingly consistent, and ours
  is the stronger claim because it's matched-loss not matched-budget.

## Scale-up design (this run)

- Model: ~124M GPT-2-like (d=768, 12 layers, 12 heads) — matches their main setting.
- Dataset: keep TinyStories for continuity + add a FineWeb/openwebtext-style slice
  as the 2nd dataset if loadable on the pod.
- Metric: log BOTH eRank/width (our fraction) AND raw eRank (their number), per
  layer + depth-averaged, so we can put our bar next to their Table 1.
- Matched loss: tune Muon/AdamW LR to the same val ar_ce; 3 seeds if time permits,
  else 2. Pull incrementally (pod-eviction lesson).
