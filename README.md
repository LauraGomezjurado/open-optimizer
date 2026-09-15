# open-optimizer

Does the choice of optimizer change what a network learns, or only how fast it
gets there?

Answer from this repo: at transformer scale it changes what the network learns.
Muon and Aurora reach the same loss as AdamW with a measurably different internal
representation. The effect is large, repeats across seeds, and grows with depth.

This file collects the results that held up. The full lab log, including every
dead end in the order we hit it, is in [NOTEBOOK.md](NOTEBOOK.md).

## Setup

A 30M parameter GPT trained on TinyStories on one H100. Each optimizer gets its
own tuned learning rate so comparisons happen at equal loss.

The main readout is **effective rank**: how many independent directions the
activations actually use, as a fraction of the model width. High means the
network spreads information across many directions. Low means it packs
information into a few directions that overlap.

Transformer code is in [llm_probe/](llm_probe/). The small image-fitting model
that started the project is in [src/](src/) and [experiments/](experiments/).

## Same loss, different representation

Three seeds each, 4000 steps, learning rates tuned so both optimizers land on
the same loss.

![Effective rank against next-token loss. Muon and Aurora sit near 0.49 to 0.50
while AdamW sits near 0.35, with all runs inside a loss range of
0.05.](results/fig_matched_loss.png)

| optimizer | next-token loss | MLP effective rank | residual effective rank |
|---|---|---|---|
| AdamW | 2.746 ± 0.015 | 0.349 ± 0.006 | 0.368 ± 0.003 |
| Muon | 2.709 ± 0.003 | **0.490 ± 0.007** | **0.478 ± 0.011** |

Muon uses 40% more independent directions at the same loss, on 3 of 3 seeds,
with error bars far smaller than the gap. Matching the loss is what makes this
informative. It rules out the reading that Muon simply optimizes better.

At 2000 steps the gap is wider (AdamW 0.218 ± 0.005, Muon 0.481 ± 0.003), and
there Muon also reaches lower loss (2.749 vs 2.897).

At 124M parameters (12 layers, width 768) the effect holds: Muon 0.499 vs
AdamW 0.420 at losses of 2.81 and 2.87.

Source: `llm_probe/results/flat_ar_seed*_optc.json`,
`llm_probe/results_pod/fair/`.

## Aurora reaches the same place as Muon

Aurora adds a second constraint to Muon's orthogonal update: every row of the
update keeps the same norm, which stops MLP neurons from going dead. Seed 0, the
same 4000-step setting:

| optimizer | next-token loss | MLP effective rank |
|---|---|---|
| AdamW | 2.730 | 0.342 |
| Muon | 2.713 | 0.495 |
| Aurora | 2.721 | 0.496 |

Aurora lands on Muon's number. In the shorter 2000-step run it goes slightly
past Muon on both axes (loss 2.725 vs 2.753, effective rank 0.523 vs 0.499).
One seed, so the ordering between Muon and Aurora is still open. The solid part
is that the whole spectral family sits well above AdamW.

Aurora is also the only design in this repo that improved on Muon at all, which
is why it heads the next-steps list. Implementation notes are in
[llm_probe/AURORA_NOTES.md](llm_probe/AURORA_NOTES.md).

## The gap comes from depth

Scanning depth at fixed width 384, and width at fixed depth 6. One seed per
point, 2500 steps, and one fixed learning rate per optimizer rather than
per-point tuning. Muon therefore reaches lower loss at every scan point, by 0.05
to 0.14. Read these as trends rather than as matched-loss comparisons. The
matched-loss control is the 3-seed table above.

![Two panels. Effective rank against depth, where Muon climbs and AdamW stays
flat. Effective rank against width, where AdamW climbs and Muon stays
flat.](results/fig_mechanism.png)

| layers | AdamW | Muon | gap |
|---|---|---|---|
| 2 | 0.246 | 0.394 | +0.148 |
| 4 | 0.249 | 0.447 | +0.198 |
| 8 | 0.269 | 0.514 | +0.245 |
| 12 | 0.248 | 0.542 | +0.293 |

AdamW sits flat near 0.25 however deep the network gets. Muon climbs from 0.39
to 0.54. The orthogonal update compounds layer over layer.

The depth trend survives the learning-rate caveat. AdamW's loss improves across
this range (2.93 to 2.82) while its effective rank does not budge, and the loss
gap between the two optimizers shows no trend with depth (0.087, 0.090, 0.066,
0.116) while the effective rank gap grows monotonically. So the growing gap is
not the loss gap in disguise.

| width | AdamW | Muon | gap |
|---|---|---|---|
| 256 | 0.190 | 0.505 | +0.316 |
| 384 | 0.253 | 0.485 | +0.233 |
| 512 | 0.302 | 0.468 | +0.166 |

Width closes the gap by pulling AdamW up. Muon is near its ceiling at every
width. Plain reading: extra width is how AdamW buys independent directions,
extra depth is how Muon buys them.

The width panel is the weaker of the two. AdamW's loss gap to Muon also shrinks
across this range (0.145 to 0.046), so some of the catching up on effective rank
could be AdamW simply getting closer in loss. Confirming it needs per-point
learning-rate tuning, which we did not run.

Both slopes are monotone across all their points. The depth one is the prediction
worth acting on: the effect should get larger in deeper models.

This also explains the null result in the small image model at the bottom of
this file. That network was too shallow for anything to accumulate.

## Every layer shows it

Per-layer effective rank, seed 0, matched loss:

| layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| AdamW | 0.228 | 0.398 | 0.428 | 0.443 | 0.467 | 0.367 |
| Muon | 0.323 | 0.537 | 0.617 | 0.637 | 0.616 | 0.514 |
| Aurora | 0.349 | 0.578 | 0.603 | 0.623 | 0.597 | 0.508 |

The weight matrices agree. Their effective rank runs 0.90 to 0.94 under AdamW
and 0.94 to 0.96 under Muon and Aurora.

Source: `llm_probe/results/mechanism_depth.json`.

## The learning rate has to be tuned per optimizer

Run Muon at AdamW's learning rate of 3e-4 and it loses on both axes: loss 3.059
vs 2.897, effective rank 0.084 vs 0.218. Muon's tuned rate here is 0.005 to
0.02, more than ten times higher.

A comparison at a shared learning rate measures the tuning rather than the
optimizer. This caught us once. An early version of the recombinability
experiment below ran Muon at AdamW's rate and had to be redone.

## A small effect on recombinability

The reason to care about high effective rank is a guess: a representation using
more independent directions should survive being perturbed, giving new coherent
outputs instead of broken ones. That is what an outer search loop needs. We
measured it.

Patch a mid-layer activation at a set fraction of its norm, let the model
free-run, then score how different the outputs are from each other (novelty) and
how plausible they are to a separate pretrained GPT-2 (coherence). Three seeds
each.

| optimizer | novelty | judge perplexity |
|---|---|---|
| AdamW | 0.832 ± 0.010 | 19.1 ± 0.2 |
| Muon | 0.874 ± 0.002 | 26.1 ± 1.7 |

Muon produces more varied output at the same perturbation size. It also
produces less plausible output, so most of that raw gap is movement along the
coherence axis. Comparing at equal coherence instead:

- matched on judge perplexity: about +0.008 novelty
- matched on output entropy: about +0.030 novelty

Both are positive and both are small, roughly 1% to 4% of the 0.83 baseline.
An earlier version of this repo reported an effect around twenty times larger.
That number came from comparing at unmatched coherence and is not correct.

Honest summary: the representation difference is large, the payoff measured this
way is small. Effective rank is the result to trust here.

Caveat: the two perplexity ranges barely overlap, so the perplexity-matched
number rests on interpolation at a single point. The entropy-matched number has
more overlap and is the sturdier of the two.

Source: `llm_probe/results_pod/fair/neutral_frontier_fair.json`.

## What the small image model says

The project started smaller: fit one fixed image with a small network, vary only
the optimizer, measure how brittle the result is. Five paired seeds, three
target images, every run reaching the same near-zero loss.

Here the lever is per-coordinate scaling. Adam divides each weight's update by
that weight's own running gradient size. Removing that single feature accounts
for the whole effect.

| optimizer | brittleness (lower is better) |
|---|---|
| Adam | 7.94 |
| SignSGD | 8.08 |
| Adam, per-coordinate scaling at half strength | 6.63 |
| Adam with one scalar step size per matrix | 6.31 |
| normalized gradient descent | **3.81** |
| Muon | 5.14 |

Three things hold across all 5 seeds and all 3 images:

- The dose response is clean. Turning per-coordinate scaling down lowers
  brittleness monotonically (rank correlation +0.96, +0.88, +0.96 on the three
  images, 5 of 5 seeds each).
- Adaptivity is fine. One scalar step size per matrix is adaptive and sits with
  the good group. Per-entry scaling is what hurts.
- Orthogonalization is not needed for this. Plain normalized gradient descent is
  the least brittle rung, and Muon slightly overshoots it.

We checked the obvious confound, units saturating at their activation limits.
Regressing saturation out leaves the gap unchanged (4.01 becomes 4.25). On one
target the two point in opposite directions, which separates them cleanly.

This is a different measurement from effective rank on a much smaller network,
so it does not have to agree with the transformer result, and it does not.

Source: `results/stats.txt`.

## Which measurements to trust

Two of the three factoredness measures we built do not survive checking. Worth
knowing before reusing any of them.

**Effective rank passes.** The test: two models, same data and objective, only
width differs. Narrow (width 128) is forced to pack features together, wide
(width 512) has room to spread. Effective rank calls it correctly, 0.119 narrow
vs 0.407 wide.

**Kurtosis fails.** We expected packed features to be spiky, so high kurtosis.
The opposite came out, 25.5 for wide against 3.1 for narrow. Reading it now:
high kurtosis marks many selective, rarely firing features, which is what a
spread-out representation looks like. The sign is flipped from the hypothesis.

**Neuron smoothness passes the evolved-versus-SGD test.** Render each hidden
unit of an evolved Picbreeder network and of an SGD network fit to the same
image. The evolved units are smoother (total variation 1.5x to 1.8x lower) and
much lower frequency (up to 136x less high-frequency energy on one target), on
3 of 3 targets. Weight-level measures get this backwards because the evolved
network is about 3% dense and the SGD one is 100% dense, so anything measured
per weight reads sparsity instead of structure.

Source: `llm_probe/validate_metric.py`, `experiments/run_ufr_ceiling.py`.

## What did not work

Five attempts to reach Muon's representation without Muon's update, all scored
on the measurements above.

**Editing the weight spectrum after training.** Take an AdamW checkpoint,
replace each MLP weight's singular values S with S^p, keep the directions and
the total norm, do not retrain. Every edit raises loss, and at equal coherence
the novelty change is zero or negative across 3 seeds (-0.008, -0.025, -0.041).
The flat spectrum is something the training path builds up. It does not
transplant.

**A decorrelation penalty.** AdamW trained on loss minus lambda times an
off-diagonal correlation term. Effective rank moved from 0.349 to 0.36 and
recombinability stayed on the AdamW baseline.

**A higher-order independence penalty.** FastICA negentropy plus a fourth-order
cross-cumulant term. It works on the statistic it targets, lowering effective
rank monotonically in lambda at matched loss (0.218 baseline, then 0.177, 0.171,
0.166 across three lambdas, 3 seeds). It moves effective rank in the wrong
direction for our purposes and does not raise recombinability. A gain on seed 0
flipped sign on seed 1.

**An optimizer preconditioned by the input covariance.** Ten configurations.
Every one lands at worse loss and lower effective rank than Muon at the same
step count. The best reached loss 2.853 and effective rank 0.474 against Muon at
2.753 and 0.499.

**Changing the objective instead of the optimizer.** Masked-denoising warmup
followed by autoregressive training, and masked diffusion throughout. Both give
lower effective rank than plain autoregressive at matched loss.

One more null worth recording: a gradient-agreement criterion does not track
factoredness in the image model. The fractured SGD network has the highest
cross-region gradient agreement, which is the opposite of what the analogy
predicts.

## Next directions

1. **Finish the Aurora comparison.** It matched Muon on one seed and edged past
   it in the shorter run. Three seeds at 4000 steps plus a depth scan is cheap
   and settles whether the extra constraint helps. If it does, Aurora's recipe
   for bolting a constraint onto Muon is the thing to reuse.
2. **Go deeper.** The depth slope is the strongest mechanism evidence in the repo
   and it is one seed per point. Repeat at three seeds and push past 12 layers,
   where the prediction is a larger gap.
3. **Find a downstream payoff people care about.** Recombinability gave a small
   one. Transfer, robustness and continual learning are untested here. A
   concurrent honors thesis (Xu, UT Austin, May 2026) found that networks trained
   toward smoother activations stay plastic under continual learning where
   baselines freeze. Plasticity is a concrete test we have not run on Muon or
   Aurora.
4. **Explain the per-token sparsity difference.** In the 4000-step matched-loss
   runs, Muon activates a much smaller fraction of MLP hidden units per token
   than AdamW (0.109 ± 0.001 vs 0.541 ± 0.008, 3 of 3 seeds) while spanning more
   directions overall. Sparse selective features plus a wide population code is a
   coherent story and would explain the kurtosis result above. In the 2000-step
   runs at a lower learning rate the difference vanishes (0.563 vs 0.571), so it
   is a lead rather than a result.
5. **Explain why effective rank rises during training.** Within an AdamW run it
   goes from about 0.03 early to 0.35 late. The naive expectation was collapse.
   Unexplained, and it matters for reading the metric.

## Running things

Transformer experiments, from `llm_probe/`:

```bash
# the matched-loss optimizer contrast (3 seeds)
python train.py --arm flat_ar --optimizer adamw --lr 0.0003 --steps 4000 --seed 0 --tag optc
python train.py --arm flat_ar --optimizer muon  --lr 0.02   --steps 4000 --seed 0 --tag optc
python train.py --arm flat_ar --optimizer aurora --lr 0.02  --steps 4000 --seed 0 --tag optc

# depth and width scans
python train.py --arm flat_ar --optimizer muon --n_layer 12 --lr 0.02 --steps 2000
python train.py --arm flat_ar --optimizer muon --d_model 512 --lr 0.02 --steps 2000

# recombinability, scored by a separate pretrained GPT-2
python neutral_frontier.py --ckpts results/ckpt/*.pt --layer 3

# the width-contrast check on the metric itself
python validate_metric.py --narrow results/ckpt/narrow.pt --wide results/ckpt/wide.pt
```

Runner scripts for each experiment are the `llm_probe/run_*.sh` files.
`llm_probe/deploy_to_pod.sh` copies the directory to a GPU pod and installs
dependencies.

Image model experiments, from the repo root:

```bash
pip install numpy matplotlib pillow
python experiments/run_multiseed.py --all            # 13 optimizers, 5 seeds
python experiments/run_multiseed.py --target apple --all
python experiments/viz_qualitative.py --target skull # renders each hidden unit
```

The two figures above are regenerated from the raw result files by
`python experiments/plot_headline.py`, so they cannot drift from the numbers in
the tables.

## Where the numbers live

| result | files |
|---|---|
| matched-loss optimizer contrast | `llm_probe/results/flat_ar_seed*_optc.json` |
| tuned-learning-rate runs, 3 seeds | `llm_probe/results_pod/fair/` |
| per-layer effective rank | `llm_probe/results/mechanism_depth.json` |
| recombinability, neutral judge | `llm_probe/results_pod/fair/neutral_frontier_fair.json` |
| weight-spectrum editing | `llm_probe/results_pod/surgery/` |
| higher-order independence penalty | `llm_probe/results_pod/ho/` |
| image model, all optimizers | `results/stats.txt`, `results/multiseed*.json` |
| rendered hidden units | `results/genome_neurons_*.png` |

## References

- Kumar, Clune, Lehman, Stanley. *Questioning Representational Optimism in Deep
  Learning: The Fractured Entangled Representation Hypothesis.* arXiv:2505.11581 (2025).
- *Muon Learns More Robust and Transferable Features than Adam.* arXiv:2606.09658 (2026).
  Our matched-loss result confirms and tightens theirs.
- *Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data.* arXiv:2502.04664 (2025).
- *The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks.* arXiv:2602.16340 (2026).
- Aurora optimizer, Tilde Research. `github.com/tilde-research/aurora-release`.
- Ahn, Xu et al. *Dion: Distributed Orthonormalized Updates.* arXiv:2504.05295 (2025).
- Xu. Honors thesis, UT Austin, May 2026. `nn.cs.utexas.edu/?xu:honorsthesis26`.
- Hughes et al. *Open-Endedness is Essential for Artificial Superhuman
  Intelligence.* ICML (2024). arXiv:2406.04268.
