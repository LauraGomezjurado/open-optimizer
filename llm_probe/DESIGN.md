# llm_probe — does the training *objective* move representational factoredness in a real transformer?

Status: **PREP ONLY.** No training runs on shared infra until a fresh go-ahead
(pod safety rule #3). Local box has no torch/GPU; this is authored to run on a
GPU dev pod.

## The question (carried over from the toy)

The CPPN toy said: optimizer *geometry* is a within-basin lever (flat on the
neuron metric), while the *objective* is the between-basin lever (lineage nudged
it, geometry didn't). We now test that at real-transformer scale, where "factored
vs fractured" has an established name: **monosemantic vs polysemantic / superposed**
features.

**Hypothesis:** holding architecture, data, optimizer, and final loss ~fixed,
varying only the *objective schedule* moves feature-structure factoredness. The
toy predicts the effect is real but small; the point of scale is to see if it
survives / grows when the representation has room to organize.

## Arms (only the objective differs)

All arms: same model, same tokens seen, same optimizer (AdamW), matched final
validation loss where possible.

1. **flat-AR** — standard autoregressive next-token prediction (the FER-regime
   baseline; analog of "fit the skull directly").
2. **staged** — curriculum over a *sequence of objectives* that build on each
   other (analog of Stanley's lineage). Candidate staging, cheapest first:
   - span/masked-denoising warmup → then autoregressive (objective changes), or
   - short-context → long-context schedule (the "coherent intermediate" grows),
   - each stage warm-started from the last.
3. **multiscale / diffusion-flavored** — a masked-diffusion / any-order objective
   (predict masked tokens at many mask ratios) vs AR. This is the *fair* version
   of the coarse→fine test the toy couldn't do justice to — real multi-scale
   pressure over the whole distribution, not a 4-step blur on one image.

Decision to make before building: whether arm 3 is "train a masked-diffusion LM
ourselves" (more infra) or "load a pretrained diffusion-LM + matched AR-LM and
only run metrics" (cheaper, inference-only). See open questions.

## The metric (the hard part — must be capacity/width-invariant)

The toy's lesson: weight-space brittleness metrics fail across architectures;
the metric that passed the UFR gate was **neuron-image structure**
(smoothness / low-frequency-ness of each neuron's response map). Transformers
have no image per neuron, so we port the *spirit*, not the code. Candidates,
to be validated against a known contrast (e.g. a base model vs its instruct
variant, or early vs late checkpoint):

- **Feature monosemanticity via an SAE** — train a sparse autoencoder on the
  residual stream; measure fraction of interpretable/monosemantic features, or
  the sparsity/reconstruction frontier. Closest to the field's UFR notion, most
  expensive.
- **Superposition proxy (cheap, no SAE)** — activation covariance effective rank
  *normalized by width* + participation ratio of features per token. Directly
  analogous to our `feat_effrank_corr_frac`, capacity-invariant by construction.
- **Weight-sweep analog** — perturb one neuron's out-weights, measure KL on
  output distribution; the transformer version of our sensitivity sweep. Known
  to be capacity-confounded (toy lesson) — include only as a diagnostic, not a
  headline.

Primary metric: superposition proxy (cheap, runs on every checkpoint). SAE only
if the cheap proxy shows a gap worth confirming.

## Scale & cost (to size the pod ask)

- Model: ~10–50M params (e.g. 6–8 layers, d_model 256–512). Small enough for one
  H100, big enough to have real superposition.
- Data: a small standard corpus (e.g. a TinyStories / a slice of C4/OpenWebText).
- Runtime target: each arm << the 12h pod window; ideally all arms in one session.
- ONE model server / training job per pod; check OOM budget (rule #2) before launch.

## Decisions (locked)

- **Train all arms ourselves** from the same init/data/optimizer — only the
  objective differs. Clean causal test, matches the toy's controlled spirit.
  (Rejected: pretrained comparison — too many uncontrolled differences.)
- **Metric: cheap proxy gates SAE.** Superposition proxy (width-normalized
  activation-covariance effective rank + per-token feature participation ratio)
  runs on every checkpoint of every arm. SAE monosemanticity is run ONLY on the
  arms/checkpoints where the proxy shows a gap worth confirming. Same cheap-first
  discipline that worked in the toy.

## Remaining questions (resolve on-pod, cheap to settle)

1. Corpus + tokenizer — reuse the pod's existing stack if present (likely a
   HF tokenizer + a TinyStories/C4 slice). Settle by inspecting the pod.
2. Exact staging schedule for arm 2 — default: masked-denoising warmup → AR
   (cleanest objective *change*); alt: short→long context.
3. Metric validation gate — before trusting the proxy, confirm it separates a
   KNOWN contrast (early vs late checkpoint of the same run: late should be more
   superposed/lower normalized rank, OR base-vs-instruct). If it can't tell a
   known pair apart, fix the metric before spending training compute.

## What runs where

- Local (here, now): model code, metric code, a CPU smoke test on random weights
  (shapes + metric sanity), deploy script. NO training.
- Pod (after go-ahead): the actual training arms + metric evaluation.
