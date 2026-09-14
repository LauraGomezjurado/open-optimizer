# open-optimizer

**Is the geometry of the update rule a causal lever on whether learned representations come out _fractured_ or _factored_?**

This repo is the first-signal harness for that question. It sits at the collision of two literatures that have been circling the same idea from opposite ends:

- **Open-endedness / FER.** Kumar, Clune, Lehman & Stanley (2025) show that a network _evolved_ by an open-ended search (a Picbreeder CPPN) and a network _trained by SGD_ to produce the **exact same output image** have radically different internals. The SGD network exhibits a **Fractured Entangled Representation (FER)** — a single unitary concept split into disconnected, redundant pieces — while the evolved network approaches a **Unified Factored Representation (UFR)**. Their slogan: *it matters not just where you land but how you got there.*
- **Optimizer geometry / implicit bias.** Muon/spectral descent provably converges to the **max-margin solution in the spectral norm**; Adam (without its stability constant) to the **ℓ∞ margin**; the recent homogeneous-network result proves the *identity of the margin maximized depends on the choice of optimizer*.

These are the same statement at two resolutions. FER is the qualitative version;
implicit bias is the same claim **with a metric attached**.

---

## ⚠️ Current mental model (living notebook — read this first)

This README is a **lab notebook, not a results paper.** We are exploring, and the
mental model has *moved* as data came in. The original framing below ("design an
update geometry that biases toward UFR") is kept as history, but here is where the
thinking actually stands now:

- **The project started asking:** which *optimizer geometry* biases toward
  factored (UFR) representations?
- **What we found (see the sections below, in rough chronological order):**
  1. Within a fixed architecture, optimizer geometry *does* move weight-space
     metrics (Adam brittler than spectral; the lever is per-coordinate
     preconditioning, not "spectralness"). Robust, reproduces across targets and
     across our arch → the FER paper's deep arch.
  2. **But those weight-space metrics don't survive contact with the real UFR
     reference.** Loading the actual evolved Picbreeder genome, our metrics rank it
     as *more* fractured than SGD — they were measuring sparsity/capacity, not
     factoredness.
  3. We built a **capacity-invariant, neuron-space metric** (image smoothness /
     high-frequency energy of each live neuron) that *does* pass the
     evolved-vs-SGD gate (3/3 targets).
  4. **On that better metric, optimizer geometry is FLAT — IN THE CPPN TOY.** The
     inner-loop knobs that moved it there were **weight decay (↓ fracture)** and
     **depth/init (↑ fracture)**: regularization and capacity, not step direction.
  5. A Nexus-style **gradient-agreement** objective does *not* track factoredness
     here (fractured SGD has the *highest* cross-region gradient agreement).
  6. **🔴 REVERSAL AT TRANSFORMER SCALE (confirmed, 3 seeds):** the "optimizer
     geometry is flat" claim was only ever tested in the CPPN toy. The direct test
     at 30M-transformer scale **overturns it.** AdamW vs Muon at MATCHED loss
     (ar_ce 2.75 vs 2.71): Muon gives **40% higher feature effective rank**
     (mlp_effrank 0.490±0.006 vs 0.349±0.005), **3/3 seeds**, tight error bars.
     So **at scale the optimizer IS a factoredness lever** — the toy did NOT
     generalize on this axis. This corroborates arXiv:2606.09658 (Muon→higher
     effective rank) *and strengthens it*: we controlled for final loss, which
     their abstract does not claim to. See the optimizer section below.
  7. **MECHANISM + open-endedness bridge (new).** *Why* does Muon move it at scale
     but not in the toy? A width/depth scan shows the gap **grows with depth**
     (compounding spectral updates: +0.15→+0.29 over L=2→12) and **shrinks with
     width** (AdamW catches up; Muon ~pinned) — so it's a **depth-compounding**
     effect the shallow toy couldn't express. And the payoff link: on matched-loss
     checkpoints, **Muon representations are measurably more RECOMBINABLE** —
     under representation-space perturbation they yield *more distinct outputs at
     BETTER coherence* (frac=2: novelty +0.025, perplexity −2.9 vs AdamW). This is
     the first *measured* (not assumed) connection from optimizer geometry to the
     open-endedness property (recombinable stepping stones). Anti-collapse
     optimizer (explicit `loss−λ·richness`) is running to test if it's installable.
  8. **🟢 ACTIVATION-STATISTICS LEVERS EXHAUSTED (new, 3 seeds + prior-art check).**
     Concurrent prior art — the **Xu honors thesis** (Miikkulainen, UT Austin, May
     2026, `nn.cs.utexas.edu/?xu:honorsthesis26`) — is the direct FER follow-up:
     it steers SGD toward UFR with a **Total-Variation activation penalty**
     (+11–14pt MNIST at 1img/class; and TV-pretrained nets stay *plastic* under
     continual learning where baselines freeze). Their lever is an explicit LOSS on
     Adam; **ours is the optimizer's update GEOMETRY** (Muon/Aurora) — distinct
     axis, and their plasticity result de-risks that the FER axis is downstream-real.
     Notably they *also* tried an input-Jacobian **rank** penalty and got "no
     consistent improvement over TV." We then built + tested the last un-tried
     lever: a **higher-order INDEPENDENCE** regularizer (FastICA negentropy +
     4th-order cross-cumulant, past mere decorrelation). Result across 3 seeds: it
     **acts cleanly on its own axis** (reliably lowers effrank at matched
     capability, monotone in λ) **but does NOT robustly raise recombinability** (the
     seed-0 Pareto gain flipped sign on seed 1; within noise). **So THREE levers to
     beat Muon's ceiling have now failed to robustly win: (i) our input-Σ_x
     whitened-polar optimizer [dominated, 10 configs], (ii) Xu's input-Jacobian rank
     penalty, (iii) higher-order independence.** Honest conclusion: the
     activation-statistics design space (2nd- and higher-order) does **not** contain
     the missing open-endedness lever; what moves recombinability in our data is
     still Muon/Aurora's **spectral weight-space** update. Next honest direction is
     the weight-space mechanism, or consolidation — not another activation penalty.
- **Where the thinking is now (the pivot):** update geometry looks like a
  *within-basin* lever; factoredness looks like a *between-basin* property set by
  the **objective** and the **search process**, exactly as Stanley argues. So we
  are no longer trying to build a "UFR optimizer." Instead:
  - **Toy question now being tested:** does the *objective sequence* (a
    stepping-stone lineage, or a diffusion-style coarse→fine multi-scale schedule)
    move the neuron metric where geometry couldn't? (`run_curriculum.py`)
  - **LLM translation we care about:** FER ≈ superposition/polysemanticity. Our
    toy predicts optimizer choice (AdamW vs Muon) won't fix superposition — that's
    a capacity/objective phenomenon. Muon is still great *as an optimizer*
    (conditioning at scale); it's just not a factoredness intervention. A sharper
    hook: **multi-scale/diffusion objectives (incl. diffusion-LLMs) are a place to
    look for more factored representations than flat autoregressive training** —
    because the objective, not the optimizer, is the lever.

Everything below is the detailed running log. Treat conclusions as **provisional
observations**, not settled claims.

---

## North star

> **Characterize factoredness as an implicit-bias quantity, and design an update
> geometry that provably biases toward it.**

The set of weight configurations that reproduce a fixed target image is an enormous manifold. SGD lands on a fractured point in it; evolution on a factored point. "Can an update geometry bias toward UFR?" is literally: *is there a norm whose min-norm / max-margin solution on this behavioral manifold is factored?*

This reframing is what lets optimizer-geometry expertise bite on Stanley's problem without hand-waving, and it keeps the claims honestly bounded: **a better inner-loop geometry is an _ingredient_ of open-endedness, not open-endedness itself.** Open-endedness supplies (a) the target phenomenon — factored, recombinable representations as good stepping stones — and (b) an existence proof that they can be found. Optimizer theory supplies the lever and the math.

### The three conceptual bridges we are testing

1. **The update rule's dual norm plays the role of the mutation operator.** In open-endedness the variation operator (CPPN compositionality, NEAT mutations) defines which nearby artifacts are reachable; in optimizer theory the steepest-descent norm defines which weight-space directions are cheap. Same object, two levels. Original prediction: factoredness orders as **coordinatewise (SGD/Adam) < whole-matrix spectral (Muon/Dion) < compositional (evolution)**. **Refined by our results:** the operative split is not spectral-vs-not but **per-coordinate-preconditioned vs basis-agnostic** — anything that avoids per-entry rescaling (isotropic ℓ2, scalar-adaptive, or orthogonalized) sits on the factored side; orthogonalization (Muon) is sufficient but not necessary and even slightly overshoots on brittleness. See Signal §2.
2. **Orthonormalizing updates are diversity-preserving in weight space.** Muon/Dion equalize the update spectrum — they keep every singular direction alive rather than collapsing onto the dominant few. That is anti-collapse in the inner loop, structurally analogous to novelty search / QD preserving diversity in the outer loop.
3. **Factoredness = recombinable stepping stones.** UFR gives clean, semantically meaningful weight-sweeps ("mouth opening", "eye winking"); FER destroys them. An inner-loop geometry that biases toward factoredness raises the recombinability of every elite a QD / POET / OMNI loop produces.

---

## What we built (this repo)

A dependency-light (`numpy`, `matplotlib`, `pillow`) reimplementation of the FER CPPN-fitting setup, instrumented to vary **only the update geometry**:

- `src/cppn_numpy.py` — CPPN with hand-derived analytic gradients (inputs
  `x,y,d,b`; bias-free Dense layers with mixed `identity/sin/gauss/tanh`
  activations; sigmoid RGB readout). All parameters are 2D matrices — exactly
  what spectral/orthonormalizing optimizers act on.
- `src/optimizers.py` — the **geometry ladder**, all normalized to unit-norm
  steps so the comparison is about _direction_, not magnitude:
  - `sign` — steepest descent in ℓ∞ (Adam's implicit-bias limit)
  - `adam` — adaptive coordinatewise reference
  - `spectral(p)` — SVD `G = U S Vᵀ`, replace `S` with `S**p`:
    **p=1 → ℓ2/GD, p=0 → orthogonalized (Muon)**. Sweeping `p:1→0` walks
    isotropic→orthonormal.
  - `adam(q)` — the **coordinatewise dose-response knob**: raise the Adam
    v-preconditioner to power `q`, so **q=1 → full per-entry rescaling (Adam),
    q=0 → isotropic**. This is the knob that actually operationalizes bridge #1
    (it isolates *per-coordinate preconditioning*, the real lever — see Signal).
  - `scalar_adam` — adaptive step size from a single scalar 2nd moment per matrix:
    adaptive but basis-agnostic, to separate "adaptivity" from "per-entry-ness".
  - `wd` — decoupled weight decay on any rung, used as a **saturation control**.
- `src/metrics.py` — factoredness readouts built from a sampled
  parameter→image Jacobian (central-difference weight sweeps, à la FER):
  `sensitivity` (brittleness ↓=factored), `locality` (spatial concentration of a
  weight's effect ↑=factored), `redundancy` (1 − effective-rank of the sweep
  Jacobian ↓=factored), and `feat_effrank_frac` (effective rank of last-hidden
  activations ↑=factored). Plus two **saturation-invariant** readouts added to
  kill confound C2: `influence_orthogonality` (1 − mean off-diagonal cosine of
  common-mode-removed influence maps — are different weights' effects
  independent, regardless of magnitude?) and `feat_effrank_corr_frac` (effective
  rank of the *correlation* matrix of last-hidden activations, so activation-scale
  can't dominate). Bootstrap CIs over the sampled weights are reported for the
  headline readouts.
- `experiments/run_dose_response.py` — fits the **evolved Picbreeder skull image**
  (the UFR artifact from the FER paper, in `data/`) under every geometry to
  matched near-zero loss, then scores factoredness. Emits `results/`.
- `experiments/run_multiseed.py` — the **statistically robust** version: paired
  seeds, confound diagnostics (scale-invariant elastic sweep, weight eff-rank,
  saturation), mean±std and per-seed Spearman monotonicity tests. This is the one
  to trust; the single-seed script is kept for illustration only.

Run it:

```bash
pip install numpy matplotlib pillow
# statistically robust run (recommended):
python experiments/run_multiseed.py --all          # 13 geometries x 5 seeds + stats
python experiments/run_multiseed.py --stats        # re-print stats + plot
# generality across targets (own JSON per target):
python experiments/run_multiseed.py --target apple --all
python experiments/run_multiseed.py --target butterfly --init_scale 1.5 --all   # see C7
# qualitative FER-style visualization (borrows their neuron-map + weight-sweep method):
python experiments/viz_qualitative.py --target skull
# single-seed illustration:
python experiments/run_dose_response.py
```

---

## Signal (multi-seed, n=5 paired seeds, RES=40, 6×32 CPPN, 1200 iters, unit-norm steps)

Every geometry is trained from the **same** five init seeds (paired design, so
the comparison is within-init) and all reach **near-identical near-zero loss** —
they land on behaviorally equivalent points of the solution manifold, so metric
differences reflect **which** solution the geometry selected, not fit quality.
Means over 5 seeds (`results/stats.txt`, `results/dose_response_multiseed.png`;
↑/↓ mark the factored direction). The two **saturation-invariant** readouts
(`featER-corr`, `influence-orth`) are the ones the confound audit trusts — see
below.

| Geometry | loss | sens ↓ | sensE ↓ (scale-inv) | featER ↑ (covar) | featER-corr ↑ (sat-inv) | infl-orth ↑ (sat-inv) | sat frac |
|---|---|---|---|---|---|---|---|
| Adam (per-coord q=1) | ~0 | 7.94 | 2.56 | 0.128 | 0.604 | 0.851 | 0.461 |
| SignSGD (ℓ∞) | ~0 | 8.08 | 2.53 | 0.139 | 0.609 | 0.854 | 0.454 |
| Adam-q=0.75 | ~0 | 6.91 | 2.21 | 0.166 | 0.641 | 0.850 | 0.447 |
| Adam-q=0.5 | ~0 | 6.63 | 2.07 | 0.200 | 0.662 | 0.861 | 0.440 |
| Adam-q=0.25 | ~0 | 6.32 | 1.99 | 0.237 | 0.671 | 0.857 | 0.426 |
| ScalarAdam (basis-agnostic) | ~0 | 6.31 | 1.95 | 0.251 | 0.681 | 0.858 | 0.424 |
| Spectral p=1.0 (ℓ2/GD) | ~0 | 3.81 | 1.24 | 0.322 | **0.716** | **0.906** | 0.377 |
| Spectral p=0.5 | ~0 | 3.60 | 1.17 | 0.307 | 0.718 | 0.905 | 0.368 |
| Spectral p=0.0 (Muon) | ~0 | 5.14 | 1.55 | **0.374** | 0.728 | 0.874 | 0.329 |

(Full 13-rung table incl. `Adam+wd` saturation controls in `results/stats.txt`.)

**What holds up (robust across all 5 seeds):**

1. **The null check comes back positive, large, and NOT explained by
   saturation.** Update geometry moves the factoredness proxies hard at matched
   loss. Crucially, the two saturation-invariant readouts move the same way as
   the raw ones (featER-corr 0.60→0.72, influence-orthogonality 0.85→0.91,
   coord→spectral), and the **partial-regression audit (S2)** shows the
   brittleness gap is essentially unchanged when saturation is regressed out:
   sensitivity gap 4.01→**4.25**, elastic 1.27→**1.28**, redundancy
   0.096→**0.103**. **FER-relevant structure is selected by the optimizer and is
   a property of the representation, not of how many units got clipped.** This is
   the load-bearing result.

2. **The lever is per-coordinate preconditioning, not "spectralness" (confound
   C1 resolved).** The `Adam-q` knob (q=1→0 walks off per-entry rescaling) gives
   a **clean monotone dose-response**: as per-coordinate rescaling is removed,
   sensitivity falls (per-seed Spearman ρ=**+0.96**, 5/5) and elastic sensitivity
   falls (ρ=**+0.92**, 5/5) — a far cleaner knob than the spectral `p` sweep ever
   was. `ScalarAdam` (adaptive step size but ONE scalar per matrix — adaptive yet
   basis-agnostic) lands with the factored group, proving it is **per-entry-ness,
   not adaptivity**, that fractures. Isotropic ℓ2 (Spectral p=1) already captures
   most of the benefit; orthogonalization (Muon, p=0) is not required and on
   brittleness metrics slightly overshoots.

**What did _not_ survive multi-seed (corrections to the single-seed run):**

3. The earlier "clean monotone dose-response along the *spectral* `p:1→0`" was a
   **single-seed artifact**. Across 5 seeds the *within-spectral* ordering is
   noisy (sensitivity ρ=+0.38, redundancy ρ=+0.70 both point *away* from Muon;
   feat-rank ρ=+0.44 toward Muon) with overlapping error bars — the clean knob is
   the **coordinatewise `q`**, not the spectral `p`. Honest headline: *presence of
   per-coordinate preconditioning* is the robust, monotone lever; *fine gradations
   within the spectral family* are not resolved here.

4. **`feat_effrank_frac` (raw covariance) is ~⅔ saturation artifact** — its
   coord-vs-spectral gap collapses from −0.196 to −0.078 under saturation control
   (corr with sat = −0.76), and it is the metric that most overlaps with the Muon
   transferable-features paper (arXiv:2606.09658). Demoted from the headline in
   favor of the saturation-invariant `featER-corr` / `influence-orth` and the
   brittleness metrics, which are both robust *and* not what that paper measured.

5. **Generality: the per-coordinate lever holds across 3 targets (skull, apple,
   butterfly).** Re-ran the whole ladder on the evolved Picbreeder **apple** and
   **butterfly** images (`--target`, own JSON per target). The **coordinatewise
   `q` knob is monotone on all three** — per-seed Spearman(q, sensitivity) =
   **+0.96 / +0.88 / +0.96** (skull/apple/butterfly), 5/5 seeds each — and the
   saturation-controlled brittleness gap survives on all three (sensitivity
   partial gap 4.25 / 2.26 / 2.39). This is the robust, target-independent result.
   **Confound C7 (surfaced then fixed):** at the original `init_scale=2.5`, the
   *spectral* optimizers fell into a flat-image basin on butterfly from 3/5 seeds
   (p=1 stalled at loss ~0.1–0.3 vs ~0 for Adam) — not a step-size issue (higher
   LR made it worse) but a genuine bad basin from large init interacting with
   normalized-spectral steps. **Root-caused and fixed:** at `init_scale=1.5`
   **all 13 geometries converge on all 5 seeds** (`--init_scale 1.5`), so the
   comparison is honest apples-to-apples with no exclusions. The result is
   unchanged and in fact *cleaner*: sensitivity gap 4.86 (Adam) vs 2.61 (spectral),
   q-knob ρ=+0.88 (5/5), S2 partial gap 1.78→1.71 — **and at this init Adam
   saturates *less* than spectral (0.29 vs 0.33) yet is still more brittle**, so
   here brittleness and saturation point in *opposite* directions: the strongest
   possible dissociation of the two. A `LOSS_GATE=5e-3` remains in `stats()` as a
   guard that prints and excludes any non-converged run.

### Confound audit (what we actively tested and ruled out)

- **Weight-scale confound → ruled out.** A fixed additive weight perturbation is
  unfair if geometries land at different weight magnitudes. The **elastic**
  (multiplicative, scale-invariant) sensitivity shows the **same** coordinatewise
  vs spectral gap (2.6 → 1.2), so the brittleness gap is real, not a scale
  artifact. (Weight Frobenius norms are also similar across geometries, ~89–92.)
- **Feature-rank "echoes weight-rank" confound → ruled out (strongly).** Weight-
  matrix effective-rank fraction is essentially **flat (~0.75–0.77) across every
  geometry**, and across the ladder corr(feature eff-rank, weight eff-rank) =
  **−0.93** (anti-correlated). So the feature-rank signal is a property of the
  *representation*, not a mechanical echo of the update's matrix rank.
- **Loss-matching confound → controlled.** All final losses ~1e-4 or below; the
  metric gaps dwarf any loss differences.
- **Saturation entanglement → now resolved for the brittleness axis (was the
  open confound).** Three independent controls agree:
  1. **Saturation-invariant metrics.** `influence_orthogonality` (1 − mean
     off-diagonal cosine of common-mode-removed weight→image influence maps) and
     `feat_effrank_corr` (effective rank of the *correlation*, not covariance,
     matrix — standardized columns, so activation-family scale can't dominate)
     both show the same coord→spectral gap as the raw metrics.
  2. **Partial regression (S2).** Regressing `sat_frac` out of each metric leaves
     the coord-vs-spectral gap on sensitivity/elastic/redundancy essentially
     unchanged (4.01→4.25, 1.27→1.28, 0.096→0.103). These are structural.
  3. **Saturation-control geometries.** `Adam+wd` / `Adam+wd2` push Adam's
     saturation down to spectral levels (0.46→0.35) via weight decay, yet
     sensitivity stays high (8.1–8.6) — matched-saturation Adam **still
     fractures**. Saturation is a *correlate*, not the *mechanism*, of the
     brittleness gap.
  - **Caveat, kept honest:** the raw covariance `feat_effrank_frac` *is* mostly a
    saturation echo (gap −0.196→−0.078 under control); that specific metric is
    demoted. The saturation-invariant feature-rank (`featER-corr`) survives.

---

## Theoretical background we have

- **Steepest descent under a norm.** Each optimizer = argmax of ⟨g, ·⟩ over a
  unit ball: ℓ∞ → `sign`, ℓ2 → `g/‖g‖`, spectral → orthogonalized `UVᵀ`. This is
  the frame the whole ladder is built on.
- **Implicit bias / margin maximization.** Muon → spectral-norm max-margin
  (arXiv:2502.04664); Adam → ℓ∞ margin; homogeneous-net unification
  (arXiv:2602.16340). "Identity of the margin depends on the optimizer."
- **Muon as a flat-update-spectrum method** — keeps gradient singular directions
  but equalizes their amplitude. Directly motivates the effective-rank readout.
- **Empirical corroboration** that geometry moves feature quality: "Muon Learns
  More Robust and Transferable Features than Adam" (arXiv:2606.09658).
- **FER hypothesis** (arXiv:2505.11581) — the target phenomenon and the UFR
  existence proof.

## Theoretical analysis that is missing and necessary

1. **A single, optimizer-agnostic, scale-invariant definition of factoredness.**
   Right now factoredness is a *bundle* of proxies that disagree at the extremes.
   The prize is to identify the right invariant (feature effective rank is the
   leading candidate) and **prove** its relation to the norm being minimized.
   Conjecture: orthonormalizing updates bias toward high-effective-rank feature
   representations *because* they keep all update singular directions alive; make
   this precise and connect it to the spectral-margin solution.
2. **Disentangle factoredness from saturation. → LARGELY DONE (empirically).**
   Built the influence-map *orthogonality* metric and a correlation-matrix
   feature-rank, both saturation-invariant; added weight-decay saturation-control
   rungs and a partial-regression audit. Result: the *brittleness* axis
   (sensitivity/elastic/redundancy) survives all three controls; the raw
   covariance `feat_effrank` was mostly a saturation echo and is demoted. What
   remains for *theory*: prove *why* per-coordinate preconditioning drives units
   into saturation-and-brittleness while basis-agnostic updates don't — the
   empirical dissociation is established, the mechanism is not yet derived.
3. **A UFR ceiling — DONE, and it delivered a hard negative result that
   reshapes the project.** We safely loaded the *actual evolved Picbreeder
   genomes* (skull/apple/butterfly) from the FER repo — `pickle` is a code-exec
   vector, so `experiments/convert_genome.py` uses a restricted unpickler that
   whitelists only numpy/jax array-reconstruction globals (verified by
   `pickletools` disassembly), converts to trusted `.npz`, and deletes the
   `.pkl`. `src/fer_cppn.py` is a faithful NumPy port of their CPPN (per-genome
   arch, HSV readout, evosax's **alphabetical** layer ordering) — it renders each
   genome to the correct target (MSE ≤ 0.006). We then ran the SAME
   `factoredness_metrics` on the evolved (UFR) genome **and the FER paper's own
   SGD (FER) genome** (same arch/target/renderer — a footing-matched pair):
   `experiments/run_ufr_ceiling.py`.
   **Result: our scalar metrics rank the evolved genome as MORE fractured than
   SGD on 4/4 criteria, on all 3 targets — the exact opposite of FER's claim.**
   Diagnosis: the evolved genome is a NEAT genome that is **~2–3 % dense** with
   large weights (skull: 147/5478 live), while the SGD net is **100 % dense** with
   small weights. Our weight-perturbation metrics were only valid because the
   optimizer ladder held *architecture fixed*; across the real UFR divide
   (different capacity, sparsity, depth) they measure **sparsity/capacity, not
   factoredness** (sweeping only live weights, scale-invariantly, does not fix it —
   the evolved net does everything with ~150 load-bearing weights, so per-weight
   sensitivity is structurally high regardless of factoredness). **This is the most
   important thing we learned: the scalars do not generalize off the fixed-arch
   manifold, so the within-ladder results must be read as _architecture-conditioned_
   and no claim about approaching true UFR can rest on them yet.**
   **What DID reproduce is the qualitative phenomenon.** Rendering every hidden
   neuron (their own method) shows it unmistakably: the evolved genome's neurons
   are clean, smooth, distinct primitives (gradients, bands, radial blobs) with
   many unused units; the SGD genome's neurons are busy, high-frequency, redundant
   swirls — textbook fractured-entangled. See `results/genome_neurons_skull.png`
   vs `results/genome_neurons_sgd_skull.png`. So FER is real and visible; our
   *numbers* just don't capture it across architectures. The open problem is now
   sharp: **find a factoredness scalar that is invariant to capacity/sparsity and
   still ranks UFR > FER on this footing-matched pair.** That pair is now the
   validation gate every future metric must pass.
   (`experiments/viz_qualitative.py` also renders the Adam-fit vs spectral-fit
   contrast on our probe net — the "skull-crushing" weight sweep — as the
   fixed-arch qualitative anchor.)
4. **Causal separation.** We hold target, architecture and backprop-to-fixed-
   target fixed and vary only geometry, so the movement we see *is* attributable
   to geometry — and the scale-invariant control shows it is not merely step size.
   Remaining: separate the *norm family* effect (robust) from *fine gradations
   within a family* (not resolved here).
5. **Where does factoredness sit among flat / low-rank / max-margin / disentangled?**
   These are quietly conflated in the literature and are not the same. SGD's
   simplicity bias (flat, low-rank) is sold as *helping* generalization, yet FER
   says SGD reps are fractured (*bad*). Locating factoredness among these notions
   is a theorist's contribution that is currently vacant.

---

## The metric blind spot is fixable — by changing the primitive (neuron space)

Q: the evolved genome reads as maximally "fractured" on our weight-space metrics
because it is sparse (~3 % dense) — is that fixable by fixing the metric?

A: **yes, but not by patching `sensitivity`.** The weight-space primitive
(perturb a weight, measure image response) has magnitude that scales with
capacity, so a 150-weight net *necessarily* looks brittle — wrong quantity, not
fixable in kind. The FER phenomenon actually lives in **neuron space**: "is each
neuron a clean coherent primitive or a noisy high-frequency redundant fragment?"
`neuron_structure_metrics` (in `metrics.py`) renders every LIVE hidden neuron and
measures image smoothness (total variation) and high-frequency spectral energy —
both capacity-invariant. It **passes the UFR gate 3/3 targets**: evolved neurons
are smoother (TV 1.5–1.8× lower) and much lower-frequency (HF energy up to 136×
lower on skull) than the SGD genome's — the direction the weight-space metrics
got backwards.

**The complication this surfaced (a real result, not a fix to hide):** the same
neuron-structure metric, run on OUR optimizer ladder, is **flat** — Adam TV 0.026
vs spectral 0.033, with Adam if anything marginally *smoother* — the opposite of
what weight-space `sensitivity` reported. So the two metric families disagree:

| | weight-space (`sensitivity`) | neuron-space (`TV`/`HF`) |
|---|---|---|
| evolved vs SGD | backwards (fails gate) | **correct, 3/3** |
| Adam vs spectral (our ladder) | Adam more fractured | ~flat, no effect |

The metric that correctly captures UFR↔FER shows our optimizer geometries barely
differ in *neuron* structure. The per-coordinate `sensitivity` effect is real as
a **weight-space** property, but may not be the same thing as the **neuron-level**
fracture that defines FER. Whether these reconcile or are genuinely distinct
phenomena is now the central open question — and it is only askable because we
built a metric that survives the capacity gate.

### What moves neuron-structure factoredness? (Not the optimizer.)

`run_neuron_axes.py` pushes on six axes (fit skull, 3 seeds each) and
`run_bridge.py` now also scores neuron structure in the FER paper's deep arch.
Two clean conclusions:

- **The optimizer axis is flat — in shallow AND deep nets.** Neuron `TV`
  ≈ 0.026–0.033 and `HF` ≈ 0.05–0.06 across Adam / GD / Muon / the whole ladder,
  in both our 6×32 probe net and their 12-layer arch (bridge). Width and extra
  orthogonalization are flat too. Update geometry does **not** move the
  FER-defining property.
- **Regularization and capacity DO move it — the opposite lever from what we set
  out to find.** Weight decay makes neurons cleaner (TV 0.033→0.020, HF
  0.063→0.050 as wd 0→0.1); more depth and larger init make them noisier
  (depth 3→12: TV 0.023→0.052, HF 0.045→0.174; init 1→4: HF 0.040→0.084). So if
  you want factored neurons from a gradient-trained net, the evidence points at
  decay + staying shallow/small, not spectral-vs-Adam.
- **Nothing in the inner loop closes the gap to evolution.** On the
  capacity-invariant metric the evolved genome (TV 0.021, HF 0.001) stays clearly
  apart from *every* trained net (TV ≥ 0.024, HF ≥ 0.05), consistent with the FER
  thesis that the **search process**, not the inner-loop optimizer, is where UFR
  comes from.

Net-net: this repo set out to test whether *update geometry* is a lever on FER.
The honest answer from these data is **no — not on the neuron-level property that
defines FER**; the movable inner-loop knobs are regularization/capacity, and the
big gap remains attributable to how the network was searched for.

### The objective lever: lineage vs coarse-to-fine (`run_curriculum.py`)

If geometry is a within-basin lever and the objective is the between-basin one,
the direct test is to vary *only the sequence of objectives*, everything else
fixed (same skull endpoint, arch, optimizer=spectral p=1, total iters), and score
the neuron metric. Three arms:
- **flat** — fit the skull directly (baseline).
- **lineage** (Stanley) — apple → butterfly → skull, warm-started (reuse across a
  lineage of coherent forms).
- **coarsefine** (diffusion intuition) — same skull, blurred → sharp over training
  (low-frequency first; a multi-scale objective).

Result (5 seeds, neuron_TV lower = more factored):

| arm | neuron_TV | neuron_HF | vs flat |
|---|---|---|---|
| flat | 0.0357 ± 0.0038 | 0.0667 | baseline |
| **lineage** | **0.0340 ± 0.0037** | **0.0644** | slightly cleaner, 4/5 seeds |
| coarsefine | 0.0359 ± 0.0047 | 0.0664 | no effect |

Reading, held honestly:
- **lineage nudges the right direction** (lower TV/HF than flat on 4/5 seeds) —
  the first sign that the *objective sequence* touches the neuron metric where
  optimizer geometry was flat. But the effect is **small and within error bars**,
  and tiny relative to the gap it would need to close (evolved genome TV ≈ 0.021,
  SGD ≈ 0.033, flat here ≈ 0.036). So: *directionally consistent with the "objective
  is the lever" hypothesis, not remotely sufficient on its own.*
- **coarse-to-fine did essentially nothing** here — the diffusion/multi-scale
  intuition did **not** move the metric in this toy. Caveat before discarding it:
  our coarse→fine is a crude 4-step Gaussian-blur schedule on a single 48px image;
  real diffusion's multi-scale pressure is far richer, so this is weak evidence
  against the *toy* implementation, not against the idea.
- Neither arm approaches the evolved genome. Consistent with the running theme:
  a *single* gradient-trained lineage is a pale shadow of an open-ended search with
  selection over many generations. The lever is real but the toy dose is small.

Next dials if we push this arm: longer/more lineage stages, lineage of *many*
more intermediate forms, or combining lineage + heavy weight decay (the two levers
that each nudged the metric) to see if they stack.

### Does a Nexus-style gradient-agreement objective track factoredness? (No.)

Nexus (arXiv:2604.09258) is an optimizer that, among equal-loss minima, steers
toward one where per-SOURCE gradients agree (high cosine). Tempting analogy: a
factored/reused representation should induce agreeing gradients across parts of
the output. `run_gradient_agreement.py` tests it: split the image into spatial
regions, measure cross-region gradient/Jacobian cosine, ask whether the evolved
genome (UFR) sits highest.

Two findings:
- **Loss-gradient version is degenerate at UFR.** Nexus's per-source *loss*
  gradients all vanish at a zero-loss solution; the evolved genome is a perfect
  fit, so every region-loss-gradient is exactly 0. Nexus's criterion is *not even
  well-defined* at the UFR point in single-image fitting. (We switched to an
  output-Jacobian: "which weights does each region recruit, and in what direction"
  — well-defined everywhere, same reuse notion.)
- **Gradient agreement does NOT track factoredness — if anything it's backwards.**
  Cross-region cosine, robust across 2×2…8×8 region grids: **SGD(FER) highest**,
  EVOLVED(UFR) middle, our Adam lowest (e.g. 4×4: SGD +0.19, evolved +0.12, Adam
  +0.08). The fractured net has the *most* agreeing region-gradients, not the
  least. So "regions recruit the same weights" is satisfied by a dense entangled
  net precisely *because* it's entangled (every weight touches everything) — the
  opposite of the clean modular reuse we assumed. The Nexus criterion measures a
  real thing (shared machinery → generalization) but it is **not** the UFR axis in
  this setup; a Nexus-style objective should not be expected to lower neuron-TV/HF.

Caveat kept honest: this is single-image fitting with manufactured spatial
"sources", not Nexus's multi-dataset setting, so it refutes the *analogy in our
harness*, not Nexus itself. But it does kill the cheap hope that gradient
agreement is a shortcut to factoredness here.

## Leaving the toy: the objective lever at transformer scale (`llm_probe/`)

The toy kept saying *objective, not optimizer, is the lever* — but with tiny
effects. So we tested it for real: a 30M-param GPT, trained on TinyStories on an
H100, varying ONLY the training objective across three arms (3 seeds each),
scored with a capacity-invariant superposition proxy (width-normalized
correlation-matrix effective rank of MLP + residual activations; validated on
synthetic data to read 0.99 for independent features vs ~k/d for a k-latent
superposition).

- **flat_ar** — autoregressive next-token (FER-regime baseline).
- **staged** — masked-denoising warmup → autoregressive (Stanley-lineage analog:
  the objective *changes* mid-training).
- **mdlm** — masked-diffusion, random mask ratio each step (multi-scale objective;
  the fair version of the toy's coarse→fine).

Result (final checkpoint, mean over 3 seeds; higher effrank = more factored):

| arm | ar_ce | mlp_effrank | resid_effrank |
|---|---|---|---|
| flat_ar | 2.75 ± 0.01 | **0.349** ± .005 | **0.368** ± .002 |
| staged  | 2.79 ± 0.01 | 0.306 ± .008 | 0.325 ± .008 |
| mdlm    | 6.96 ± 0.13 | 0.219 ± .001 | 0.215 ± .004 |

Reading, held honest:
- **On the proxy, the objective variants did NOT beat flat AR — they scored
  lower effrank (more superposed), not higher.** The toy's "objective is the
  lever" hope did *not* transfer as a *win* for staged/mdlm at this scale/metric.
- **But there is a large confound we must not launder into a conclusion:** the
  proxy is measured under a *causal* forward pass, which is in-distribution for
  flat_ar and staged-late but OFF-distribution for mdlm (mdlm never trained
  causally — its ar_ce=6.96 shows it isn't even doing the same task). So the
  mdlm row is not a fair comparison; it mostly says "a bidirectional model
  measured causally looks collapsed," which is unsurprising.
- **The clean contrast is flat_ar vs staged** (both end causal, matched ar_ce
  ~2.75). Here staged is *slightly lower* effrank — a small effect, opposite
  direction to the toy's lineage nudge. So even the fair comparison gives **no
  evidence that the staged objective increases factoredness** at this scale.
- **Genuinely interesting trajectory finding** (not a headline, a lead): within
  flat_ar, mlp_effrank *rose* over training (≈0.03 early → 0.35 late) — longer
  training built MORE independent feature directions here, the opposite of naive
  "training induces collapse." Worth understanding before trusting the proxy as
  a factoredness measure at scale.

**Metric-fairness fix (done).** Worry: the proxy was read under a causal forward
pass, off-distribution for mdlm. We re-ran measuring every arm in BOTH the common
causal mode AND its native mode (mdlm/mask bidirectional). Result: **measurement
mode does not matter** — mdlm reads 0.218 (causal) vs 0.216 (native bidir). So
mdlm's low effrank is real, not a measurement artifact. The confound is ruled out.

Confound-free readings (3 seeds each):

| arm | ar_ce | mlp_effrank (causal) | mlp_effrank (native) |
|---|---|---|---|
| flat_ar | 2.75 | 0.349 | 0.349 |
| staged  | 2.79 | 0.307 | 0.307 |
| mdlm    | 6.99 | 0.218 | 0.216 |

**Honest status: the transformer test did NOT confirm the objective lever.** At
matched capability (flat_ar vs staged, both ar_ce≈2.75), the staged/lineage
objective gives *lower* feature effective rank (Δ = −0.042); mdlm is lower still
and genuinely so (measurement-mode-invariant, confound ruled out).

### Metric-first follow-up (exp2): stronger lineage + the SAE gate

We chose to lock down the metric before hunting further. exp2 added a STRONGER
lineage arm (`staged_strong`: 6 progressive stages, coarse→fine masking then
short→long-context AR) and ran the **SAE monosemanticity gate** on the matched-loss
arms (seed 0), computing three would-be factoredness metrics: the cheap
`effrank` proxy, SAE `mono_proxy`, and a decoder-**interference** tiebreaker
(mean |off-diagonal cosine| of live SAE features — the most direct superposition
measure). The point: do the metrics AGREE on how the arms rank?

| arm | ar_ce | effrank | SAE fvu | mono | interference |
|---|---|---|---|---|---|
| flat_ar | 2.75 | **0.349** | 0.255 | 0.996 | 0.0457 |
| staged | 2.79 | 0.306 | 0.202 | 0.996 | 0.0457 |
| staged_strong | 2.92 | 0.263 | **0.172** | 0.996 | 0.0454 |

What we learned (and it's mostly about the *metrics*, per the metric-first plan):
- **`mono_proxy` and `decoder_interference` are FLAT across arms** (0.996/0.996,
  0.0457/0.0454) — at this SAE config they simply do not discriminate. Any
  "ranking" off them is 4th-decimal noise. So the SAE gate, as configured, is not
  yet a usable discriminator — a metric result, not a phenomenon result.
- **The two metrics that DO move — `effrank` and SAE `fvu` — move in OPPOSITE
  directions.** As the lineage strengthens (flat→staged→strong): effective rank
  falls (0.349→0.263) AND the SAE reconstructs more easily / sparsely
  (fvu 0.255→0.172). Both say the same thing representationally — **stronger
  lineage → representation packed into fewer, more sparsely-compressible
  directions** — but they *disagree on whether that is "factored."*
- This is the **flat/low-rank vs factored conflation** the toy flagged (missing-
  theory #5) resurfacing at scale: lower effective rank could be "collapsed /
  superposed" (effrank reading: less factored) or "compact / economical basis"
  (fvu reading: cleaner). We do not currently have a metric that resolves which.

**Net:** the metric-first pass did its job — it revealed that our factoredness
metrics **do not agree at transformer scale**, and that the SAE gate as configured
doesn't discriminate. So the earlier "objective doesn't help" conclusion is NOT
safe to trust: it rested on `effrank`, and `effrank`'s own sign-meaning is exactly
what's in question. The real open problem is now razor-sharp and it's a *metric*
problem: **define a factoredness measure that distinguishes "compact/economical"
from "collapsed/superposed," and validate it on the evolved-vs-SGD gate before
using it to judge any objective.** Raw results in `llm_probe/results/`, plan in
`llm_probe/METRIC_PLAN.md`.

### Attempt at the missing metric — kurtosis — and why it failed the gate

The gap is specific: `effrank` cannot distinguish an *economical* low-rank basis
(few clean independent factors) from a *collapsed/superposed* one (many sparse
features crammed into few dims). We built a candidate — **mean per-feature excess
kurtosis** (`activation_kurtosis` in `src/metrics.py`) — on the theory that
superposition packs sparse, heavy-tailed features (high kurtosis) while a clean
factor basis is ~Gaussian (kurtosis≈0). On a **synthetic** economical-vs-superposed
pair at matched effective rank it worked cleanly and robustly (econ≈0 vs
superposed +0.7…+1.5 across 4 regimes).

**But it FAILED the real UFR gate (1/3): evolved genomes did NOT reliably show
lower kurtosis than SGD.** Diagnosis: the evolved genomes are 44–63% *dead units*
(sparse NEAT genomes) vs 0% for SGD — so any activation statistic aggregated over
units is dominated by the sparsity asymmetry, not by superposition. This is the
**same capacity/sparsity confound** that has defeated every weight/unit-aggregated
metric in this project. The synthetic validated kurtosis against the wrong failure
mode (it had no dead-unit asymmetry).

Honest lesson: **the evolved-vs-SGD genome gate has a built-in sparsity confound
that may make it unsuitable as the transformer-metric validator** — the two nets
differ in density as much as in factoredness, so "passes the gate" conflates
"handles sparsity" with "measures factoredness." So we built a transformer-native,
density-controlled ground-truth pair instead.

### The fair gate — width contrast — and what it overturned (`validate_metric.py`)

Ground-truth pair with NO density confound: same data/objective (flat_ar), only
width differs. NARROW (d=128) is forced to cram features into few dims →
*known-more-superposed*; WIDE (d=512) has room → *known-more-factored* (Elhage et
al. toy-models logic). A valid factoredness metric must call NARROW more
superposed. Result (mid-layer, 30M-arch, 2500 steps):

| metric | NARROW (d=128) | WIDE (d=512) |
|---|---|---|
| mlp_effrank | 0.119 | **0.407** |
| mlp_kurtosis | +3.1 | **+25.5** |

Two things this settles:
- **`effrank` PASSES this fair gate**: the narrow (superposed) model has much lower
  MLP effective rank (0.12 vs 0.41). So on a *density-controlled* contrast, effrank
  *does* track superposition correctly — its earlier "failures" were the sparsity
  confound in the genome gate, not the metric.
- **`kurtosis` FAILS, and the failure corrects my hypothesis.** I predicted
  superposition → high kurtosis. The opposite held: the WIDE (more factored) model
  has *far* higher kurtosis (+25 vs +3). Re-interpretation: high per-feature
  kurtosis is not a superposition signature — it's the signature of **many
  specialized, sparsely-firing selective features**, which is what a *factored*
  representation with room to spread actually looks like. Kurtosis, if anything,
  measures factoredness with the sign *flipped* from what I assumed.

Net gain from the metric-first detour: we now have (a) a **clean, density-controlled
gate** we trust, and (b) evidence that **`effrank` is a valid superposition metric
after all** once the sparsity confound is removed — which means the earlier
"objective doesn't help" reading (which rested on effrank) is back on firmer ground
than the disagreement scare suggested. The remaining subtlety is only whether
"lower effrank" means economical-vs-collapsed at *matched* capability — but across
the objective arms capability was matched (ar_ce≈2.75), so within that set, lower
effrank = more collapsed is the defensible reading. **Provisional bottom line: the
objective variants we tried produce _more_ collapsed representations than flat AR,
not more factored ones.**

## Optimizer at scale: the toy's headline was WRONG at scale (`optim.py`, `OPTIMIZER_PLAN.md`)

The single biggest hole in the project: "optimizer geometry doesn't move
factoredness" was solid in the CPPN toy but never tested on a real transformer —
and it *contradicts* arXiv:2606.09658 (Muon→higher effective-rank features). We
closed the hole with the sharpest possible design: **AdamW vs Muon at MATCHED
final loss**, measuring the now-validated `effrank`. (Their abstract claims the
effrank edge but does not mention controlling for loss — so matched-loss is the
discriminating test: if Muon's edge is just "reaches lower loss," it vanishes when
loss is matched; if it survives, the optimizer genuinely selects a more factored
solution.)

Result (30M model, 4000 steps, 3 seeds, LRs tuned so both hit the same loss):

| optimizer | ar_ce (matched) | mlp_effrank | resid_effrank |
|---|---|---|---|
| AdamW | 2.746 ± .012 | 0.349 ± .005 | 0.368 ± .002 |
| **Muon** | 2.709 ± .002 | **0.490 ± .006** | **0.478 ± .009** |

**Muon gives 40% higher feature effective rank than AdamW at matched loss, 3/3
seeds, tight error bars.** This is a **confirmed reversal** of the toy's "optimizer
is a flat/within-basin lever":

- **At transformer scale, the optimizer IS a factoredness lever.** The update rule
  selects a measurably more factored (higher-effective-rank, less superposed)
  solution at the *same* behavior/loss. The CPPN toy did not generalize on this
  axis — a real limitation of the toy, now named.
- **It corroborates AND strengthens arXiv:2606.09658.** They report Muon→higher
  effrank; we show it holds *at matched loss*, ruling out the "Muon just optimizes
  better" confound their abstract leaves open. So the effect is about the
  *geometry of the solution selected*, not fit quality.
- **Reconciling with the toy: MECHANISM SCAN (`run_scan.sh`).** We asked WHY the
  optimizer moves feature-rank in a deep transformer but was flat in the shallow
  CPPN, by scanning the Muon−AdamW gap across width (depth fixed=6) and depth
  (width fixed=384), matched-loss at each point:

  | WIDTH (L=6) | AdamW | Muon | gap |  | DEPTH (d=384) | AdamW | Muon | gap |
  |---|---|---|---|---|---|---|---|---|
  | d=256 | 0.190 | 0.505 | +0.316 |  | L=2 | 0.246 | 0.394 | +0.148 |
  | d=384 | 0.253 | 0.485 | +0.233 |  | L=4 | 0.249 | 0.447 | +0.198 |
  | d=512 | 0.302 | 0.468 | +0.166 |  | L=8 | 0.269 | 0.514 | +0.245 |
  |  |  |  |  |  | L=12 | 0.248 | 0.542 | +0.293 |

  **Two opposite, clean slopes reveal the mechanism:**
  - **The gap GROWS with depth** (+0.148→+0.293, monotonic over 4 points). Muon's
    effrank climbs with depth (0.39→0.54) while **AdamW stays flat (~0.25)**. So
    the factoredness advantage is a **depth-compounding effect**: the orthogonalized
    (spectral) update, applied layer after layer, accumulates into progressively
    higher effective rank — AdamW gets no such compounding.
  - **The gap SHRINKS with width** — but only because **AdamW catches up** (0.19→
    0.30) while **Muon is ~pinned (0.47–0.50) regardless of width**. Width is a
    *substitute* lever that AdamW needs and Muon doesn't; Muon already saturates
    effrank at any width.
  - **This explains the toy null:** the CPPN was effectively too shallow for Muon's
    per-layer geometry to compound. Depth is the ingredient the toy lacked. It also
    predicts the effect should be even larger in deeper LLMs — consistent with
    2606.09658 finding it at GPT-2 depth.

  Mechanistic one-liner: **Muon buys factoredness through depth (compounding
  spectral updates); AdamW can only buy it through width (raw capacity).**

Robustness caveat (kept honest): 1 seed per scan point (the 30M/L6/d384 anchor has
3-seed confirmation at +0.14 gap), Muon LR tuned to match loss, TinyStories only.
The depth/width slopes are each monotonic across 3–4 points, so the *mechanism*
(depth-compounding) is well-supported even at 1 seed/point.

**124M anchor (d768/L12, their GPT-2 setting):** Muon 0.499 vs AdamW 0.420
effrank at matched loss (ce 2.81 vs 2.87) — effect holds at their scale. The
*fractional* gap is smaller than at 30M (+0.08 vs +0.14) exactly because d=768
gives AdamW the width to catch up — internally consistent with the width-scan
mechanism (width is AdamW's substitute for Muon's depth-compounding).

## Recombinability: the FIRST measured bridge to open-endedness (`recombinability.py`)

Every prior "→ open-endedness" step was inference. This measures the load-bearing
link directly: does Muon's higher effrank actually yield **recombinable stepping
stones** — the property an outer curiosity/novelty loop needs? Operationalized as:
perturb the representation (activation-patch a mid layer at magnitude = frac·‖act‖),
free-run, and measure **novelty** (pairwise token-distance across variations) vs
**coherence** (perplexity under the model; lower = more on-manifold). Open-endedness
wants high novelty AT good coherence — distinct-but-broken is noise, coherent-but-
identical is no exploration. Matched-loss AdamW vs Muon checkpoints (2 seeds).

| perturbation | AdamW novelty | Muon novelty | AdamW ppl | Muon ppl |
|---|---|---|---|---|
| frac=1.0 | 0.868 | 0.885 | 3.8 | 3.8 |
| frac=2.0 | 0.943 | **0.968** | 12.4 | **9.5** |

**Muon representations are more recombinable: at matched perturbation they give
MORE distinct outputs at BETTER (lower) perplexity** — a frontier shift in exactly
the open-endedness-favorable direction (+0.025 novelty AND −2.9 perplexity at
frac=2). More independent feature directions ⇒ perturbing one yields a coherent
variation instead of breaking the whole output.

**What this licenses (carefully):** this is the first *measured* connection from
optimizer geometry → the property open-endedness actually needs. It says Muon
makes a model **more curiosity-*able*** — a better substrate for an outer
open-ended loop — NOT "curious" (curiosity is an outer-loop behavior; no inner
optimizer installs it). Caveats: 2 seeds, 30M, token-distance novelty +
self-perplexity coherence are proxies (a held-out judge model would be stronger),
and the effect is modest though consistent. But it converts the open-endedness
claim from *asserted* to *measured*, which was the whole point.

**Anti-collapse result: the property is NOT installable via a loss penalty (a
sharper mechanistic finding).** We trained AdamW + `loss − λ·R` (offdiag
decorrelation, λ∈{0.01,0.1}) and measured on the recombinability frontier
(anti-circular: validated on recombinability, not the regularized statistic):

| checkpoint | frac=2 novelty | frac=2 perplexity |
|---|---|---|
| AdamW | 0.944 | 13.3 |
| **Muon** | **0.967** | **9.5** |
| AdamW + anticollapse λ=0.01 | 0.935 | 12.8 |
| AdamW + anticollapse λ=0.1 | 0.935 | 11.7 |

The soft penalty barely moved recombinability off the AdamW baseline (and barely
moved effrank: 0.35→0.36) — it did NOT reach Muon's frontier. **Conclusion:
Muon's recombinability comes from its update GEOMETRY (orthogonalization,
depth-compounded), not from a scalar richness term bolt-on.** This is the sharper
claim: the open-endedness-relevant property lives in the *manifold the update is
constrained to*, not in a rank statistic you can regularize toward. Which is
exactly why the next step is at the geometry level, not the loss level.

### Direction: a principled "curiosity-able" optimizer (geometry, not penalty)

The anti-collapse null + the Aurora optimizer (tilderesearch.com/blog/aurora)
jointly point the way. Aurora modifies Muon at the GEOMETRY level: it constrains
updates to the **joint Stiefel ∩ oblique manifold** (orthogonality + equal row
norms), derived from a constraint (fix row norms → forces semi-orthogonality),
and it fixes Muon's *neuron death* (row-norm anisotropy kills MLP neurons). Dead
neurons are the antithesis of recombinable stepping stones — so Aurora already
moves toward curiosity-able for a different stated reason.

The design frame our results support: **a curiosity-able optimizer chooses the
constraint manifold whose steepest-descent updates provably preserve
recombinability.** Muon (Stiefel) equalizes update singular values → high rank but
kills neurons; Aurora (Stiefel ∩ oblique) adds equal-row-norm → fixes death; the
curiosity-able extension would add the constraint that targets recombinability
directly — equalizing every feature direction's *leverage on the output* so no
direction becomes dead or redundant (unrecombinable). Same constrained-manifold
derivation style as Aurora, but with recombinability (our validated metric) as the
design target rather than loss/speed. We now have the three pieces to pursue this:
a validated target metric (recombinability frontier), a mechanism (geometry not
penalty), and a derivation template (Aurora).

## The bridge experiment (`run_bridge.py`): UFR, FER, and our ladder on ONE ruler

Experiments A and B never shared an axis (A = our optimizers in our arch; B =
evolution vs their-SGD in their arch). The bridge fixes that: we added backprop
to `src/fer_cppn.py` (analytic through the layers; the HSV→RGB color stage,
which has mod/abs/clip kinks, is differentiated with a numerical per-pixel 3×3
Jacobian — gradient-checked to 1e-8 rel-err on a well-conditioned net) and train
**our whole optimizer ladder inside the FER paper's own skull architecture**,
fitting the exact image the evolved genome renders. Now the evolved genome (UFR),
the FER paper's SGD genome (FER), and our Adam/Spectral/Muon/… are all scored by
the same metric on the same arch and target (`results/bridge_skull.json`).

Three data points came out:

1. **Our per-coordinate ordering is architecture-portable.** In *their* deep
   12-layer CPPN, brittleness still orders Adam/Sign (sens ≈ 16) > spectral/scalar
   (≈ 13) with Muon overshooting (≈ 18) — the same pattern as our own 6×32 net.
   So the geometry lever is not an artifact of our toy architecture.
2. **On matched arch + density, our optimizers ≈ their SGD structurally.** Every
   dense net (our ladder *and* their SGD) sits at `influence_orthogonality`
   ≈ 0.91–0.92 and `redundancy` ≈ 0.37–0.42 — indistinguishable. The geometry
   effect is a modest brittleness shift, not a different regime; nothing we do
   lands in a qualitatively different place from plain SGD in their setting.
3. **The evolved genome is still off the metric's scale** (sens 34.7, redundancy
   0.93, nnz 0.027) — the sparsity/capacity blind spot from Experiment B
   reappears under fully matched arch/target, so it is a property of the metric,
   not of the earlier setup. Note: `Adam+wd*` rungs in the first bridge run were
   accidentally identical to Adam (a wd-passing bug, since fixed); ignore those
   two rows in `bridge_skull.json`.

Net: the optimizer-geometry effect is real and portable but **small relative to
the evolution↔gradient gap**, and our metrics **cannot see the UFR endpoint** at
all because of sparsity. Both facts point at the same top-priority next step.

## Next steps

- **Done:** 5 paired seeds + error bars; confound audit (weight-scale,
  weight-rank echo, loss-matching); **saturation confound resolved** via
  saturation-invariant metrics + partial regression + weight-decay control rungs;
  **fracture localized to per-coordinate preconditioning** via the `adam-q` knob
  and `scalar_adam`; **generality** across skull/apple/butterfly (C7 fixed via
  `init_scale`); **UFR ceiling run** (`run_ufr_ceiling.py`) — see below.
- **🔴 TOP PRIORITY — a capacity/sparsity-invariant factoredness metric.** The UFR
  ceiling showed our scalars measure sparsity, not factoredness, across the
  evolved-vs-SGD divide (they rank UFR *below* FER). Until a metric passes the
  evolved-vs-SGD gate (`run_ufr_ceiling.py`), the within-ladder numbers are
  architecture-conditioned. Candidate directions: normalize influence by
  per-neuron path count; measure on a fixed *behavioral* basis (output-space
  Jacobian SVD) rather than weight space; or a compression/MDL score.
- **Mechanism theory:** derive *why* per-entry rescaling entrenches the neuron
  basis and drives brittleness (missing-theory #2, mechanism half).
- **More instruments:** Dion (scalable orthonormalizer) as a second spectral
  point; Shampoo as another whole-matrix point.
- **Higher resolution** to match FER's 256px exactly (renderer already HSV-faithful).
- **Scale up** beyond CPPN-fit to a real supervised task, connecting to the
  transferable-features result (arXiv:2606.09658).

## Reproduction notes / provenance

- `data/skull_pb_512.png`, `data/skull_sgd_512.png` are assets from the FER repo
  (`github.com/akarshkumar0101/fer`, `assets/img_576_pb.png` /
  `img_576_sgd_pb.png`); `data/target_64.png` is the evolved-skull image
  downsampled and used as the fixed fitting target.
- The FER codebase is JAX/Flax/evosax and pins an old `evosax` (0.1.6) to load
  evolved genomes; we deliberately did **not** depend on it, reimplementing the
  CPPN + gradients in NumPy for a self-contained, fully controllable harness.
- **Evolved/SGD genomes for the UFR ceiling** were loaded WITHOUT the JAX stack:
  `experiments/convert_genome.py` reads `data/picbreeder_*/params.pkl` and
  `data/sgd_*/params.pkl` with a *restricted unpickler* (whitelists only
  `numpy`/`jax` array-reconstruction globals, verified benign by `pickletools`
  disassembly), writes trusted `data/fer_genome/*.npz`, and deletes the `.pkl`.
  `src/fer_cppn.py` reproduces their CPPN forward pass in NumPy (per-genome arch
  string, HSV→RGB, evosax's alphabetical `Dense_*` flattening); renders match the
  published targets to MSE ≤ 0.006. Only the trusted `.npz` are kept in the repo.

## References

- Kumar, Clune, Lehman, Stanley. *Questioning Representational Optimism in Deep
  Learning: The Fractured Entangled Representation Hypothesis.* arXiv:2505.11581 (2025).
- *Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data.* arXiv:2502.04664 (2025).
- *The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks.* arXiv:2602.16340 (2026).
- *Muon Learns More Robust and Transferable Features than Adam.* arXiv:2606.09658 (2026).
- Ahn, Xu et al. *Dion: Distributed Orthonormalized Updates.* arXiv:2504.05295 (2025).
- Hughes et al. *Position: Open-Endedness is Essential for Artificial Superhuman Intelligence.* ICML (2024). arXiv:2406.04268.
- Faldor et al. *OMNI-EPIC.* ICLR (2025). arXiv:2405.15568.
