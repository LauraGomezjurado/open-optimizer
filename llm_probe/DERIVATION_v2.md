# FOUNDATION AUDIT — the "Muon +40% recombinability" headline is a metric artifact

Motivated by the surgery result (spectral flatness is a correlate, not graftable):
before generalizing ANY optimizer, we re-ran the Muon-vs-AdamW frontier fixing the
two flaws that inflated the original claim. `neutral_frontier.py`, `run_fair_*.sh`;
data in `results_pod/fair/`.

**Two fixes vs the original comparison:**
1. **Tuned Muon LR.** The original ran Muon at AdamW's lr=3e-4 (crippled). LR
   ladder (seed 0): Muon optimum is **lr=0.005** (ar_ce 2.74, beats AdamW's 2.90).
   At lr=3e-4 Muon's effrank was suppressed; at 0.005 it is 0.48 vs AdamW 0.21 —
   the effrank gap is REAL and survives fair tuning (not an LR artifact).
2. **Neutral coherence judge.** Coherence is now neg-NLL under pretrained GPT-2
   (shares our exact GPT-2 BPE vocab), not the model's SELF-perplexity (circular).
   Plus per-gen token entropy logged for an entropy-matched read.

**Result (3 seeds each, matched loss ~2.75–2.90):**
- Raw novelty gap Muon−AdamW grows with perturbation: +0.019 / +0.023 / +0.027 /
  +0.042 across frac 0/0.25/0.5/1.0 — looks significant (up to ~7σ at frac 1.0).
- BUT Muon's neutral-judge perplexity is HIGHER at every frac (19.0 vs 26.1 at
  frac 1.0): the raw gap is still riding the coherence axis.
- **At MATCHED neutral-judge perplexity, Δnovelty = +0.014** (interp across the
  frontier). **Entropy-controlled regression: +0.029.** Both ≈ 1/20th of the
  implied "+40%" (which would be Δ≈+0.33 on a 0.82 baseline).

**Verdict: Muon's recombinability advantage is REAL but SMALL** (~0.01–0.03
novelty at matched coherence), not the order-of-magnitude effect the v2 Recombinator
was premised on. The original headline was inflated ~20× by comparing at unmatched
coherence (Muon generations were "more novel" largely because they were less
coherent). Two robust facts survive: (i) Muon's activation effrank is ~2.3× AdamW's
at matched capability; (ii) that buys a small-but-consistent matched-coherence
novelty edge. What does NOT survive: any claim large enough to justify building a
new optimizer to chase it. **Recommendation: do not build the v2 Recombinator on the
recombinability framing.** Either reframe around the robust effrank result itself,
or treat the small novelty edge as a modest confirm and stop over-claiming.

---

# PRIOR-ART CHECK: Xu honors thesis (Miikkulainen, UT Austin, May 2026)

Full read of "Measuring and Mitigating Fractured Entangled Representations in
Neural Networks" (nn.cs.utexas.edu/?xu:honorsthesis26). Same FER lineage as the
Kumar et al. paper we ported. What they did and how we differ:

**Their metrics** (CPPN, NEAT vs SGD): neuron-contribution effective-rank of the
contribution matrix C (Cᵢ(x,y)=‖f−f_¬i‖₂, ablate neuron i); per-pixel effective
neuron count eff=(ΣᵢCᵢ)²/ΣᵢCᵢ²; Gini of Cᵢ; DCI-style disentanglement. NEAT ~7–10
effective neurons vs SGD ~83–92 → order-of-magnitude gap. (Convergent with our
independently-built neuron-structure TV+HF metric — good external corroboration.)

**Their penalty:** Total Variation on activation maps,
  L_TV(h) = (1/N) Σᵢ (‖∇ₓhᵢ‖₁ + ‖∇_yhᵢ‖₁)   — spatial smoothness, bolted onto Adam.
On MNIST they apply TV to the *weight matrices reshaped as 28×28 filters*.

**Their results:** +11–14 pts test acc at 1 img/class (data-constrained); the
headline is continual learning — TV-pretrained nets stay PLASTIC (learn held-out
digit to 0.88–0.99) where baselines freeze (0.00–0.17), especially synergizing
with EWC. They explicitly note TV ≈ implicit sparsification / contractive-AE
Jacobian penalty, and that an input-Jacobian *rank* penalty (∂h/∂input high rank)
"proved stable but gave no consistent improvement over TV."

**How our work is DISTINCT (do NOT claim novelty on the explicit-penalty axis):**
1. THEIR lever is an explicit auxiliary LOSS on Adam; OURS is the optimizer's
   UPDATE-RULE GEOMETRY (Muon/Aurora spectral update) — no auxiliary term, no
   task-specific coefficient. "Does the optimizer's implicit bias move the FER
   axis by itself?" is a question they do not ask.
2. We CONFIRMED a mechanism (Muon keeps W's singular spectrum flat → activation
   rank compounds through depth) — they leave "what changes at the weight level"
   as explicit open future work (§5.3).
3. Their axis is SPATIAL smoothness (TV) — presupposes hidden units have spatial
   structure (CPPN pixels, reshaped MNIST filters). Ours is basis-free activation
   rank / independence — applies to transformer hidden states with no spatial grid.
4. Our recombinability bridge (novelty/coherence frontier) is a different
   downstream than their continual-learning/generalization.

**What their thesis TELLS US for the next lever:** they already tried the
input-Jacobian *rank* penalty (∂h/∂x high rank) and it did NOT beat TV. Combined
with OUR negative result on input-Σ_x whitening, both *second-order input-side*
levers have now failed independently (them: activation loss; us: preconditioner).
This is convergent evidence that the missing structure is HIGHER-ORDER
(independence, not decorrelation) — exactly the un-tried negentropy/ICA target.
Also: their continual-learning "plasticity" result is the strongest external
validation that the FER axis is real and downstream-meaningful — it de-risks the
whole program.

---

# NEXT LEVER built: higher-order independence regularizer (negentropy + cumulant)

anticollapse.py now has two HIGHER-ORDER surrogates (past decorrelation):
- `negentropy`: FastICA log-cosh contrast on ZCA-whitened features. Maximizing
  drives whitened marginals non-Gaussian => independent basis (breaks the O(m)
  rotation invariance covariance targets can't).
- `cumulant`: off-diagonal 4th-order cross-cumulant E[u_i²u_j²]→1 on whitened
  features. Direct 4th-order decorrelation.

SMOKE TEST (numpy mirror, math validated; torch version is line-by-line same):
| data | offdiag (2nd-ord) | negentropy | cumulant (4th-ord) |
|---|---|---|---|
| uniform, INDEPENDENT | -0.0002 | ~0 | -0.006 |
| gaussian | -0.0002 | 0.000 | -0.001 |
| **4th-order DEPENDENT** (u₂=u₁²−1, zero corr) | **-0.0003** | +0.0003 | **-0.120** |

KEY: the 4th-order-dependent data has ZERO linear correlation but strong variance
coupling. The second-order `offdiag` lever (same family as Xu's TV and our failed
whitened-polar) is BLIND to it (-0.0003 = same as independent). `cumulant` fires
20× harder (-0.120). This is direct evidence the higher-order lever captures
structure that EVERY second-order lever (ours + Xu's) is provably blind to.
negentropy correctly ≈0 on Gaussian. → justified to test on the transformer.

---

# HIGHER-ORDER LEVER — first result (seed 0, 2000 steps, lr default 3e-4)

CAVEAT: sweep ran at default lr=3e-4 (correct for AdamW, WRONG for Muon — the
tuned-Muon +40%-effrank headline is at Muon's own lr). So the muon rows here
(0.084) are undertrained Muon, NOT the headline; do not compare against them.

AdamW arms (correct lr) — effrank vs AdamW baseline 0.213:
| lever | ar_ce | mlp_effrank |
|---|---|---|
| adamw (ref) | 2.896 | 0.213 |
| adamw+negent 0.01 | 2.918 | 0.215 |
| adamw+negent 0.1 | 3.035 | 0.116 |
| adamw+negent 1.0 | 3.578 | 0.020 |
| adamw+cumul 0.01 | 2.909 | 0.177 |
| adamw+cumul 0.1 | 2.899 | 0.167 |
| adamw+cumul 1.0 | 2.910 | 0.155 |
| muon+negent 0.1 | 3.057 | 0.079 |
| muon+cumul 0.1 | 3.068 | 0.088 |

READING (honest): on the effrank FER-proxy the higher-order lever does NOT raise
it — it LOWERS it, monotonically with λ. The lever is clearly ACTING (effrank
moves smoothly with strength, not inert). BUT effrank is a SECOND-ORDER (rank)
metric and the lever targets INDEPENDENCE: an ICA/negentropy objective concentrates
variance into few independent components, so lower effrank is exactly what a
WORKING independence lever produces. → effrank is likely the wrong yardstick here.
Proper test = a DIFFERENT downstream property (recombinability novelty/coherence),
per the anticollapse anti-circularity rule. Checkpoints saved for it.

## 3-SEED FOLLOW-UP (seeds 0,1,2) — the seed-0 Pareto "win" does NOT replicate

ROBUST (std ≤0.008, all seeds agree):
- cumulant RELIABLY lowers effrank at matched capability: 0.218→0.166, monotone
  in λ. The lever unambiguously ACTS on its own (independence/rank) axis.
- capability-neutral: ar_ce 2.897→2.925 (~1%; only λ=1.0 drifts).

NOT ROBUST (this corrects my single-seed over-read):
- recombinability NOVELTY@1 gain flips sign on seed 1 for every λ (all_pos=False);
  mean Δ +0.003–0.007 but with a negative seed = WITHIN NOISE, not a Pareto win.
- perplexity improvement holds ONLY for λ=0.01 (3/3 seeds negative, mean −0.18);
  at λ=0.1,1.0 it flips sign across seeds.

The seed-0 "strictly Pareto-better" claim was a SINGLE-SEED ARTIFACT. With 3 seeds
the recombinability effect washes out for novelty and survives only weakly (λ=0.01)
for coherence.

### THREE levers now tested to beat Muon's ceiling — all fail to robustly win:
1. Input-Σ_x whitened-polar (our optimizer): dominated, 10 configs.
2. Xu's input-Jacobian RANK penalty: "no consistent improvement over TV" (theirs).
3. Higher-order independence (negentropy/cumulant): acts cleanly (effrank↓, robust)
   but does NOT robustly transfer to recombinability (3 seeds).

HONEST CONCLUSION: the covariance→independence axis is not the missing lever for
the open-endedness (recombinability) property. effrank and independence are the
SAME family of second/higher-order STATISTICAL-STRUCTURE targets; none of them
move the downstream open-endedness metric robustly. What DOES move recombinability
in our data remains Muon/Aurora's SPECTRAL update geometry (established earlier) —
a weight-space, not activation-statistics, lever. The next honest direction is
therefore NOT another activation-statistics regularizer but either (a) deeper
characterization of WHY the spectral update helps recombinability (weight-space
mechanism), or (b) consolidation — the negative results now tile the
activation-statistics design space cleanly.

---

# FULL DESIGN SWEEP verdict: NO whitened variant beats Muon/Aurora (10 configs)

Correcting the earlier "we only tried twice": ran a proper sweep over ALL the knobs
the critics named — α∈{0.25,0.5,1.0}, placement (up/both), φ-gap post-correction
(0.5,1.0), Aurora-base, and Aurora+whiten+postcorr (the full stack). 2000 steps,
lr0.01, seed0, same-budget baselines for fair reference:

| config | ar_ce | effrank | ‖W‖ |
|---|---|---|---|
| **muon** | 2.753 | 0.499 | 179 |
| **aurora** | 2.725 | **0.523** | 413 |
| whiten α0.25 up | 2.853 | 0.474 | 4018 |
| whiten α0.5 up | 2.863 | 0.413 | 3629 |
| whiten α1.0 up | 2.905 | 0.357 | 3165 |
| whiten α0.5 both | 3.011 | 0.300 | 5178 |
| whiten postcorr0.5 | 2.963 | 0.367 | 5665 |
| whiten postcorr1.0 | 2.986 | 0.364 | 5657 |
| aurora+whiten | 2.853 | 0.432 | 3692 |
| aurora+whiten+postcorr | 2.956 | 0.378 | 5664 |

**Every whitened variant is DOMINATED by plain Aurora and Muon** — worse loss AND
lower effrank AND 10–30× the weight norm. Systematic trends:
- **α: MORE whitening is MONOTONICALLY WORSE** (0.25→0.474, 0.5→0.413, 1.0→0.357).
  The best whitened point is the one closest to Muon (α→0). Input-whitening doesn't
  help; it hurts, proportionally to how much you apply it.
- **placement both < up** (0.300 vs 0.413): whitening the down-proj too is worse.
- **φ-gap postcorr HURTS** (0.367/0.364 < 0.413): the diagonal activation-magnitude
  correction the debate agent proposed does not recover rank; it lowers it.
- **Aurora+whiten (0.432) < Aurora alone (0.523):** adding whitening to Aurora
  strictly degrades it. The stack (full) is worse still (0.378).

**Honest verdict (now on 10 configs, not 2): the input-Σ_x-whitening program does
NOT produce a better optimizer — across every knob it underperforms the spectral
baselines it was meant to improve.** This is a real negative result for the derived
optimizer. It does NOT mean "Muon is near-optimal in general" (we know Muon is far
from the UFR/recombinability ceiling) — it means *this particular lever* (input
covariance whitening) is the wrong one; the large remaining gap to the ceiling must
be closed by something other than reconditioning the update's input metric. The
info-theoretic higher-order targets (negentropy/likelihood-manifold) remain
un-implemented and are the honest next candidate — a covariance-based lever is now
shown insufficient, consistent with the critics' "second-order proxies miss
higher-order dependence" warning.

---

# ✅ MECHANISM CONFIRMED (Path B): Muon's activation-rank gap compounds with depth

Per-layer activation effrank (scale-invariant correlation metric), from checkpoints:

| layer | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| AdamW | 0.228 | 0.398 | 0.428 | 0.443 | 0.467 | 0.367 |
| Muon | 0.323 | 0.537 | 0.617 | 0.637 | 0.616 | 0.514 |
| **Muon−AdamW gap** | +0.096 | +0.139 | +0.189 | +0.194 | +0.149 | +0.147 |

Weight-spectrum effrank (flatness of W's singular values):
| layer | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| AdamW | 0.937 | 0.908 | 0.909 | 0.902 | 0.904 | 0.909 |
| Muon | 0.938 | 0.953 | 0.955 | 0.956 | 0.958 | 0.959 |

**The derivation's mechanism chain is empirically confirmed end-to-end:**
1. Muon keeps W's singular values FLATTER than AdamW (weight-effrank rises 0.94→0.96
   with depth for Muon; FALLS 0.94→0.90 for AdamW) — the derivation's premise.
2. ⇒ activations stay higher-rank, and the gap COMPOUNDS with depth (+0.10→+0.19
   over layers 0→3), exactly the derivation's prediction and matching the earlier
   depth-scan. (Gap tapers at the last 2 layers — expected: the final layers are
   shaped by the readout/output task, breaking pure compounding.)

This is the real, derived-AND-measured scientific result of the whole thread:
**a mechanism for why spectral (Muon) optimizers produce more factored
representations — flat update spectrum → flat weight spectrum → high-rank
activations, compounding through depth.** It stands independent of whether any new
optimizer beats Muon.

---

# PATH A RETRY (fixed: renorm + up-proj-only + LR-sweep): competitive, does NOT beat Muon

The v1 whitened-polar failure WAS the unit-norm bug (the trailing Σ^{−α} broke
polar's unit-spectral-norm, so the step was uncontrolled → runaway ‖W‖=4055,
effrank 0.22). Fixed by renormalizing O to unit RMS after whitening, restricting to
tall up-proj matrices, and sweeping LR:

| whitened lr | ar_ce | effrank | ‖W‖ |
|---|---|---|---|
| 0.005 | 2.756 | 0.425 | 2501 |
| 0.01 | 2.798 | 0.473 | 3773 |
| 0.02 | 2.937 | 0.452 | 5568 |
| 0.04 | 3.221 | 0.318 | 8027 |
| **Muon (ref)** | **2.71** | **0.49** | ~440 |

The fix rescued it from broken (0.22) to COMPETITIVE (0.43–0.47 effrank at
near-matched loss) — confirming the bug diagnosis. **But it does NOT beat Muon:**
at best matched-loss (lr0.005) it trails Muon on both loss (2.76 vs 2.71) and
effrank (0.43 vs 0.49), and it carries 5–20× the weight norm and ~4× the compute
(per-layer eigendecomp). 

**Verdict on the derived optimizer:** input-covariance whitening is a real, working
mechanism (it lifts effrank well above AdamW's 0.34) but it is NOT a free
improvement over Muon's spectral update — Muon already achieves higher activation
rank more cheaply. The principled "generalize Muon" program, as instantiated, does
not yield a better optimizer; it reproduces (slightly under) Muon at higher cost.
Honest conclusion: **Muon appears near-optimal for this objective; the value is in
UNDERSTANDING it (Path B mechanism), not replacing it.**

---

# (superseded) PATH 1 v1: the input-whitened polar (as built) UNDERPERFORMS Muon

Built and trained the derived optimizer (O*=polar(M Σ_x^{−1/2})Σ_x^{−1/2}, α=0.5,
lr=0.02, 2 seeds). Result:

| optimizer | ar_ce | mlp_effrank | ‖W‖ |
|---|---|---|---|
| AdamW | 2.75 | 0.34 | 92 |
| Muon | 2.71 | 0.49 | ~440 |
| Aurora | 2.72 | 0.50 | — |
| **Whitened-polar** | **2.97** | **0.22** | **4055** |

The derived optimizer did WORSE on both loss (2.97 vs 2.71) and effrank (0.22 —
below even AdamW), with an enormous weight norm. So the Σ_x-whitening theory, as
implemented, does NOT deliver its predicted benefit; it underperforms the thing it
was meant to generalize.

Honest caveats (why this may be tuning, not concept): (1) α=0.5, lr=0.02 untuned —
Σ_x^{−1/2} rescales the effective step, so the LR is likely wrong (the runaway
‖W‖=4055 suggests an unstable/too-large effective step); (2) applying Σ_x^{−1/2} to
BOTH mlp linears including the down-proj (in=d_ff) may be wrong — the derivation was
about the up-proj activations; (3) per-layer eigendecomp made it ~4× slower than
Muon (a practical cost even if it worked). But at face value: **the elegant derived
optimizer failed its first empirical test.** The theory that "target Σ_x-whitening ⇒
higher activation rank" is not confirmed — possibly because whitening the INPUT is
not the same as raising the rank of the post-activation OUTPUT (φ's gating breaks
the pass-through the derivation assumed; the debate agent flagged exactly this as a
caveat). Next if pursued: LR-sweep + up-proj-only + the diagonal post-correction the
agent said was needed to close the φ-gating gap — but the honest status is the
derived optimizer is NOT yet a win, and Muon/Aurora remain the empirical best.

---

# ✅ CORRECTED THEORY (post-gate rebuild): input-whitened polar, target Σ_a not JᵀJ

Two debate agents, informed by the gate data, converged on a corrected optimizer
that fixes the category error AND matches the empirics. Pending the artifact-check
on Gate 2 and Gate 1b's weight-norm result, this is the new candidate.

**GATE 2b RESULT (artifact-corrected: log-softmax probe + 2 seeds).** The original
Gate 2 "clean anti-correlation" was PARTLY the confidence/weight-norm artifact the
critic predicted. Corrected numbers (L_effrank_frac):
  seed0: aurora 0.051 > adamw 0.049 > muon 0.047
  seed1: aurora 0.064 > adamw 0.050 > muon 0.041
So leverage-isotropy now neither cleanly tracks NOR anti-tracks recombinability —
aurora (best recomb) is highest, muon (also good) is lowest, adamw (worst) middling.
**Verdict: L=JᵀJ isotropy is simply NOT PREDICTIVE of recombinability** (not the
dramatic inversion Gate 2 suggested, but also not a valid target). Meanwhile
activation-covariance effrank DID cleanly track recomb (aurora≈muon>adamw). So the
redirect stands — target activation statistics, not leverage — but for the softer
reason "leverage isn't predictive," not "leverage is inverted." The input-whitened
polar (Σ_x) remains the right candidate: cheap, well-posed, Muon-generalizing, and
its motivation (target forward activation rank) survives; only the strength of the
anti-JᵀJ argument was downgraded.

**Why leverage-isotropy anti-correlates (pigeonhole mechanism, now understood).**
In the activation eigenbasis, L_{ij}=⟨Ju_i,Ju_j⟩. rank(L)≤rank(J). If activation
effective rank r_a > rank(J), the r_a images {Ju_i} live in a ≤rank(J)-dim space,
so they CANNOT be mutually orthogonal with equal norm ⇒ high activation rank
*mechanically forces* anisotropic, correlated leverage (Muon: L-effrank 0.027,
off-diag 0.247). Conversely a collapsed rep (few active dirs, AdamW) fits inside
the output-rank budget ⇒ readout can place them orthogonally+equal ⇒ L≈αI "for
free." So **leverage-isotropy L∝I is a FLATNESS / uninformativeness signature**
(max-entropy, O(m)-rotation-invariant, certifies NO privileged direction — a
random high-gain net attains it). Coherent LMs must route variance UNEQUALLY
(semantic dirs get large ‖Ju_i‖), which is exactly the heterogeneous L-spectrum
that lowers isotropy. The anti-correlation is EXPECTED, not paradoxical. This also
retro-explains the toy: pushing L→I pushes toward the collapsed regime.

**The corrected, well-posed target.** Maximize effective rank of the ACTIVATION
covariance Σ_a = E[φ(Wx)φ(Wx)ᵀ] at fixed loss. Unlike JᵀJ (built from downstream
weights → not a function of the update O → category error), Σ_a depends on the
very W being updated: ∂Σ_a/∂W is real, so the lever is legitimate.

**The corrected optimizer (input-whitened polar).**
    O* = polar(M · Σ_x^{−1/2}) · Σ_x^{−1/2},   knob P = Σ_x^{α}  (α=0 ⇒ Muon)
where Σ_x = E[xxᵀ] is the INPUT covariance to the layer (n×n, small). This
right-whitens by input covariance, equalizing Σ_z = WΣ_xWᵀ, which passes through
φ (ReLU/GELU preserve nonzero-variance directions) to raise effrank(Σ_a). It is
the exact MIRROR of the rejected whitened-polar: right-whiten by forward Σ_x
(feasible, cheap) instead of left-condition by output L=JᵀJ (infeasible, wrong).
- **Cost: trivial.** Σ_x is n×n; one eigendecomp of a cheap EMA of xxᵀ. NO vjps,
  NO Hutchinson, NO rank floor — all the estimator problems of the L-version vanish.
- Damping Σ_x ← Σ_x + ε(trΣ_x/n)I. Diagonal variant O*≈polar(M diag(Σ_x)^{−1/2}).
- Residual-gap fix: φ's gating means effrank(Σ_z) upper-bounds effrank(Σ_a); a
  diagonal post-correction on E[a_i²] (Aurora-style row-norm equalization, now
  derived against Σ_a instead of n/m) closes it.

**Muon mechanism, DERIVED (matches our depth-scan).** polar keeps W's singular
values flat ⇒ Σ_z=WΣ_xWᵀ stays isotropic ⇒ high effrank; the conditioning
compounds MULTIPLICATIVELY over depth (x^{(l+1)}=a^{(l)}), so the Muon−AdamW
effrank gap GROWS with depth — exactly the depth-scan result (+0.15→+0.29 over
L=2→12). This is the first derivation that predicts our own empirical mechanism.

Caveat: exact in the linear-Gaussian, m>rank(J) regime the checkpoints occupy;
sign can weaken if r_a ≤ rank(J). Must survive the Gate-2 artifact-check + Gate 1b.

---

# ⚠️ GATE 1 VERDICT: effrank gap is real across the loss curve, BUT weight-norm-confounded

Multi-LR sweep (4 LRs each), effrank-vs-loss curve pooled per optimizer, +weight-norm:

| ar_ce bin | AdamW effrank | AdamW ‖W‖ | Muon effrank | Muon ‖W‖ |
|---|---|---|---|---|
| [2.6,2.8) | 0.362 | 124 | 0.504 | 441 |
| [2.8,3.0) | 0.274 | 106 | 0.409 | 746 |
| [3.0,3.3) | 0.152 | 92 | 0.315 | 519 |

- **PASS half:** Muon's effrank-vs-loss curve sits ABOVE AdamW's at *every*
  overlapping loss bin — so "+40%" is NOT an artifact of comparing at one lucky
  matched-loss point. The geometry effect is real along the whole curve.
- **FAIL half (confound CONFIRMED):** Muon runs at **3–7× larger weight norm**
  (441–746 vs 85–124). We cannot yet separate "Muon selects higher-effrank
  geometry" from "Muon operates at larger weight scale and larger-norm nets have
  higher effrank." The falsification critic's #1 confound is real and unresolved.
- **Required follow-up before any headline:** match weight-norm (weight decay tuned
  per-optimizer so ‖W‖ coincides, OR normalize activations before effrank) and
  re-check the gap. Until then the effrank result is "real along the curve but
  possibly a weight-scale effect."

## GATE 1b VERDICT (weight-norm matched via Muon weight-decay sweep): CONFOUND IS LOAD-BEARING

Held AdamW fixed (‖W‖=92, effrank=0.289, ce=2.77); swept Muon weight-decay to pull
its norm down:

| run | ar_ce | effrank | ‖W‖ |
|---|---|---|---|
| AdamW (wd0.1) | 2.77 | 0.289 | 92 |
| Muon wd0.1 | 2.83 | 0.448 | 244 |
| Muon wd0.3 | 3.10 | 0.115 | 122 |
| Muon wd0.8 | 3.55 | **0.046** | 62 |
| Muon wd2.0 | 4.16 | 0.031 | 34 |

**As Muon's ‖W‖ is forced toward AdamW's, its effrank COLLAPSES (0.448→0.046) and
its loss degrades (2.83→4.16).** At the norm-closest point (‖W‖ 62 vs 92), Muon
effrank 0.046 vs AdamW 0.289 — the gap **REVERSES**. So Muon cannot hold both
matched-loss AND matched-norm; its high effrank is substantially **entangled with
its larger weight scale**, not a clean norm-independent geometry effect.

**Honest downgrade:** the "+40% effrank at matched loss" headline is confounded —
at matched loss Muon runs at 3× the weight norm, and removing that norm advantage
removes (reverses) the effrank gap. The remaining defensible statement: *at its
natural (large-norm) operating point, Muon reaches higher effrank than AdamW at
matched loss* — but "the optimizer geometry, independent of scale, selects higher
effrank" is NOT supported; it may be a scale effect.
Caveat the other way: weight decay changes the optimization too (loss rises), so
this isn't a clean scale-only isolation either — it shows the two can't be
decoupled cheaply, which itself weakens the clean-geometry claim. A cleaner
isolation (e.g. compare at matched ‖W‖ by rescaling activations before the effrank
measurement, no retraining) is the next check.

**Crucial clarification (checked the metric):** our `effrank_frac` uses the
CORRELATION matrix — it standardizes each feature to unit variance before the
spectrum, so it is ALREADY invariant to per-feature scale and largely to weight
norm. Therefore Gate 1b's effrank collapse is NOT a "bigger-norm→bigger-effrank"
measurement artifact. It means something realer: **Muon's high effrank and its
large weight norm are two faces of ONE mechanism** — forcing the norm down (via wd)
genuinely collapses the representation AND hurts loss simultaneously. So "matched
loss AND matched norm" may be *infeasible* for Muon (the two are coupled), which is
itself the finding. This REHABILITATES the corrected theory: the Σ_x-whitening view
predicts exactly this — spreading activation variance (high effrank) requires the
weights to have the spectral room to do so (large-ish norm/flat singular values);
you cannot decouple "many active directions" from "the weight scale that hosts
them." The clean claim becomes: *Muon's geometry couples norm-growth and
rank-growth; AdamW's does not* — a geometry difference, just not a norm-independent
one. The scale-vs-geometry dichotomy was itself too clean.

## PATH 2 VERDICT (no-retrain scale isolation): the gap is STRUCTURAL, not scale

Measured effrank under three normalizations on the natural checkpoints (mid MLP):

| opt | er_cov (scale-SENSITIVE) | er_corr (scale-INVARIANT) | er_l2 (per-token L2-normed) |
|---|---|---|---|
| AdamW | 0.406 | 0.443 | 0.430 |
| Muon | 0.516 | **0.637** | 0.519 |
| Aurora | 0.519 | 0.623 | 0.530 |

If Muon's advantage were a weight-SCALE artifact it would appear in er_cov and
VANISH under per-feature standardization (er_corr) and per-token L2-norm (er_l2).
The opposite holds: the gap is present and **largest in the scale-INVARIANT
correlation metric** (Muon 0.637 vs AdamW 0.443, +0.19), and survives L2-norm.
**Verdict: Muon genuinely produces more independent activation directions than
AdamW — structural, not scale.** This RESOLVES the Gate-1b confusion: 1b's collapse
was weight-decay destroying the representation (a co-training effect that also
raised loss), NOT effrank tracking norm. At the natural operating point, on a
fully scale-invariant metric, the geometry advantage is real. The +40%-recombinability
headline is rehabilitated as a structural effect — Gate 1b's scale-confound
concern is answered by measuring scale-invariantly rather than by de-norming via wd.

---

# ⚠️ GATE 2 VERDICT (run after v2): the JᵀJ target is EMPIRICALLY WRONG

Before building anything we ran the gate the critics demanded: does isotropy of the
output-leverage Gram L=JᵀJ (the quantity this derivation's whitened-polar optimizer
targets) actually PREDICT the measured recombinability? On the matched checkpoints
(adamw/muon/aurora, seed 0, mid MLP layer, k=32 Hutchinson probes):

| opt | recomb (nov@f2, ppl@f2) | ACTIVATION effrank | LEVERAGE L_effrank_frac | L_offdiag |
|---|---|---|---|---|
| adamw | 0.944, 13.3 (worst) | 0.342 | **0.043 (most isotropic!)** | 0.169 |
| muon | 0.967, 9.5 | 0.495 | 0.027 | 0.247 |
| aurora | 0.973, 8.8 (best) | 0.496 | 0.032 | 0.226 |

- **ACTIVATION effective rank** (covariance of hidden activations) orders
  aurora>muon>adamw — **perfectly tracks recombinability.** ✓
- **LEVERAGE isotropy** (L=JᵀJ, what the whitened-polar optimizer below targets)
  orders adamw>aurora>muon — **ANTI-tracks recombinability.** ✗ AdamW (worst
  recombinability) has the MOST isotropic leverage.

**Verdict: the whitened-polar / L^{-1/2} derivation targets the WRONG object.**
Making JᵀJ isotropic would push toward AdamW-like reps — opposite of the goal. The
critics (definitions, info-theory, open-endedness) predicted exactly this ("JᵀJ∝I
neither necessary nor sufficient"); the gate turns "insufficient" into "inverted."

**What this redirects us to:** the predictive quantity is *activation-space*
effective rank (how many independent directions the representation USES), not
output-leverage isotropy (how uniformly units drive logits). A principled
"recombinability optimizer" should therefore bias toward high **activation**
covariance rank — which is much closer to what Muon *already does implicitly*, and
suggests the lever is on the forward activation statistics, not the Jacobian. The
whitened-polar math below is preserved as a rigorous exercise, but its target L
must be replaced (candidate: precondition by the activation-covariance, not JᵀJ).
Caveats: n=1 seed, k=32 low-rank L estimate, one layer — but the anti-correlation
is large and directionally unambiguous. Re-run with more seeds/probes before final.

---

# Recombinator derivation v2 — hardened by 10 adversarial-mathematical critics

v1 (DERIVATION.md) had the right instinct and three real errors. Eight independent
critics (Riemannian geometry, matrix analysis, optimization theory, norm-duality /
implicit bias, two-timescale convergence, information theory, open-endedness
theory, adversarial-empirical) converged on the SAME corrected object from
different directions. This is v2.

## The single biggest correction: it's a METRIC, not a manifold projection

v1 framed the new constraint as "project the update O onto {O : JᵀJ ∝ I} via
damped alternating projection." **This is a category error** (flagged independently
by the geometry, optimization, and norm-duality critics): J = ∂output/∂hidden
depends on the downstream weights and data, NOT on the update O. So {JᵀJ∝I} is not
a feasible set for O — it constrains nothing the optimizer controls.

**Correct object.** Fix L = JᵀJ ⪰ 0 (m×m SPD, current step — "frozen" within the
step). The genuine lever is orthonormality of the update *in the L-metric*:

    O* = argmin_O ⟨O, −M⟩   s.t.  Oᵀ L O = I_n

Lagrangian ℒ = −tr(OᵀM) + tr(Λ(OᵀLO − I)); stationarity gives the CLOSED FORM

    O* = L^{−1/2} · polar( L^{−1/2} M )              (the "whitened polar")

Derived independently three ways: (a) as the KKT solution above (optimization
critic); (b) as the exact fixed point Aurora's damped loop chases (convergence
critic); (c) as the **duality map of the L-weighted spectral norm** ‖A‖_L =
‖L^{−1/2}A‖₂ (implicit-bias critic). All three agree. Properties:
- L ∝ I  ⇒  O* = polar(M) = **Muon** (so Muon is the α=0 special case).
- One polar solve. **No alternating projection, no inner loop, no damping β, no
  convergence proof owed** — because L is frozen within the step.
- The relaxed family **P = L^α, α∈[0,1]** gives ONE derived knob: α=0 Muon,
  α=1 full leverage-whitening. (Replaces v1's infeasible hard constraint.)

This is strictly better than v1: cleaner, cheaper, and it makes the method
"steepest descent under a norm that encodes the output geometry" — the same
principled frame as Muon itself.

## Correction 2: the impossibility result was literally false; the real one is deeper

v1 claimed "OOᵀ can't be diagonal for tall O (rank n<m)." **False** (matrix critic):
a rank-n projector CAN be diagonal (n ones, m−n zeros). The true statements:
- **Trade-off theorem.** For O ∈ St(m,n), m>n: OOᵀ diagonal ⇒ exactly m−n rows are
  ZERO (dead). So "all rows alive AND mutually orthogonal" is impossible — you can
  decorrelate weight-rows only by killing m−n neurons. Aurora's equal-diagonal n/m
  is the correct trade (all alive, approximately decorrelated).
- **The deeper, unflagged impossibility (matrix + info critics).** L = JᵀJ has
  rank ≤ rank(J) ≤ min(d_out·batch, m). If effective output rank r < m = d_ff,
  then **L ∝ I_m is literally unattainable by ANY optimizer** (rank floor). The
  target must be restricted to the reachable subspace: flatten the *nonzero*
  spectrum of L (P_S L P_S ∝ P_S on S = row(J)), not all of R^m. This is why the
  right knob is L^α (whitens the live spectrum) not "L→I" (impossible).
- **The honest reason weight space is insufficient** is NOT a DOF shortage but
  SEMANTIC: weight-row correlation (OOᵀ) ≠ output-effect correlation (JᵀJ). Two
  weight-orthogonal rows can map to the same output direction downstream. Hence
  L must enter — confirming Route B, for the right reason.

## Correction 3: the target is a second-order shadow; the honest claim is narrow

Multiple critics (definitions, information-theory, open-endedness) independently
hit the same wall: **JᵀJ ∝ I is neither necessary nor sufficient for the measured
recombinability.**
- **Not sufficient**: JᵀJ is first-order/Gaussian/Euclidean/single-step. Isotropy
  with high gain maximizes novelty AND destroys coherence — the two frontier axes
  share the ‖Jδ‖ knob, so no single L-target encodes the tradeoff. Uncorrelated
  (JᵀJ off-diag 0) ≠ independent (perturbations can be higher-order dependent).
- **Not necessary**: the frontier only needs *many independent coherent*
  directions, not *equal* leverage in *every* direction.
- **Linearization gap**: J is a local Jacobian; recombinability is a global,
  nonlinear, multi-step rollout property. Local isotropy is at most a *hygiene*
  condition.
- **Rotation-invariance gap** (info critic): JᵀJ∝I is O(m)-invariant, but
  factoredness names a *privileged basis*. A rotation-invariant target cannot
  certify a factored basis; a random high-rank net maximizes it with zero reusable
  structure.
- **Category concern** (open-endedness critic): Stanley's stepping stone is
  SEMANTIC reuse (one cause, many aligned effects) — which shows up as CORRELATED
  rows, exactly what isotropy penalizes. Decorrelation can punish reuse.

**The honest, defensible claim** (what all critics agreed survives):
> Rank/leverage-equalizing updates (Muon/Aurora/whitened-polar) are an
> **anti-collapse conditioning** that prevents the representation collapsing onto
> few output directions, raising *yield-per-perturbation* for a downstream loop.
> This is a **necessary hygiene condition and efficiency multiplier**, NOT a
> source or measure of open-endedness (which lives in the outer loop's objective/
> selection/environment). "Recombinability" as we measure it (perturbation
> frontier) is tracked by this only in a Gaussian-linear limit.

## What the critics say to BUILD (the revised program)

1. **The optimizer**: whitened polar O* = L^{−1/2}·polar(L^{−1/2}M), knob P=L^α.
   Estimator critic verdict on getting L per step (J=∂logits/∂hidden, m=d_ff):
   - "one vjp gives L" is **FALSE** — a vjp gives Jᵀv for one probe. Full L needs
     ~m vjps/step (≈1000× a training step) AND Hutchinson L̂ is rank-k singular for
     k<m, so L^{−1/2} is ill-posed exactly where used. **Full m×m L: NOT viable.**
   - **Diagonal-L: viable, cheap** (Aurora-like): diag(L)_i=‖∂y/∂h_i‖² via k≈4–16
     Hutchinson probes; fixes dead directions (mode a), misses redundancy (mode b).
   - **Low-rank L = εI + UΛUᵀ, r≈16–32: VIABLE, recommended.** Top-r leverage dirs
     by randomized subspace iteration on J (r vjps + r jvps); Woodbury inverse-sqrt
     (εI+UΛUᵀ)^{−1/2} = ε^{−1/2}[I − U(I−(I+Λ/ε)^{−1/2})Uᵀ], O(r) vjps/step. This
     captures the leading off-diagonal (redundancy) Aurora's diagonal-only misses —
     it is the concrete "goes beyond Aurora" mechanism. Freeze L̂ across the polar
     iteration (slow-drift). Mandatory damping ε=δ·tr(L)/m, δ≈1e-2–1e-1.
2. **The gating experiment (do BEFORE building)**: compute L=JᵀJ on the existing
   matched checkpoints and test whether **isotropy of L actually correlates with
   the measured recombinability frontier**. If the quantity the optimizer targets
   doesn't predict the quantity we measure, the derivation targets the wrong thing.
3. **Beyond second order** (info critic): the rigorous target is total-correlation /
   mutual-information independence of perturbation channels + a likelihood/score
   (manifold-preservation) term for coherence + a basis-selection (ICA-style
   negentropy) term to break rotation-invariance. JᵀJ∝I is the Gaussian shadow of
   this. A truly principled optimizer would precondition with these, not just L.
4. **Empirical hardening** (falsification critic, ranked): (i) the "+40% at matched
   loss" rests on two hand-tuned LRs and loss isn't exactly matched (2.71 vs 2.75)
   — plot the WHOLE effrank-vs-loss curve, not one point, and match weight-norms;
   (ii) recombinability proxies are partly circular (self-perplexity coherence;
   token-Jaccard novelty gameable by entropy) — use a neutral judge model +
   entropy-matched novelty; (iii) seed-up (2→5) the frontier and depth-scan
   endpoints. Build the Recombinator ONLY if these survive.

## CAUSAL TEST — is Muon's flat WEIGHT spectrum the transplantable cause? NO.

After three activation-statistics levers failed to beat Muon, the last standing
correlate was Muon/Aurora's flat weight singular spectrum. Correlation ≠ cause, so
we ran the `do()`-operation directly (`spectrum_surgery.py`): load a *trained*
checkpoint, edit each MLP up/down-proj spectrum S → Sᵖ (U,V untouched, Frobenius
norm preserved → shape not scale), NO retraining, then remeasure. p<1 flattens
(toward Muon), p>1 sharpens (toward Adam), p=1 is an exact identity gate.

Results (seeds 0,1,2 AdamW + seed-0 Muon; `results_pod/surgery/*.json`):

1. **Identity gate passes** — p=1.0 is the capability minimum for every model
   (adamw ar_ce 2.89–2.90; muon 3.06), confirming the harness reproduces the
   untouched net exactly.
2. **The trained spectrum is optimal; any edit hurts.** Flattening AdamW further
   (p=0.5) raises ar_ce 2.90→3.25–3.29; sharpening (p=2.0) wrecks it (→4.6–5.5,
   ppl blows up). The U/V bases and the rest of the network are CO-ADAPTED to the
   spectrum they trained with — it is not a free-floating graftable knob.
3. **Flattening an AdamW model does NOT buy recombinability at matched coherence.**
   The flat-spectrum-is-causal hypothesis predicts novelty↑ under p<1. Measured
   Δnovelty@frac0.5 (p0.5 vs p1): −0.008, −0.025, −0.041 across the three AdamW
   seeds — flat-to-NEGATIVE, and ppl worsens each time. The only novelty gain is at
   p≥1.5, but that is the model breaking (ppl 4→17–27 = incoherent), not free
   novelty.
4. **Sharpening the Muon model toward Adam barely dents it** (Δnov −0.01 at equal
   ppl, p1→1.5) — inside the cross-seed noise band. Its advantage does not sit in
   the final singular values.

**Verdict: Muon's flat weight spectrum is a CORRELATE, not a transplantable cause.**
The recombinability advantage is a property of the co-adapted whole produced by the
training *trajectory* (orthogonalized updates shaping bases + spectrum + downstream
net jointly), not a static spectral shape you can paste onto a differently-trained
model. This closes the "just flatten the weights" shortcut: to get Muon's frontier
you must *train* with the trajectory, which is exactly what the v2 conditioner below
proposes to generalize. (Limitation: Muon replication is seed-0 only — the
sharpen-Muon direction rests on one seed; the flatten-Adam direction, which carries
the verdict, is 3-seed robust.)

## One-line v2 thesis

Muon = steepest descent in the spectral norm; the principled generalization is
**steepest descent in an output-leverage-weighted spectral norm**, O* =
L^{−1/2}polar(L^{−1/2}M) with L=JᵀJ frozen per step (Muon at α=0) — a single
duality-map solve, not a manifold projection. It provably equalizes the update's
*output* leverage on the reachable subspace (the L∝I-everywhere target is
rank-obstructed and must be restricted to row(J)). This is a rigorous
**anti-collapse conditioner**; calling it "recombinability/open-endedness" is only
justified in the Gaussian-linear limit and must be earned by the L-vs-frontier
correlation gate and a neutral-judge recombinability test — not assumed.
