"""
The optimizer-geometry ladder.

Every method here is a *normalized steepest-descent* rule: given a (momentum-
smoothed) gradient matrix G for a weight, we produce a unit-norm step direction
in some geometry, then take a step of fixed size `lr`. Normalizing every method
to a unit step makes the comparison about *direction/geometry*, not magnitude --
which is the whole point (implicit bias lives in the direction).

Geometries, from coordinatewise to whole-matrix spectral:
  - sign   : sign(G)            -> steepest descent in the l-infinity norm
                                   (this is the implicit-bias limit of Adam)
  - adam   : diagonal-preconditioned (adaptive coordinatewise) reference point
  - spectral(p): SVD G = U S V^T, replace S by S**p (renormalized).
       p = 1.0 -> direction of G itself      = l2 / Frobenius steepest descent
       p = 0.0 -> U V^T (orthogonalized)      = spectral-norm steepest descent = Muon
       0<p<1   -> a continuous "spectrum-flattening" knob between them.

The spectral(p) family is the experimental instrument: sweeping p from 1 -> 0
walks the update geometry from Euclidean toward orthonormal (Muon/Dion), giving
the *dose-response curve* rather than a binary A/B.
"""
import numpy as np


def _unit_frob(M, eps=1e-12):
    n = np.linalg.norm(M)
    return M / (n + eps)


class SteepestDescent:
    def __init__(self, params, kind="spectral", p=0.0, lr=0.05, momentum=0.9,
                 adam_b2=0.999, eps=1e-8, wd=0.0):
        self.kind = kind
        self.p = p
        self.lr = lr
        self.momentum = momentum
        self.b2 = adam_b2
        self.eps = eps
        self.wd = wd                                   # decoupled weight decay
        self.m = [np.zeros_like(w) for w in params]   # 1st moment / momentum
        self.v = [np.zeros_like(w) for w in params]   # 2nd moment (adam only)
        self.t = 0

    def _direction(self, G):
        """Unit-norm (Frobenius) step direction for a single weight matrix."""
        if self.kind == "sign":
            D = np.sign(G)
            return _unit_frob(D)
        if self.kind == "spectral":
            # economy SVD; all our matrices are small
            U, S, Vt = np.linalg.svd(G, full_matrices=False)
            if self.p == 0.0:
                Snew = np.ones_like(S)
            else:
                Snew = S ** self.p
            D = (U * Snew) @ Vt
            return _unit_frob(D)
        raise ValueError(self.kind)

    def _adam_dir(self, G, i):
        """coord-adaptive direction with a knob q: v-preconditioner raised to q.
        q=1 -> full per-entry rescaling (Adam); q=0 -> no rescaling (isotropic).
        Sweeping q is the COORDINATEWISE analog of the spectral p knob: it
        localizes fracture to per-entry rescaling rather than 'spectralness'."""
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (G * G)
        vhat = self.v[i] / (1 - self.b2 ** self.t)
        q = 1.0 if self.p is None else self.p
        precond = (np.sqrt(vhat) + self.eps) ** q
        return _unit_frob(G / precond)

    def step(self, params, grads):
        self.t += 1
        for i, (w, g) in enumerate(zip(params, grads)):
            # momentum on the raw gradient (Muon/Adam style)
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * g
            mhat = self.m[i] / (1 - self.momentum ** self.t)
            if self.kind == "adam":
                D = self._adam_dir(mhat, i)
            elif self.kind == "scalar_adam":
                # basis-agnostic adaptivity: ONE scalar 2nd-moment for the whole
                # matrix (RMS of g), so adaptive step size but no per-entry
                # rescaling -> separates 'adaptivity' from 'per-coordinate-ness'.
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * float(np.mean(g * g))
                vhat = self.v[i] / (1 - self.b2 ** self.t)
                D = _unit_frob(mhat / (np.sqrt(vhat) + self.eps))
            else:
                D = self._direction(mhat)
            params[i] = w - self.lr * D
            if self.wd:
                params[i] = params[i] - self.lr * self.wd * w
        return params


def build_ladder():
    """The set of geometries to sweep.

    Entries are (label, kind, p, opts?). The first 7 are the ORIGINAL ladder
    (labels frozen so prior results stay valid). The appended rungs isolate the
    real lever (confound C1): 'spectral' is not the operative property -- absence
    of per-COORDINATE rescaling is. So we add:
      - a COORDINATEWISE dose-response knob q (adam with p=q): q:1->0 walks from
        full per-entry rescaling (Adam) to none (isotropic). This is the direct
        analog of the spectral p knob and localizes fracture to per-entry rescaling.
      - ScalarAdam: adaptive step size but ONE scalar per matrix -> basis-agnostic
        adaptivity, separating 'adaptivity' from 'per-coordinate-ness'.
      - Adam+wd: weight-decay saturation control (confound C2/S3) -- if matched-
        saturation Adam still fractures, saturation is not the mechanism.
    """
    return [
        # label,                   kind,          p,     opts
        ("Adam (adaptive coord.)", "adam",        None),
        ("SignSGD (L-inf)",        "sign",        None),
        ("Spectral p=1.0 (L2/GD)", "spectral",    1.0),
        ("Spectral p=0.75",        "spectral",    0.75),
        ("Spectral p=0.5",         "spectral",    0.5),
        ("Spectral p=0.25",        "spectral",    0.25),
        ("Spectral p=0.0 (Muon)",  "spectral",    0.0),
        # --- appended diagnostic rungs (C1 / S3 / S4) ---
        ("Adam-q=0.75 (coord knob)", "adam",      0.75),
        ("Adam-q=0.5 (coord knob)",  "adam",      0.5),
        ("Adam-q=0.25 (coord knob)", "adam",      0.25),
        ("ScalarAdam (basis-agnostic)", "scalar_adam", None),
        ("Adam+wd (sat-control)",   "adam",       None,  {"wd": 0.02}),
        ("Adam+wd2 (sat-control)",  "adam",       None,  {"wd": 0.05}),
    ]
