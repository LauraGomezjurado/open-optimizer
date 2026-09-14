"""
Minimal CPPN (Compositional Pattern Producing Network) in pure NumPy, with
hand-derived reverse-mode gradients for image fitting.

This is a faithful-in-spirit reimplementation of the CPPN used in the FER paper
(Kumar, Clune, Lehman, Stanley 2025, arXiv:2505.11581 -- src/cppn.py), reduced
to what we need to probe *optimizer geometry*:

  - inputs per pixel: (x, y, d, b) with d = radial distance, b = bias=1
  - a stack of bias-free Dense layers, each followed by a *mixed* set of
    per-neuron activations (identity / sin / gaussian / tanh) -- the hallmark of
    CPPNs that lets them express symmetry, repetition and smooth gradients
  - a bias-free Dense readout to 3 channels + sigmoid -> RGB in [0,1]

We deliberately drop the HSV color transform of the original so the whole
forward/backward pass is elementary and analytic (no autodiff dependency). This
is an initial-signal harness, not a paper-grade reproduction.

All parameters are 2D weight matrices, which is exactly what the spectral /
orthonormalizing optimizers (Muon, Dion) act on.
"""
import numpy as np

# ---- activations and their derivatives (elementwise) ----
def _identity(z): return z
def _d_identity(z): return np.ones_like(z)
def _sin(z): return np.sin(z)
def _d_sin(z): return np.cos(z)
def _gauss(z): return np.exp(-z * z)
def _d_gauss(z): return -2.0 * z * np.exp(-z * z)
def _tanh(z): return np.tanh(z)
def _d_tanh(z): return 1.0 - np.tanh(z) ** 2

ACT = {
    "identity": (_identity, _d_identity),
    "sin": (_sin, _d_sin),
    "gauss": (_gauss, _d_gauss),
    "tanh": (_tanh, _d_tanh),
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def make_coordinate_grid(res):
    """Return (N, 4) input matrix of (x, y, d, b) over an res x res grid."""
    lin = np.linspace(-1.0, 1.0, res)
    xx, yy = np.meshgrid(lin, lin, indexing="ij")
    d = np.sqrt(xx ** 2 + yy ** 2) * 1.4
    b = np.ones_like(xx)
    X = np.stack([xx.ravel(), yy.ravel(), d.ravel(), b.ravel()], axis=-1)
    return X.astype(np.float64)


class CPPN:
    def __init__(self, n_hidden_layers=6, width=32, activations=None,
                 init_scale=2.5, seed=0):
        """
        activations: list of (name, count) summing to `width`, applied per hidden
        layer. Defaults to a balanced identity/sin/gauss/tanh mix.
        """
        if activations is None:
            q = width // 4
            r = width - 3 * q
            activations = [("identity", q), ("sin", q), ("gauss", q), ("tanh", r)]
        assert sum(c for _, c in activations) == width
        self.width = width
        self.n_hidden = n_hidden_layers
        self.activations = activations
        # per-neuron activation index for vectorized application
        self._act_names = []
        for name, count in activations:
            self._act_names += [name] * count
        rng = np.random.default_rng(seed)
        self.params = []
        fan_in = 4
        for l in range(n_hidden_layers):
            std = init_scale / np.sqrt(fan_in)
            W = rng.standard_normal((fan_in, width)) * std
            self.params.append(W)
            fan_in = width
        # readout
        self.params.append(rng.standard_normal((width, 3)) * (1.0 / np.sqrt(width)))

    # ---- vectorized mixed activation over a (N, width) preactivation ----
    def _act(self, Z):
        out = np.empty_like(Z)
        i = 0
        for name, count in self.activations:
            f, _ = ACT[name]
            out[:, i:i + count] = f(Z[:, i:i + count])
            i += count
        return out

    def _dact(self, Z):
        out = np.empty_like(Z)
        i = 0
        for name, count in self.activations:
            _, df = ACT[name]
            out[:, i:i + count] = df(Z[:, i:i + count])
            i += count
        return out

    def forward(self, X, params=None, cache=False):
        """X: (N,4) -> RGB (N,3). If cache, also return activations for backward."""
        P = self.params if params is None else params
        A = X
        Zs, As = [], [X]
        for l in range(self.n_hidden):
            Z = A @ P[l]
            A = self._act(Z)
            Zs.append(Z); As.append(A)
        Zout = A @ P[-1]
        rgb = _sigmoid(Zout)
        if cache:
            return rgb, (Zs, As, Zout)
        return rgb

    def loss_and_grad(self, X, target, params=None):
        """MSE image loss and gradients wrt every weight matrix."""
        P = self.params if params is None else params
        N = X.shape[0]
        rgb, (Zs, As, Zout) = self.forward(X, params=P, cache=True)
        diff = rgb - target                      # (N,3)
        loss = np.mean(diff ** 2)
        # dL/dZout
        drgb = (2.0 / (N * 3)) * diff
        dZout = drgb * rgb * (1.0 - rgb)         # sigmoid'
        grads = [None] * len(P)
        grads[-1] = As[-1].T @ dZout             # (width,3)
        dA = dZout @ P[-1].T                     # (N,width)
        for l in reversed(range(self.n_hidden)):
            dZ = dA * self._dact(Zs[l])
            grads[l] = As[l].T @ dZ
            dA = dZ @ P[l].T
        return loss, grads

    def render(self, X, params=None):
        return self.forward(X, params=params)

    def copy_params(self):
        return [w.copy() for w in self.params]
