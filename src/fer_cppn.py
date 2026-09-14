"""
Faithful NumPy port of the FER paper's CPPN (akarshkumar0101/fer, src/cppn.py),
for loading and rendering the EVOLVED Picbreeder genomes so we can score them on
our own factoredness metrics -- the "UFR ceiling".

Differences from our probe CPPN (src/cppn_numpy.py), all matched here:
  - per-genome architecture string, e.g. "12;cache:15,gaussian:4,identity:2,sin:1"
    = 12 hidden layers, each width = 15+4+2+1 = 22, activations applied in that
      block order.
  - inputs "y,x,d,b"  (note y,x order; d = sqrt(x^2+y^2)*1.4; b = 1)
  - bias-free Dense layers; readout to 3 = (h,s,v) with NO output activation
  - color: rgb = hsv2rgb((h+1)%1, s.clip(0,1), |v|.clip(0,1))
  - activation semantics: cache/identity = id; gaussian = exp(-x^2)*2-1;
    sigmoid = logistic(x)*2-1; sin = sin; tanh = tanh; cos = cos; relu = relu.
  - params are stored as a SINGLE FLAT VECTOR (evosax ParameterReshaper order:
    layer kernels in forward order, each flattened row-major (fan_in, width)).

Loads only the trusted .npz produced by experiments/convert_genome.py.
"""
import os
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _identity(z): return z
def _sin(z): return np.sin(z)
def _cos(z): return np.cos(z)
def _tanh(z): return np.tanh(z)
def _gaussian(z): return np.exp(-z * z) * 2.0 - 1.0
def _sigmoid(z): return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))) * 2.0 - 1.0
def _relu(z): return np.maximum(z, 0.0)

ACT = {"cache": _identity, "identity": _identity, "sin": _sin, "cos": _cos,
       "tanh": _tanh, "gaussian": _gaussian, "sigmoid": _sigmoid, "relu": _relu}

# elementwise derivatives (for backprop)
def _d_identity(z): return np.ones_like(z)
def _d_sin(z): return np.cos(z)
def _d_cos(z): return -np.sin(z)
def _d_tanh(z): return 1.0 - np.tanh(z) ** 2
def _d_gaussian(z): return -4.0 * z * np.exp(-z * z)          # d/dz[exp(-z^2)*2-1]
def _d_sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return 2.0 * s * (1.0 - s)                                # d/dz[sigmoid*2-1]
def _d_relu(z): return (z > 0).astype(z.dtype)

DACT = {"cache": _d_identity, "identity": _d_identity, "sin": _d_sin,
        "cos": _d_cos, "tanh": _d_tanh, "gaussian": _d_gaussian,
        "sigmoid": _d_sigmoid, "relu": _d_relu}


def hsv2rgb(h, s, v):
    """Vectorized HSV->RGB, matching src/color.py (inputs in [0,1])."""
    h = h * 360.0
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c
    conds = [((0 <= h) & (h < 60)), ((60 <= h) & (h < 120)),
             ((120 <= h) & (h < 180)), ((180 <= h) & (h < 240)),
             ((240 <= h) & (h < 300)), ((300 <= h) & (h < 360))]
    rs = [c, x, np.zeros_like(c), np.zeros_like(c), x, c]
    gs = [x, c, c, x, np.zeros_like(c), np.zeros_like(c)]
    bs = [np.zeros_like(c), np.zeros_like(c), x, c, c, x]
    r = sum(ri * ci for ri, ci in zip(rs, conds)) + m
    g = sum(gi * ci for gi, ci in zip(gs, conds)) + m
    b = sum(bi * ci for bi, ci in zip(bs, conds)) + m
    return np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)


def parse_arch(arch):
    """'12;cache:15,gaussian:4,...' -> (n_layers, [(act,count),...], width)."""
    nl, neur = arch.split(";")
    n_layers = int(nl)
    blocks = [(x.split(":")[0], int(x.split(":")[1])) for x in neur.split(",")]
    width = sum(c for _, c in blocks)
    return n_layers, blocks, width


# arch strings recovered by convert_genome.py (kept here so no pkl is needed)
ARCH = {
    "skull": "12;cache:15,gaussian:4,identity:2,sin:1",
    "butterfly": "16;cache:17,gaussian:4,sigmoid:1,sin:1",
    "apple": "33;cache:28,gaussian:3,sigmoid:5,sin:2",
}
# the SGD-trained genomes share architecture with the evolved ones (same target)
ARCH.update({f"sgd_{k}": v for k, v in list(ARCH.items())})


class FERCppn:
    def __init__(self, arch, flat_params):
        self.n_layers, self.blocks, self.width = parse_arch(arch)
        self.params = self._unflatten(np.asarray(flat_params, np.float64))
        # per-neuron activation names in block order
        self._acts = []
        for name, count in self.blocks:
            self._acts += [name] * count

    def _shapes(self):
        """Numeric forward order Dense_0..Dense_{n_layers} (readout last)."""
        fan = 4
        sh = []
        for _ in range(self.n_layers):
            sh.append((fan, self.width)); fan = self.width
        sh.append((fan, 3))
        return sh

    def _unflatten(self, v):
        """evosax ParameterReshaper flattens flax params in ALPHABETICAL key
        order: Dense_0, Dense_1, Dense_10, Dense_11, Dense_12, Dense_2, ...
        so the readout (Dense_{n_layers}) is NOT at the end of the vector. We lay
        segments out in alphabetical order, then return them in numeric order."""
        shapes = self._shapes()                       # numeric index -> shape
        n_dense = len(shapes)                          # = n_layers + 1
        names = sorted(f"Dense_{i}" for i in range(n_dense))  # alphabetical
        out = {}
        off = 0
        for nm in names:
            i = int(nm.split("_")[1])
            a, b = shapes[i]
            n = a * b
            out[i] = v[off:off + n].reshape(a, b); off += n
        assert off == v.size, (off, v.size)
        return [out[i] for i in range(n_dense)]        # numeric forward order

    def coordinate_grid(self, res):
        lin = np.linspace(-1, 1, res)
        xx, yy = np.meshgrid(lin, lin, indexing="ij")
        d = np.sqrt(xx ** 2 + yy ** 2) * 1.4
        b = np.ones_like(xx)
        # their input order is "y,x,d,b"
        return np.stack([yy.ravel(), xx.ravel(), d.ravel(), b.ravel()], -1)

    def forward(self, X, params=None, cache=False):
        """params=None uses self.params. Accepting an explicit params list makes
        this drop-in compatible with src/metrics.py (weight-sweep Jacobian)."""
        P = self.params if params is None else params
        A = X
        As = [X]
        Zs = []
        for l in range(self.n_layers):
            Z = A @ P[l]
            A = self._act(Z)
            Zs.append(Z); As.append(A)
        hsv = A @ P[-1]                  # (N,3), no activation
        if cache:
            return hsv, (Zs, As, hsv)
        return hsv

    def render(self, X, params=None):
        hsv = self.forward(X, params=params)
        return self._hsv_to_rgb(hsv)

    # ---- color stage + its (numerical) Jacobian ----
    def _hsv_to_rgb(self, hsv):
        h = (hsv[:, 0] + 1) % 1
        s = np.clip(hsv[:, 1], 0, 1)
        v = np.clip(np.abs(hsv[:, 2]), 0, 1)
        r, g, b = hsv2rgb(h, s, v)
        return np.stack([r, g, b], -1)

    def _color_jacobian(self, hsv, eps=1e-4):
        """Per-pixel d(rgb)/d(hsv_raw): (N,3 rgb,3 hsv). Numerical central diff.
        The color stage ((h+1)%1, clip, |.|, hsv2rgb) is a cheap 3->3 map with
        kinks (mod/abs/clip/sector boundaries), so a numerical Jacobian is more
        robust than hand-deriving the piecewise analytic form; the rest of the
        network is backpropped analytically."""
        N = hsv.shape[0]
        J = np.zeros((N, 3, 3))
        for k in range(3):
            hp = hsv.copy(); hp[:, k] += eps
            hm = hsv.copy(); hm[:, k] -= eps
            J[:, :, k] = (self._hsv_to_rgb(hp) - self._hsv_to_rgb(hm)) / (2 * eps)
        return J

    def _act(self, Z):
        return self._act_apply(Z, ACT)

    def _dact(self, Z):
        return self._act_apply(Z, DACT)

    def _act_apply(self, Z, table):
        out = np.empty_like(Z)
        i = 0
        for name, count in self.blocks:
            out[:, i:i + count] = table[name](Z[:, i:i + count])
            i += count
        return out

    def loss_and_grad(self, X, target, params=None):
        """MSE(rgb, target) and gradients wrt every weight matrix.
        target: (N,3) in [0,1]. Backprop: numerical through the color stage,
        analytic through the CPPN layers."""
        P = self.params if params is None else params
        N = X.shape[0]
        hsv, (Zs, As, _) = self.forward(X, params=P, cache=True)
        rgb = self._hsv_to_rgb(hsv)
        diff = rgb - target
        loss = float(np.mean(diff ** 2))
        drgb = (2.0 / (N * 3)) * diff                 # dL/drgb  (N,3)
        # chain through color stage: dL/dhsv = J^T drgb  (per pixel)
        Jc = self._color_jacobian(hsv)                # (N,3rgb,3hsv)
        dhsv = np.einsum("nij,ni->nj", Jc, drgb)      # (N,3hsv)
        grads = [None] * len(P)
        grads[-1] = As[-1].T @ dhsv                   # readout (width,3)
        dA = dhsv @ P[-1].T                           # (N,width)
        for l in reversed(range(self.n_layers)):
            dZ = dA * self._dact(Zs[l])
            grads[l] = As[l].T @ dZ
            dA = dZ @ P[l].T
        return loss, grads


def load_genome(name):
    npz = os.path.join(ROOT, "data", "fer_genome", f"{name}.npz")
    data = np.load(npz)
    key = list(data.keys())[0]
    return FERCppn(ARCH[name], data[key])
