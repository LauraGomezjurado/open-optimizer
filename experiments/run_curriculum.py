"""
Objective-lever experiment: does *what* we optimize (not the step direction)
move NEURON-STRUCTURE factoredness toward the evolved (UFR) genome?

Three schedules, ALL ending on the same skull target, same architecture, same
optimizer (spectral p=1), same total iterations -- so the ONLY thing that varies
is the sequence of objectives. Scored with the capacity-invariant neuron metric.

  flat        : fit the skull directly the whole time (baseline = standard SGD).
  lineage     : Stanley's stepping-stones -- fit a sequence of DIFFERENT coherent
                images (apple -> butterfly -> skull), each phase warm-started from
                the last. Tests "reuse across a lineage of coherent forms".
  coarsefine  : diffusion-flavored multi-scale objective -- fit the SAME skull but
                from heavily blurred -> sharp over training (low-frequency first,
                high-frequency last). Tests "multi-scale objective" intuition.

Prediction (if the objective lever is real): lineage and/or coarsefine end with
LOWER neuron_tv / neuron_hf than flat -- i.e. cleaner, more factored neurons --
even though all three reach the same final skull at the same loss.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cppn_numpy import CPPN, make_coordinate_grid       # noqa
from optimizers import SteepestDescent                   # noqa
from metrics import neuron_structure_metrics             # noqa
from PIL import Image, ImageFilter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = 48
N_ITERS = 1800
N_HIDDEN, WIDTH = 6, 32
INIT_SCALE = 2.5
SEEDS = [0, 1, 2, 3, 4]
LR = 0.05                       # spectral p=1 baseline geometry for all arms
KIND, P = "spectral", 1.0


def load_img(name):
    im = Image.open(os.path.join(ROOT, "data", name)).convert("RGB")
    if im.size != (RES, RES):
        im = im.resize((RES, RES), Image.LANCZOS)
    return im


def to_arr(im):
    return np.asarray(im, float).reshape(-1, 3) / 255.0


def blurred(im, radius):
    if radius <= 0:
        return to_arr(im)
    return to_arr(im.filter(ImageFilter.GaussianBlur(radius=radius)))


def cosine_lr(base, it, total):
    return base * 0.5 * (1 + np.cos(np.pi * it / total))


def train(schedule, seed):
    """schedule: list of (target_array, n_iters). Trains through them in order,
    warm-started, with a fresh cosine LR decay per phase."""
    c = CPPN(N_HIDDEN, WIDTH, init_scale=INIT_SCALE, seed=seed)
    opt = SteepestDescent(c.params, kind=KIND, p=P, lr=LR)
    loss = None
    for target, n in schedule:
        for it in range(n):
            opt.lr = cosine_lr(LR, it, n)
            loss, g = c.loss_and_grad(make_coordinate_grid(RES), target)
            c.params = opt.step(c.params, g)
    return c, float(loss)


def build_schedules():
    skull = load_img("target_64.png")
    apple = load_img("target_apple_64.png")
    butt = load_img("target_butterfly_64.png")
    sk = to_arr(skull)

    flat = [(sk, N_ITERS)]

    # lineage: 3 equal phases, different coherent images, ending on skull
    third = N_ITERS // 3
    lineage = [(to_arr(apple), third), (to_arr(butt), third),
               (sk, N_ITERS - 2 * third)]

    # coarse->fine: same skull, decreasing blur across phases
    n_phase = 4
    per = N_ITERS // n_phase
    radii = [4.0, 2.0, 1.0, 0.0]
    coarsefine = [(blurred(skull, r), per if i < n_phase - 1
                   else N_ITERS - per * (n_phase - 1))
                  for i, r in enumerate(radii)]
    return {"flat": flat, "lineage": lineage, "coarsefine": coarsefine}, sk


def main():
    schedules, skull = build_schedules()
    X = make_coordinate_grid(RES)
    rows = []
    print(f"### curriculum / objective-lever (skull, {len(SEEDS)} seeds, "
          f"{KIND} p={P}, {N_ITERS} iters) ###")
    print("all arms end on skull; only the OBJECTIVE SEQUENCE differs. "
          "neuron_tv/hf LOWER = more factored\n")
    for name, sched in schedules.items():
        tvs, hfs, ls = [], [], []
        for s in SEEDS:
            c, loss = train(sched, s)
            m = neuron_structure_metrics(c, X, RES)
            tvs.append(m["neuron_tv"]); hfs.append(m["neuron_hf_energy"]); ls.append(loss)
            rows.append(dict(schedule=name, seed=s, final_loss=loss, **m))
        print(f"  {name:12s} loss={np.mean(ls):.4f}  "
              f"neuron_TV={np.mean(tvs):.4f}±{np.std(tvs):.4f}  "
              f"neuron_HF={np.mean(hfs):.4f}±{np.std(hfs):.4f}")
    out = os.path.join(ROOT, "results", "curriculum.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nWrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
