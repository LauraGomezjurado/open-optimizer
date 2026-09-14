"""
Train one arm, evaluate the factoredness proxy on a FIXED probe batch, log.

Fairness controls (mirroring the toy's discipline):
  - same model init seed, same data, same optimizer, same total steps across arms;
    only the objective (objectives.step_loss) differs.
  - the factoredness proxy is evaluated IDENTICALLY for every arm: same fixed
    eval tokens, same forward mode (causal=True, capture=True), so the metric is
    never contaminated by the arm's training-time masking.
  - losses across arms are NOT directly comparable (AR-CE vs masked-CE differ);
    we therefore also report a COMMON yardstick: causal-AR validation CE measured
    the same way for every arm (even the mdlm/masked arms), so "did they reach
    comparable capability" is checkable on one axis.

PREP ONLY. Run on a GPU pod after a fresh go-ahead. Writes results/<arm>.json.
"""
import os, sys, json, argparse, time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT, count_params            # noqa
from objectives import step_loss                    # noqa
from factoredness import factoredness_proxy         # noqa
from data import get_data                            # noqa


def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_ar_ce(model, probe, device):
    """Common yardstick: causal next-token CE on the fixed probe, for EVERY arm."""
    import torch.nn.functional as F
    x, y = probe[:, :-1].to(device), probe[:, 1:].to(device)
    logits = model(x, causal=True)
    return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)))


@torch.no_grad()
def eval_factoredness(model, probe, device, causal=True):
    model(probe.to(device), causal=causal, capture=True)
    return factoredness_proxy(model.captured())


def native_causal(arm, step, total_steps):
    """The forward mode the arm is actually TRAINED under at this step, so the
    factoredness proxy can be read in-distribution (fair) as well as on the
    common causal ruler."""
    if arm == "flat_ar":
        return True
    if arm == "mdlm":
        return False
    if arm == "staged":
        return step >= total_steps // 2      # mask (bidir) first, then AR
    if arm == "staged_strong":
        return step >= total_steps // 2      # first 3 stages masked, last 3 AR
    return True


def train(arm, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    model = TinyGPT(vocab_size=data.vocab_size, d_model=cfg["d_model"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                    max_len=cfg["seq_len"]).to(device)
    print(f"[{arm}] params={count_params(model)/1e6:.1f}M device={device}")
    from optim import build_optimizer
    opts = build_optimizer(cfg.get("optimizer", "adamw"), model, cfg["lr"], cfg["wd"],
                           wcfg=cfg.get("wcfg"))
    probe = data.probe_batch()          # FIXED across arms (same seed/data)
    mask_id = data.mask_token
    total = cfg["steps"]
    log = []
    t0 = time.time()
    acl_kind = cfg.get("anticollapse", "none")
    acl_lambda = cfg.get("acl_lambda", 0.0)
    acl_layer = cfg.get("acl_layer", cfg["n_layer"] // 2)
    for step in range(total):
        batch = data.train_batch()
        loss, kind = step_loss(arm, model, batch, step, total, mask_id)
        ce_val = loss.item()                          # CE part, logged separately
        if acl_kind != "none" and acl_lambda > 0:
            from anticollapse import richness_reg
            # grad-carrying forward to get the mid-layer activation for R
            model(batch[:, :-1] if arm == "flat_ar" else batch,
                  causal=True, capture_grad=True)
            A = model.blocks[acl_layer].mlp_hidden     # (B,T,d_ff), grad on
            R = richness_reg(A, kind=acl_kind)
            loss = loss - acl_lambda * R               # maximize R => -lambda*R
        for o in opts:
            o.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        if step % cfg["eval_every"] == 0 or step == total - 1:
            # measure the proxy TWO ways: on the common causal ruler (comparable
            # across arms) AND in each arm's native mode (fair / in-distribution).
            fm_c = eval_factoredness(model, probe, device, causal=True)
            nat = native_causal(arm, step, total)
            fm_n = fm_c if nat else eval_factoredness(model, probe, device, causal=False)
            ce = eval_ar_ce(model, probe, device)
            # GATE 1 confound control: log total weight-norm of the 2D hidden
            # matrices, so we can check whether a Muon-AdamW effrank gap is just a
            # weight-scale difference rather than a geometry effect.
            wnorm = float(sum(p.detach().norm().item()**2 for n_, p in model.named_parameters()
                              if p.ndim == 2 and "tok" not in n_ and "pos" not in n_ and "head" not in n_) ** 0.5)
            rec = dict(step=step, train_loss=ce_val, kind=kind,
                       ar_ce=ce,
                       mlp_effrank=fm_c["mlp_hidden_effrank_frac_mean"],       # causal ruler
                       resid_effrank=fm_c["resid_effrank_frac_mean"],
                       mlp_pr=fm_c["mlp_hidden_token_pr_frac_mean"],
                       mlp_effrank_native=fm_n["mlp_hidden_effrank_frac_mean"], # native mode
                       resid_effrank_native=fm_n["resid_effrank_frac_mean"],
                       resid_raw_erank=fm_c["resid_raw_erank_mean"],           # cf. their Table 1
                       mlp_raw_erank=fm_c["mlp_hidden_raw_erank_mean"],
                       weight_norm=wnorm,
                       native_causal=nat)
            log.append(rec)
            print(f"[{arm}] step {step:5d} {kind:5s} loss={rec['train_loss']:.3f} "
                  f"ar_ce={ce:.3f} mlp_effrank(causal)={rec['mlp_effrank']:.3f} "
                  f"mlp_effrank(native)={rec['mlp_effrank_native']:.3f}")
    out = dict(arm=arm, cfg=cfg, seconds=round(time.time()-t0, 1),
               final=log[-1], log=log)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    optn = cfg.get("optimizer", "adamw")
    tag = f"_{cfg['tag']}" if cfg.get("tag") else ""
    if optn != "adamw":
        tag = f"_{optn}{tag}"                         # keep optimizer runs separate
    fname = f"{arm}_seed{cfg['seed']}{tag}.json"     # per-seed/tag/opt: no overwrite
    json.dump(out, open(os.path.join(cfg["out_dir"], fname), "w"), indent=2)
    print(f"[{arm}] done in {out['seconds']}s -> {cfg['out_dir']}/{fname}")
    if cfg.get("save_ckpt"):
        ckdir = os.path.join(cfg["out_dir"], "ckpt")
        os.makedirs(ckdir, exist_ok=True)
        ckpt = os.path.join(ckdir, f"{arm}_seed{cfg['seed']}{tag}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg,
                    "vocab_size": data.vocab_size}, ckpt)
        print(f"[{arm}] saved checkpoint -> {ckpt}")


DEFAULT_CFG = dict(
    seed=0, corpus="tinystories", seq_len=256, batch_size=32,
    d_model=384, n_layer=6, n_head=6,
    lr=3e-4, wd=0.1, steps=4000, eval_every=200,
    out_dir=os.path.join(os.path.dirname(__file__), "results"),
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["flat_ar", "staged", "mdlm", "staged_strong"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--d_model", type=int, default=None,
                    help="override width (for the superposition ground-truth pair: "
                         "narrow=forced superposition, wide=room to factor)")
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for output filenames (keep contrast runs separate)")
    ap.add_argument("--optimizer", choices=["adamw", "muon", "sgd", "aurora", "whitened"], default="adamw")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n_layer", type=int, default=None, help="override depth (depth scan)")
    ap.add_argument("--anticollapse",
                    choices=["none", "offdiag", "logdet", "negentropy", "cumulant"],
                    default="none", help="explicit richness regularizer (principle #2); "
                    "negentropy/cumulant = HIGHER-ORDER independence (past decorrelation)")
    ap.add_argument("--acl_lambda", type=float, default=0.0)
    ap.add_argument("--w_alpha", type=float, default=0.5, help="whitened: Sigma_x^-alpha strength")
    ap.add_argument("--w_place", choices=["up", "both", "down", "none"], default="up")
    ap.add_argument("--w_postcorr", type=float, default=0.0, help="phi-gap diagonal post-correction")
    ap.add_argument("--w_base", choices=["muon", "aurora"], default="muon")
    args = ap.parse_args()
    cfg = dict(DEFAULT_CFG)
    cfg["optimizer"] = args.optimizer
    cfg["wcfg"] = dict(alpha=args.w_alpha, place=args.w_place,
                       postcorr=args.w_postcorr, base=args.w_base)
    cfg["anticollapse"] = args.anticollapse
    cfg["acl_lambda"] = args.acl_lambda
    if args.lr is not None: cfg["lr"] = args.lr
    if args.steps is not None: cfg["steps"] = args.steps
    if args.wd is not None: cfg["wd"] = args.wd
    if args.seed is not None: cfg["seed"] = args.seed
    if args.n_layer is not None: cfg["n_layer"] = args.n_layer
    if args.d_model is not None:
        cfg["d_model"] = args.d_model
        # keep heads valid for the chosen width
        cfg["n_head"] = max(1, args.d_model // 64)
    cfg["save_ckpt"] = args.save_ckpt
    cfg["tag"] = args.tag
    train(args.arm, cfg)
