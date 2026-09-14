"""
Safely convert the FER paper's evolved Picbreeder genomes (params.pkl) into
trusted .npz weight files, WITHOUT executing arbitrary pickle code.

Why this file exists: params.pkl came from an external repo (akarshkumar0101/fer)
and `pickle.load` is an arbitrary-code-execution vector. Static disassembly
(`pickletools`) showed these pickles reference ONLY array-reconstruction globals:
    jax._src.array._reconstruct_array
    numpy._core.multiarray._reconstruct
    numpy.ndarray
    numpy.dtype
so a RestrictedUnpickler that whitelists exactly those (and routes the jax
reconstructor to numpy's) can materialize the arrays with no code-exec surface.

After conversion the .pkl files are deleted; downstream code touches only .npz.

Run:  python experiments/convert_genome.py
"""
import os, io, pickle, glob
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GENOME_DIR = os.path.join(ROOT, "data", "fer_genome")


def _jax_reconstruct_array(*args, **kwargs):
    """jax._src.array._reconstruct_array(fun, args, arr_state, aval_state).
    We only need the ndarray it wraps; jax reconstructs by calling `fun(*args)`
    where fun is numpy's _reconstruct. Emulate that with numpy only."""
    fun = args[0]
    fun_args = args[1] if len(args) > 1 else ()
    arr_state = args[2] if len(args) > 2 else None
    arr = fun(*fun_args)
    if arr_state is not None:
        arr.__setstate__(arr_state)
    return arr


# whitelist: name -> callable, nothing else may be globalized
_ALLOWED = {
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("jax._src.array", "_reconstruct_array"): _jax_reconstruct_array,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key in _ALLOWED:
            return _ALLOWED[key]
        raise pickle.UnpicklingError(
            f"blocked non-whitelisted global: {module}.{name}")


def safe_load(path):
    with open(path, "rb") as fh:
        return RestrictedUnpickler(io.BytesIO(fh.read())).load()


def flatten_arrays(obj, prefix=""):
    """Walk the (possibly nested dict) param pytree into {name: ndarray}."""
    out = {}
    if hasattr(obj, "items"):
        for k, v in obj.items():
            out.update(flatten_arrays(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_arrays(v, f"{prefix}/{i}"))
    else:
        arr = np.asarray(obj)
        out[prefix] = arr
    return out


def main():
    genomes = sorted(d for d in glob.glob(os.path.join(GENOME_DIR, "*"))
                     if os.path.isdir(d))
    for gdir in genomes:
        name = os.path.basename(gdir)
        ppath = os.path.join(gdir, "params.pkl")
        if not os.path.exists(ppath):
            print(f"{name}: no params.pkl, skipping"); continue
        params = safe_load(ppath)
        arrays = flatten_arrays(params)
        outp = os.path.join(GENOME_DIR, f"{name}.npz")
        np.savez(outp, **arrays)
        shapes = {k: v.shape for k, v in arrays.items()}
        print(f"{name}: {len(arrays)} arrays -> {os.path.relpath(outp, ROOT)}")
        for k, s in shapes.items():
            print(f"    {k}: {s}")
        # arch is tiny plain text/tuple; try to read it safely too (may be a str)
        apath = os.path.join(gdir, "arch.pkl")
        if os.path.exists(apath):
            try:
                arch = RestrictedUnpickler(io.BytesIO(open(apath, "rb").read())).load()
                print(f"    arch: {arch!r}")
            except Exception as e:
                # arch may pickle only builtins (str/tuple) -> allow those
                print(f"    arch: (needs builtins allow) {type(e).__name__}")


if __name__ == "__main__":
    main()
