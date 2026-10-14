"""Validate the layer detector on synthetic cross-sections with known layers.

Two experiments, both on images that were NOT used while developing the method
(different random seeds):

1. Random test set: 1-4 coating layers, random thicknesses (5-14 µm), particle sizes,
   densities and boundary waviness; neighbouring layers differ by 30-80 % in particle size.
   Measured: how often the number of layers is found, boundary error in µm (mean
   absolute vertical distance to the true boundary) and substrate interface error.

2. Contrast sweep: 3-layer images where neighbouring layers differ by only 5-80 % in
   particle size (same particle area fraction), to find where the method stops working.

    python validate.py              # ~4 min, prints tables, writes images/validation.png
    python validate.py --quick      # smaller run
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer_analysis import Params, analyze
from make_synthetic import SyntheticSpec, make_cross_section

def random_spec(rng, seed):
    n = int(rng.integers(1, 5))
    med = [rng.uniform(0.18, 0.5)]
    for _ in range(n - 1):                       # neighbouring layers differ by 30-80 % in size
        f = rng.uniform(1.3, 1.8)
        smaller = med[-1] / f
        med.append(smaller if (smaller > 0.15 and rng.random() < 0.6) or med[-1] * f > 0.6 else med[-1] * f)
    return SyntheticSpec(layer_thickness_um=tuple(rng.uniform(5, 14, n)),
                         particle_median_um=tuple(med),
                         particle_density=tuple(rng.uniform(0.12, 0.26, n)),
                         wave_amplitude_um=rng.uniform(0.3, 1.5), seed=seed)


def boundary_error(res, truth, ppu):
    """Mean absolute vertical distance (µm) over all inner boundaries; NaN if counts differ."""
    if res.n_layers != truth["n_layers"] or res.n_layers < 2:
        return np.nan
    return float(np.mean([np.abs(res.curve(i) - truth["boundaries"][i]).mean() / ppu
                          for i in range(res.n_layers - 1)]))


def evaluate(specs, params):
    rows = []
    for spec in specs:
        img, truth = make_cross_section(spec)
        ppu = spec.px_per_um
        auto = analyze(img, params)
        given = analyze(img, Params(**{**params.__dict__, "n_layers": int(truth["n_layers"])}))
        rows.append({"true_n": int(truth["n_layers"]), "found_n": auto.n_layers,
                     "boundary_err_um": boundary_error(auto, truth, ppu),
                     "boundary_err_given_n_um": boundary_error(given, truth, ppu),
                     "substrate_err_um": np.abs(auto.curve(auto.n_layers - 1) - truth["substrate"]).mean() / ppu})
    return pd.DataFrame(rows)


def summarize(df):
    multi = df[df.true_n > 1]
    return {"images": len(df),
            "layer count correct (%)": round(100 * (df.true_n == df.found_n).mean(), 1),
            "boundary error, median (µm)": round(multi.boundary_err_given_n_um.median(), 2),
            "boundary error, 90th percentile (µm)": round(multi.boundary_err_given_n_um.quantile(0.9), 2),
            "substrate error, median (µm)": round(df.substrate_err_um.median(), 2)}


def contrast_specs(contrast, n_images, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_images):
        base = rng.uniform(0.25, 0.4)
        med = (base * contrast, base, base * contrast) if rng.random() < 0.5 else (base, base * contrast, base)
        out.append(SyntheticSpec(layer_thickness_um=tuple(rng.uniform(6, 12, 3)), particle_median_um=med,
                                 particle_density=(0.2, 0.2, 0.2), wave_amplitude_um=rng.uniform(0.3, 1.5),
                                 seed=seed * 1000 + k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_test, n_sweep = (30, 8) if a.quick else (100, 20)
    t0 = time.time()

    rng = np.random.default_rng(2024)                       # unseen seed (development used seed 0)
    specs = [random_spec(rng, 50_000 + k) for k in range(n_test)]
    res = evaluate(specs, Params())
    table = pd.Series(summarize(res), name="value")
    print(table.to_string())
    print("\nConfusion: rows = true number of layers, columns = found")
    print(pd.crosstab(res.true_n, res.found_n).to_string())

    contrasts = [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.8]
    sweep = []
    for c in contrasts:
        df = evaluate(contrast_specs(c, n_sweep, seed=int(c * 100)), Params())
        sweep.append({"size_contrast": c, "layer count correct (%)": 100 * (df.true_n == df.found_n).mean(),
                      "boundary error (µm)": df.boundary_err_given_n_um.median()})
    sweep = pd.DataFrame(sweep)
    print("\nContrast sweep (3 layers, same particle area fraction):")
    print(sweep.round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    multi = res[res.true_n > 1]
    ax[0].hist(multi.boundary_err_given_n_um, bins=np.linspace(0, 3, 31), color="#1f5fbf", alpha=0.85)
    ax[0].axvline(multi.boundary_err_given_n_um.median(), color="k", ls="--", lw=1,
                  label=f"median {multi.boundary_err_given_n_um.median():.2f} µm")
    ax[0].set_xlabel("Boundary error (µm)"); ax[0].set_ylabel("Images")
    ax[0].set_title("Boundary error (100 test images)"); ax[0].legend()
    ns = sorted(res.true_n.unique())
    acc = [100 * (res[res.true_n == n].found_n == n).mean() for n in ns]
    ax[1].bar([str(n) for n in ns], acc, color="#1f5fbf")
    for i, (n, v) in enumerate(zip(ns, acc)):
        ax[1].text(i, v + 1, f"{v:.0f} %\n(n={int((res.true_n == n).sum())})", ha="center", fontsize=9)
    ax[1].set_ylim(0, 115); ax[1].set_xlabel("True number of layers")
    ax[1].set_ylabel("Found correctly (%)"); ax[1].set_title("Number of layers found correctly")
    ax2 = ax[2].twinx()
    ax[2].plot(sweep.size_contrast, sweep["layer count correct (%)"], "o-", color="#1f5fbf")
    ax2.plot(sweep.size_contrast, sweep["boundary error (µm)"], "s--", color="#d9822b")
    ax[2].set_xlabel("Particle size ratio between neighbouring layers")
    ax[2].set_ylabel("Layer count correct (%)", color="#1f5fbf")
    ax2.set_ylabel("Boundary error (µm)", color="#d9822b")
    ax[2].set_title("How different must the layers be?")
    fig.tight_layout()
    fig.savefig("images/validation.png", dpi=110)
    table.to_csv("images/validation_summary.csv")
    sweep.round(3).to_csv("images/contrast_sweep.csv", index=False)
    print(f"\nSaved images/validation.png  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
