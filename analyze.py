"""Command-line tool: analyse one or more cross-section images.

    python analyze.py demo/cross_section.png --px-per-um 16.5
    python analyze.py my_images/*.tif --px-per-um 16.5 --out results
    python analyze.py sample.tif --px-per-um 45 --layers 2      # force the number of layers

For every image a folder <out>/<image name>/ is written with
    analysis.png         overview figure
    layers.csv           thickness, area and particle statistics per layer
    boundaries.csv       boundary depth (µm) along the cross-section
    particles.csv        every particle with position, size and layer
    gradient.csv         particle size vs depth inside each layer
and <out>/summary.csv has one row per image (useful to compare samples).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from layer_analysis import Params, analyze, load_gray  # noqa: E402
from plot import plot_analysis  # noqa: E402


def run_one(path, p, out_dir):
    res = analyze(load_gray(path), p)
    d = Path(out_dir) / Path(path).stem
    d.mkdir(parents=True, exist_ok=True)
    fig = plot_analysis(res, title=f"{Path(path).name}: {res.n_layers} coating layer(s)", out=d / "analysis.png")
    plt.close(fig)
    table = res.layer_table()
    table.to_csv(d / "layers.csv", index=False)
    res.boundary_table().to_csv(d / "boundaries.csv", index=False)
    res.particles.assign(x_um=res.particles["x"] / p.px_per_um, y_um=res.particles["y"] / p.px_per_um) \
        .round(4).to_csv(d / "particles.csv", index=False)
    res.gradient_table().to_csv(d / "gradient.csv", index=False)
    row = {"image": Path(path).name, "n_layers": res.n_layers, "particles": len(res.particles)}
    for _, r in table.iterrows():
        key = r["region"].lower().replace(" ", "_")
        row[f"{key}_thickness_um"] = round(r["mean_thickness_um"], 3)
        if r["region"] != "Substrate":
            row[f"{key}_median_diameter_um"] = round(r["median_diameter_um"], 4)
    print(f"{Path(path).name}: {res.n_layers} layer(s) | " +
          ", ".join(f"{r.region} {r.mean_thickness_um:.2f} µm" for r in table.itertuples()))
    return row


def main():
    ap = argparse.ArgumentParser(description="Detect coating layers from particle size in cross-section images")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--px-per-um", type=float, required=True, help="calibration, e.g. 16.5 if 10 µm = 165 px")
    ap.add_argument("--layers", type=int, default=None, help="number of coating layers (default: automatic)")
    ap.add_argument("--max-layers", type=int, default=5)
    ap.add_argument("--no-substrate", action="store_true", help="image shows no substrate")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    p = Params(px_per_um=a.px_per_um, n_layers=a.layers, max_layers=a.max_layers,
               detect_substrate=not a.no_substrate)
    rows = [run_one(img, p, a.out) for img in a.images]
    pd.DataFrame(rows).to_csv(Path(a.out) / "summary.csv", index=False)
    print(f"Results in {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
