"""Overview figure of a layer analysis (one figure, six panels)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde

COLORS = ["#e4572e", "#f3a712", "#29bf12", "#00a5cf", "#7b2cbf", "#ff4fa3"]
SUBSTRATE = "#8d99ae"


def region_colors(res):
    cols = [COLORS[i % len(COLORS)] for i in range(res.n_layers)]
    return cols + ([SUBSTRATE] if res.substrate is not None else [])


def overlay(res, alpha=0.32):
    """Grayscale image with a transparent colour per region."""
    rgb = np.dstack([res.gray / 255.0] * 3)
    out = rgb.copy()
    for k, c in enumerate(region_colors(res)):
        m = res.layer_map == k
        out[m] = (1 - alpha) * rgb[m] + alpha * np.array(to_rgb(c))
    return out


def depth_statistics(res, bin_um=0.6):
    """Median particle diameter (µm) and particle density (per µm²) in depth bins of the coating."""
    p, (H, W) = res.params, res.gray.shape
    coating = res.layer_map < res.n_layers
    df = res.particles[res.particles["layer"] < res.n_layers]
    edges = np.arange(0, H + 1, bin_um * p.px_per_um)
    centers, size, dens = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        area_um2 = coating[int(a):int(b)].sum() / p.px_per_um ** 2
        sel = df[(df["y"] >= a) & (df["y"] < b)]["diameter_um"]
        if area_um2 < 0.3 * (b - a) * W / p.px_per_um ** 2 or len(sel) < 3:
            continue
        centers.append((a + b) / 2 / p.px_per_um)
        size.append(sel.median())
        dens.append(len(sel) / area_um2)
    return np.array(centers), np.array(size), np.array(dens)


def plot_analysis(res, title=None, out=None, show=False):
    p, (H, W) = res.params, res.gray.shape
    ext = [0, W / p.px_per_um, H / p.px_per_um, 0]
    cols = region_colors(res)
    names = [f"Layer {i + 1}" for i in range(res.n_layers)] + (["Substrate"] if res.substrate is not None else [])
    xs_um = np.arange(W) / p.px_per_um
    mean_bounds = [res.curve(i).mean() / p.px_per_um for i in range(res.n_layers - 1)]

    fig = plt.figure(figsize=(17, 10.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], hspace=0.32, wspace=0.28,
                           left=0.05, right=0.98, top=0.88, bottom=0.07)
    fig.suptitle(title or f"Coating layer analysis: {res.n_layers} layer{'s' * (res.n_layers > 1)} detected",
                 fontsize=15, fontweight="bold", y=0.985)

    # (a) image with layers
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(overlay(res), extent=ext)
    for i in range(res.n_layers - 1):
        ax.plot(xs_um, res.curve(i) / p.px_per_um, color="white", lw=2.6)
        ax.plot(xs_um, res.curve(i) / p.px_per_um, color=cols[i + 1], lw=1.4)
    if res.substrate is not None:
        ax.plot(xs_um, res.curve(res.n_layers - 1) / p.px_per_um, color="white", lw=2.6)
        ax.plot(xs_um, res.curve(res.n_layers - 1) / p.px_per_um, color="#3d405b", lw=1.4)
    th = res.local_thickness()
    for k, name in enumerate(names):
        top = 0 if k == 0 else res.curve(k - 1).mean() / p.px_per_um
        mid = top + th[k].mean() / 2
        ax.text(ext[1] * 0.985, mid, f"{name}\n{th[k].mean():.1f} ± {th[k].std():.1f} µm", ha="right",
                va="center", fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=cols[k], ec="none", alpha=0.9))
    ax.set_xlabel("Length (µm)"); ax.set_ylabel("Depth (µm)")
    ax.set_title("Detected layers (wavy boundaries)")

    # (b) depth profile
    ax = fig.add_subplot(gs[0, 1])
    yc, size, dens = depth_statistics(res)
    ax.plot(size, yc, "o-", ms=3, color="#1f5fbf", label="median particle diameter")
    ax.set_xlabel("Median particle diameter (µm)", color="#1f5fbf")
    ax.invert_yaxis(); ax.set_ylabel("Depth (µm)")
    ax2 = ax.twiny()
    ax2.plot(dens, yc, "s-", ms=3, color="#d9822b", alpha=0.8)
    ax2.set_xlabel("Particles per µm²", color="#d9822b")
    for b in mean_bounds:
        ax.axhline(b, color="k", ls="--", lw=0.9)
    if res.substrate is not None:
        ax.axhline(res.curve(res.n_layers - 1).mean() / p.px_per_um, color="#3d405b", lw=1.2)
    ax.set_title("Depth profile: what changes between layers", pad=38)

    # (c) model selection
    ax = fig.add_subplot(gs[0, 2])
    ms = sorted(res.bic)
    vals = np.array([res.bic[m] for m in ms], float)
    ref = np.nanmin(vals)
    top = max(np.nanmax(vals - ref), 1.0)
    for m, v in zip(ms, vals):
        if np.isfinite(v):
            h = max(v - ref, 0.012 * top)
            ax.bar(m, h, color="#1f5fbf" if m == res.n_layers else "#c5cfdb")
            if m == res.n_layers:
                ax.text(m, h + 0.02 * top, "best", ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color="#1f5fbf")
        else:
            ax.text(m, 0.02 * top, "not\nplausible", ha="center", va="bottom", fontsize=8, color="#888")
    ax.set_xticks(ms); ax.set_xlim(0.4, max(ms) + 0.6); ax.set_xlabel("Number of layers in the model")
    ax.set_ylabel("ΔBIC (lower = better)")
    ax.set_title("How many layers? (model selection)")

    # (d) particle size distributions
    ax = fig.add_subplot(gs[1, 0])
    grid = np.linspace(np.log10(0.05), np.log10(3), 300)
    for k in range(res.n_layers):
        d = res.particles[res.particles["layer"] == k]["diameter_um"]
        if len(d) < 5:
            continue
        kde = gaussian_kde(np.log10(d), bw_method=0.25)(grid)
        ax.fill_between(10 ** grid, kde, alpha=0.25, color=cols[k])
        ax.plot(10 ** grid, kde, lw=1.8, color=cols[k], label=f"{names[k]}: median {d.median():.2f} µm (n = {len(d)})")
        ax.axvline(d.median(), color=cols[k], ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("Particle equivalent diameter (µm)"); ax.set_ylabel("Probability density (per log₁₀ µm)")
    ax.legend(fontsize=9); ax.set_title("Particle size distribution per layer")

    # (e) thickness along the cross-section
    ax = fig.add_subplot(gs[1, 1])
    for k in range(res.n_layers):
        ax.plot(xs_um, th[k], color=cols[k], lw=1.6, label=names[k])
    ax.set_xlabel("Length (µm)"); ax.set_ylabel("Local thickness (µm)")
    ax.legend(fontsize=9); ax.set_title("Layer thickness along the cross-section")
    ax.grid(alpha=0.3)

    # (f) gradient inside each layer
    ax = fig.add_subplot(gs[1, 2])
    g = res.gradient_table()
    slopes = res.gradient_trend()
    for k in range(res.n_layers):
        gk = g[g["layer"] == k + 1]
        if len(gk):
            s = slopes.get(names[k], np.nan)
            ax.errorbar(gk["mean_diameter_um"], gk["depth_um"], xerr=gk["std_diameter_um"] / np.sqrt(gk["n"]),
                        fmt="o-", ms=4, color=cols[k], capsize=2,
                        label=f"{names[k]}: {s * 10:+.3f} µm per 10 µm")
    ax.invert_yaxis(); ax.set_xlabel("Mean particle diameter (µm)"); ax.set_ylabel("Depth (µm)")
    ax.legend(fontsize=8.5, title="size change with depth", title_fontsize=8.5)
    ax.set_title("Particle size gradient inside each layer"); ax.grid(alpha=0.3)

    if out:
        fig.savefig(out, dpi=110)
    if show:
        plt.show()
    return fig


def plot_truth_comparison(res, truth, out=None):
    """Detected vs true boundaries on a synthetic image."""
    p, (H, W) = res.params, res.gray.shape
    xs = np.arange(W) / p.px_per_um
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.imshow(res.gray, cmap="gray", extent=[0, W / p.px_per_um, H / p.px_per_um, 0])
    for i, b in enumerate(truth["boundaries"]):
        ax.plot(xs, b / p.px_per_um, color="#29bf12", lw=3, label="true boundary" if i == 0 else None)
    ax.plot(xs, truth["substrate"] / p.px_per_um, color="#29bf12", lw=3)
    for i in range(res.n_layers - 1):
        ax.plot(xs, res.curve(i) / p.px_per_um, color="#e4572e", lw=1.6, ls="--", label="detected" if i == 0 else None)
    ax.plot(xs, res.curve(res.n_layers - 1) / p.px_per_um, color="#e4572e", lw=1.6, ls="--")
    errs = [np.abs(res.curve(i) - truth["boundaries"][i]).mean() / p.px_per_um for i in range(res.n_layers - 1)] \
        if res.n_layers == truth["n_layers"] else []
    ax.set_title("Synthetic test image: detected vs true boundaries" +
                 (f"  |  mean error {np.mean(errs):.2f} µm" if errs else ""))
    ax.set_xlabel("Length (µm)"); ax.set_ylabel("Depth (µm)"); ax.legend(loc="lower left")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=110)
    return fig
