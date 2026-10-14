"""Synthetic cross-section images of multilayer coatings, with known ground truth.

Each image shows a coating on a substrate, seen in cross-section (depth = image rows).
Every coating layer contains dark, angular particles (pores / inclusions) whose size
differs from layer to layer. The boundaries between layers are wavy; the substrate at
the bottom is particle-free with fine polishing scratches.

Because the true boundary positions and the true number of layers are known, the
detector can be validated against them (see validate.py).

    python make_synthetic.py                 # writes demo/cross_section.png (+ truth .npz)
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SyntheticSpec:
    width: int = 1024
    px_per_um: float = 16.5
    layer_thickness_um: tuple = (12.0, 14.0)      # top -> bottom, one entry per coating layer
    particle_median_um: tuple = (0.42, 0.28)      # median equivalent diameter per layer
    particle_density: tuple = (0.24, 0.22)        # fraction of the layer area covered by particles
    size_spread: float = 0.45                     # lognormal sigma of particle diameters
    gradient: tuple = None                        # optional relative size change top->bottom per layer
    wave_amplitude_um: float = 1.2
    substrate_um: float = 6.0                     # visible substrate height below the coating
    noise: float = 6.0
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _wavy_curve(rng, width, base_y, amp_px):
    """Smooth random boundary: sum of a few sinusoids with random phase."""
    x = np.arange(width)
    y = np.full(width, float(base_y))
    for period in (rng.uniform(300, 600), rng.uniform(150, 280)):
        y += amp_px * rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * x / period + rng.uniform(0, 2 * np.pi)) / 1.3
    return y


def _particle(rng, img, cx, cy, d_px):
    """Draw one angular dark particle (random polygon) of equivalent diameter d_px."""
    n = rng.integers(4, 8)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = d_px / 2 * rng.uniform(0.7, 1.35, n) * rng.uniform(0.8, 1.2)
    stretch = rng.uniform(0.6, 1.0)
    rot = rng.uniform(0, np.pi)
    xs = r * np.cos(angles)
    ys = r * np.sin(angles) * stretch
    pts = np.stack([xs * np.cos(rot) - ys * np.sin(rot) + cx, xs * np.sin(rot) + ys * np.cos(rot) + cy], 1)
    cv2.fillPoly(img, [np.round(pts * 4).astype(np.int32)], int(rng.uniform(15, 45)), lineType=cv2.LINE_AA, shift=2)


def make_cross_section(spec: SyntheticSpec):
    """Return (uint8 image, truth dict). Truth boundaries are in pixel rows per image column."""
    rng = np.random.default_rng(spec.seed)
    W, ppu = spec.width, spec.px_per_um
    n_layers = len(spec.layer_thickness_um)
    amp = spec.wave_amplitude_um * ppu

    # boundaries: between coating layers, then coating / substrate
    tops = np.cumsum([0.0] + [t * ppu for t in spec.layer_thickness_um])
    boundaries = [_wavy_curve(rng, W, tops[i], amp) for i in range(1, n_layers)]
    substrate = _wavy_curve(rng, W, tops[-1], amp * 0.35)
    H = int(tops[-1] + spec.substrate_um * ppu)

    # matrix (coating) background with fine texture, substrate slightly darker and smooth
    img = rng.normal(132, 7, (H, W))
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    rows = np.arange(H)[:, None]
    sub_mask = rows >= substrate[None, :]
    img[sub_mask] = cv2.GaussianBlur(rng.normal(118, 3, (H, W)), (0, 0), 2.0)[sub_mask]
    img = img.astype(np.uint8)

    # polishing scratches across the whole cross-section (stronger on the substrate)
    for _ in range(40):
        ang = rng.uniform(-0.5, 0.5) + (np.pi / 2 if rng.random() < 0.3 else 0)
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        L = rng.uniform(150, 700)
        p1 = (int(x0 - L * np.cos(ang)), int(y0 - L * np.sin(ang)))
        p2 = (int(x0 + L * np.cos(ang)), int(y0 + L * np.sin(ang)))
        cv2.line(img, p1, p2, int(rng.choice([118, 150])), 1, cv2.LINE_AA)

    # particles, layer by layer (placed where the pixel belongs to that layer)
    layer_index = np.zeros((H, W), np.int16)
    for b in boundaries:
        layer_index += rows >= b[None, :]
    layer_index[sub_mask] = n_layers
    grad = spec.gradient or (0.0,) * n_layers
    for k in range(n_layers):
        area_px = (layer_index == k).sum()
        med_px = spec.particle_median_um[k] * ppu
        mean_area = np.pi / 4 * (med_px ** 2) * np.exp(2 * spec.size_spread ** 2)
        n = int(spec.particle_density[k] * area_px / mean_area)
        ys_all, xs_all = np.nonzero(layer_index == k)
        pick = rng.integers(0, len(ys_all), n)
        y_top, y_bot = ys_all.min(), ys_all.max()
        for y, x in zip(ys_all[pick], xs_all[pick]):
            rel = (y - y_top) / max(y_bot - y_top, 1)            # 0 at layer top, 1 at bottom
            d = med_px * (1 + grad[k] * (rel - 0.5)) * rng.lognormal(0, spec.size_spread)
            if 1.2 <= d <= 40:
                _particle(rng, img, x + rng.uniform(-.5, .5), y + rng.uniform(-.5, .5), d)
    # keep particles out of the substrate (re-paint substrate region after particles may overlap it)
    base_sub = cv2.GaussianBlur(rng.normal(118, 3, (H, W)), (0, 0), 2.0).astype(np.uint8)
    img[sub_mask] = np.maximum(img[sub_mask], base_sub[sub_mask] - 2)

    img = cv2.GaussianBlur(img.astype(float), (0, 0), 0.7) + rng.normal(0, spec.noise, (H, W))
    img = np.clip(img, 0, 255).astype(np.uint8)
    truth = dict(boundaries=np.array(boundaries).reshape(-1, W), substrate=substrate,
                 n_layers=n_layers, px_per_um=ppu,
                 particle_median_um=np.array(spec.particle_median_um))
    return img, truth


def save_with_truth(path, img, truth):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    np.savez(path.with_suffix(".truth.npz"), **truth)


def load_truth(image_path):
    d = np.load(Path(image_path).with_suffix(".truth.npz"))
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    spec = SyntheticSpec(layer_thickness_um=(9.0, 10.0, 8.0), particle_median_um=(0.45, 0.30, 0.20),
                         particle_density=(0.22, 0.22, 0.18), gradient=(-0.3, 0.0, 0.0), seed=7)
    img, truth = make_cross_section(spec)
    save_with_truth("demo/cross_section.png", img, truth)
    print("wrote demo/cross_section.png", img.shape, "layers:", truth["n_layers"])
