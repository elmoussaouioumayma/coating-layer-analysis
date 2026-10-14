"""Detect coating layers in a cross-section image from the particles inside them.

Many coatings are built from several layers that look almost the same in a micrograph:
same matrix, same kind of particles (pores, inclusions, precipitates). What differs is
the particle *size*, and with it the number of particles per area. This module finds the
layer boundaries from that difference:

    1. Particles       dark objects are segmented (Otsu) and measured (equivalent diameter)
    2. Substrate       the particle-free substrate at the bottom is found per image column
                       (particle density + local texture, refined on the intensity gradient)
    3. Depth profile   median log-diameter and particle density in thin depth bins
    4. Boundaries      change points in the profile give a first, straight guess; each
                       boundary is then refined along x on the individual particles
                       (likelihood split in overlapping windows) and smoothed
    5. Layer count     models with 1..max_layers layers are compared with the BIC on the
                       particles; implausible models (too thin, too similar) are rejected
    6. Metrics         thickness, area and particle statistics per layer, particle-size
                       gradient inside each layer

All lengths in the results are in micrometres (calibration `px_per_um`).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, median_filter
from skimage import filters, measure


@dataclass
class Params:
    px_per_um: float = 16.5
    n_layers: int = None              # None = choose automatically (BIC)
    max_layers: int = 5
    # particles
    min_particle_px: int = 3
    # substrate
    detect_substrate: bool = True
    substrate_threshold: float = 0.20  # fraction of the peak "coating signal" per column
    substrate_cols: int = 120
    substrate_median: int = 8
    substrate_sigma: float = 1.5
    substrate_search_frac: float = 0.01
    # number of layers
    min_size_contrast: float = 1.15   # neighbouring layers must differ by >= 15 % in median particle size ...
    min_density_contrast: float = 1.30  # ... or by >= 30 % in particle density
    min_thickness_um: float = 1.0
    boundary_dof: int = 3             # BIC parameters counted per boundary
    # layer boundaries
    row_bin_um: float = 0.6           # height of one depth bin
    n_cols: int = 60                  # positions along x for the wavy boundary refinement
    window_um: float = 2.0            # half-width of the refinement window
    search_frac: float = 0.6          # how far towards a neighbouring boundary the search may go
    min_particles: int = 10           # minimum particles on each side of a local split
    median_window: int = 7
    smooth_sigma: float = 2.0
    gradient_bins: int = 10


# --------------------------------------------------------------------------- particles
def segment_particles(gray, p: Params):
    """Dark particles -> binary mask and a table (x, y in px; diameter_um, area_um2)."""
    binary = gray < filters.threshold_otsu(gray)
    labels = measure.label(binary)
    sizes = np.bincount(labels.ravel())
    small = sizes < p.min_particle_px
    small[0] = False
    labels[small[labels]] = 0
    binary = labels > 0
    labels = measure.label(binary)
    df = pd.DataFrame(measure.regionprops_table(labels, properties=("area", "centroid", "equivalent_diameter_area")))
    df = df.rename(columns={"centroid-0": "y", "centroid-1": "x", "equivalent_diameter_area": "diameter_px"})
    df["diameter_um"] = df["diameter_px"] / p.px_per_um
    df["area_um2"] = df["area"] / p.px_per_um ** 2
    return binary, df


# --------------------------------------------------------------------------- substrate
def _norm(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)


def detect_substrate(gray, binary, p: Params):
    """Row of the coating/substrate interface for each of `substrate_cols` columns.

    Per column, a "coating signal" (particle density + intensity variation along the
    row) is high in the coating and low in the smooth substrate. The interface is the
    lowest row where the signal is still above `substrate_threshold` x its peak; it is
    then moved to the strongest vertical intensity gradient nearby.
    """
    H, W = gray.shape
    edges = np.linspace(0, W, p.substrate_cols + 1).astype(int)
    centers = (edges[:-1] + edges[1:]) / 2
    rows = np.full(p.substrate_cols, float(H - 1))
    y_smooth = max(3, int(H * 0.015))
    grad_y = np.gradient(cv2.GaussianBlur(gray.astype(float), (0, 0), 1.5), axis=0)
    win = max(5, int(H * p.substrate_search_frac))
    for j in range(p.substrate_cols):
        x0, x1 = edges[j], edges[j + 1]
        density = gaussian_filter1d(binary[:, x0:x1].mean(axis=1).astype(float), y_smooth)
        texture = gaussian_filter1d(gray[:, x0:x1].astype(float).std(axis=1), y_smooth)
        signal = 0.5 * _norm(density) + 0.5 * _norm(texture)
        if signal.max() < 0.05:
            continue
        above = np.nonzero(signal > p.substrate_threshold * signal.max())[0]
        coarse = above[-1] + 1 if above.size else H - 1
        lo, hi = max(0, coarse - win), min(H, coarse + win)
        seg = np.abs(grad_y[lo:hi, x0:x1].mean(axis=1))
        rows[j] = lo + int(np.argmax(seg)) if seg.size and seg.max() > 0 else coarse
    rows = median_filter(rows, size=p.substrate_median, mode="nearest")
    rows = gaussian_filter1d(rows, p.substrate_sigma, mode="nearest")
    return centers, np.clip(rows, 0, H - 1)


# --------------------------------------------------------------------------- depth profile
def depth_profile(df, height, n_rows, row_width=None):
    """Particle statistics per depth bin, shape (n_rows, 2):
    median of log(diameter) and log(number of particles per unit area).

    row_width (optional, length `height`): coating width in pixels per image row, used to
    turn particle counts into densities where the bin is partly substrate.
    Empty bins are filled by linear interpolation.
    """
    bins = np.linspace(0, height, n_rows + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    n_feat = 2
    out = np.full((n_rows, n_feat), np.nan)
    if len(df):
        idx = np.clip(np.digitize(df["y"], bins) - 1, 0, n_rows - 1)
        d = df["diameter_px"].to_numpy()
        for r in np.unique(idx):
            v = d[idx == r]
            out[r, 0] = np.median(np.log(v))
        counts = np.bincount(idx, minlength=n_rows).astype(float)
        if row_width is not None:
            rb = np.clip(bins.round().astype(int), 0, len(row_width))
            area = np.array([row_width[a:b].sum() for a, b in zip(rb[:-1], rb[1:])], float)
            area = area / max(area.max(), 1.0)
            counts = np.where(area > 0.2, counts / np.maximum(area, 1e-6), np.nan)
        out[:, 1] = np.log(counts + 1.0)
    for c in range(n_feat):
        ok = ~np.isnan(out[:, c])
        if ok.sum() >= 2 and not ok.all():
            out[:, c] = interp1d(np.nonzero(ok)[0], out[ok, c], fill_value="extrapolate")(np.arange(n_rows))
        elif not ok.any():
            out[:, c] = 0.0
    return out, centers


def standardize(profile):
    """Scale every feature to unit noise level so that size and density count equally.

    The noise level is estimated from differences between neighbouring bins (robust to
    the steps themselves): sigma = 1.4826 * MAD(diff) / sqrt(2).
    """
    out = profile.astype(float).copy()
    for c in range(out.shape[1]):
        d = np.diff(out[:, c])
        sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) else 1.0
        out[:, c] = (out[:, c] - out[:, c].mean()) / (sigma if sigma > 1e-9 else 1.0)
    return out


# --------------------------------------------------------------------------- change points
def _sse(x):
    return float(((x - x.mean(axis=0)) ** 2).sum()) if len(x) else 0.0


def detect_breaks(signal, n_breaks):
    """Binary segmentation: repeatedly split the segment whose best split lowers the SSE most."""
    segments, breaks = [(0, len(signal))], []
    for _ in range(n_breaks):
        best = (None, -np.inf, None)
        for i, (s, e) in enumerate(segments):
            base = _sse(signal[s:e])
            for k in range(s + 2, e - 1):
                gain = base - _sse(signal[s:k]) - _sse(signal[k:e])
                if gain > best[1]:
                    best = (i, gain, k)
        if best[0] is None:
            break
        i, _, k = best
        s, e = segments.pop(i)
        segments += [(s, k), (k, e)]
        segments.sort()
        breaks.append(k)
    return sorted(breaks)


# --------------------------------------------------------------------------- wavy boundaries
def _split_position(y, v, lo, hi, min_side):
    """Most likely depth of a single boundary among particles sorted by depth y.

    Log-likelihood of a split = size term (log-diameters v: two means, common variance)
    + density term (two Poisson rates along depth). Returns None if
    there are too few particles on either side.
    """
    n = len(v)
    if n < 2 * min_side:
        return None
    c1, c2 = np.cumsum(v), np.cumsum(v ** 2)
    k = np.arange(min_side, n - min_side + 1)
    sse = (c2[k - 1] - c1[k - 1] ** 2 / k) + ((c2[-1] - c2[k - 1]) - (c1[-1] - c1[k - 1]) ** 2 / (n - k))
    ll = -n / 2 * np.log(np.maximum(sse, 1e-12) / n)
    pos = (y[k - 1] + y[np.minimum(k, n - 1)]) / 2
    l1, l2 = np.maximum(pos - lo, 1e-6), np.maximum(hi - pos, 1e-6)
    ll = ll + k * np.log(k / l1) + (n - k) * np.log((n - k) / l2)
    return pos[int(np.argmax(ll))]


def wavy_boundaries(df, height, width, n_layers, p: Params, floor=None):
    """Global change points on the depth profile, then a local refinement along x.

    Refinement: at each of `n_cols` positions, the particles inside an overlapping window
    (width 2 x `window_um`) and inside the search band around the global boundary are
    sorted by depth, and the boundary is placed at the most likely split (different
    particle sizes and densities above and below). This works on individual particles,
    so it is not limited by the height of the depth bins.
    """
    n_rows = max(12, int(round(height / (p.row_bin_um * p.px_per_um))))
    col_x = np.linspace(0, width, p.n_cols + 1)
    col_x = (col_x[:-1] + col_x[1:]) / 2
    floor = np.full(width, float(height)) if floor is None else np.minimum(floor, height)
    row_width = (np.arange(height)[:, None] < floor[None, :]).sum(axis=1)
    signal, yc = depth_profile(df, height, n_rows, row_width)
    gb = detect_breaks(standardize(signal), n_layers - 1)
    global_y = np.array([(yc[b - 1] + yc[b]) / 2 for b in gb])
    if n_layers < 2:
        return col_x, np.zeros((0, p.n_cols)), global_y
    raw = np.tile(global_y[:, None], (1, p.n_cols)).astype(float)
    bounds = np.concatenate(([0.0], global_y, [float(height)]))
    x_all, y_all = df["x"].to_numpy(), df["y"].to_numpy()
    v_all = np.log(df["diameter_px"].to_numpy())

    hw = p.window_um * p.px_per_um
    for i, gy in enumerate(global_y):
        # search band: towards each neighbouring boundary, at most search_frac of the way
        lo = gy - p.search_frac * (gy - bounds[i])
        hi = gy + p.search_frac * (bounds[i + 2] - gy)
        in_band = (y_all >= lo) & (y_all <= hi)
        for j, cx in enumerate(col_x):
            x0, x1 = int(max(0, cx - hw)), int(min(width, cx + hw + 1))
            hi_j = min(hi, floor[x0:x1].min())
            m = in_band & (np.abs(x_all - cx) <= hw) & (y_all <= hi_j)
            if m.sum() < 2 * p.min_particles:
                continue
            order = np.argsort(y_all[m])
            pos = _split_position(y_all[m][order], v_all[m][order], lo, hi_j, p.min_particles)
            if pos is not None:
                raw[i, j] = pos

    curves = np.array([gaussian_filter1d(median_filter(r, p.median_window, mode="nearest"), p.smooth_sigma,
                                         mode="nearest") for r in raw])
    min_gap = 0.3 * height / n_layers
    for i in range(1, len(curves)):
        curves[i] = np.maximum(curves[i], curves[i - 1] + min_gap)
    return col_x, np.clip(curves, 0, height - 1), global_y


# --------------------------------------------------------------------------- result
@dataclass
class LayerResult:
    gray: np.ndarray
    binary: np.ndarray
    particles: pd.DataFrame           # with column "layer" (n_layers = substrate)
    n_layers: int
    col_x: np.ndarray                 # px, columns of the boundary curves
    boundaries: np.ndarray            # (n_layers-1, n_cols) rows in px
    substrate_x: np.ndarray
    substrate: np.ndarray             # rows in px (None if not detected)
    layer_map: np.ndarray             # region id per pixel
    bic: dict
    profile: np.ndarray
    profile_y: np.ndarray
    params: Params = field(repr=False, default=None)

    def curve(self, i, xs=None):
        """Boundary i (0-based; i = n_layers-1 -> substrate) evaluated at pixel columns xs."""
        xs = np.arange(self.gray.shape[1]) if xs is None else xs
        if i < len(self.boundaries):
            cx, cy = self.col_x, self.boundaries[i]
        else:
            cx, cy = self.substrate_x, self.substrate
        return np.interp(xs, cx, cy)

    def local_thickness(self):
        """Thickness of every region at every pixel column, shape (n_regions, W), in µm.

        The first layer starts at the top of the image, so its value is the *visible*
        thickness if the image does not show the coating surface.
        """
        H, W = self.gray.shape
        n_curves = self.n_layers - 1 + (self.substrate is not None)
        edges = [np.zeros(W)] + [self.curve(i) for i in range(n_curves)] + [np.full(W, float(H))]
        return np.diff(np.vstack(edges), axis=0) / self.params.px_per_um

    def layer_table(self):
        """Thickness, area and particle statistics per region, in µm."""
        p, W = self.params, self.gray.shape[1]
        names = [f"Layer {i + 1}" for i in range(self.n_layers)] + (["Substrate"] if self.substrate is not None else [])
        rows = []
        for k, name in enumerate(names):
            area_px = int((self.layer_map == k).sum())
            part = self.particles[self.particles["layer"] == k]["diameter_um"]
            th = self.local_thickness()[k]
            rows.append({"region": name,
                         "mean_thickness_um": area_px / W / p.px_per_um,
                         "thickness_std_um": th.std(),
                         "min_thickness_um": th.min(),
                         "max_thickness_um": th.max(),
                         "area_um2": area_px / p.px_per_um ** 2,
                         "particles": len(part),
                         "mean_diameter_um": part.mean() if len(part) else np.nan,
                         "median_diameter_um": part.median() if len(part) else np.nan,
                         "particle_area_fraction": (self.binary & (self.layer_map == k)).sum() / max(area_px, 1)})
        return pd.DataFrame(rows).round(4)

    def boundary_table(self):
        p = self.params
        out = {"x_um": np.arange(self.gray.shape[1]) / p.px_per_um}
        for i in range(self.n_layers - 1):
            out[f"boundary_{i + 1}_um"] = self.curve(i) / p.px_per_um
        if self.substrate is not None:
            out["substrate_um"] = self.curve(self.n_layers - 1) / p.px_per_um
        return pd.DataFrame(out).round(4)

    def gradient_table(self):
        """Mean particle diameter vs depth inside each coating layer."""
        p, rows = self.params, []
        for k in range(self.n_layers):
            sub = self.particles[self.particles["layer"] == k]
            if len(sub) < 3:
                continue
            nb = max(3, min(p.gradient_bins, len(sub) // 10))
            bins = np.linspace(sub["y"].min(), sub["y"].max() + 1e-6, nb + 1)
            idx = np.clip(np.digitize(sub["y"], bins) - 1, 0, nb - 1)
            for b in range(nb):
                d = sub["diameter_um"][idx == b]
                if len(d):
                    rows.append({"layer": k + 1, "depth_um": (bins[b] + bins[b + 1]) / 2 / p.px_per_um,
                                 "mean_diameter_um": d.mean(), "std_diameter_um": d.std(), "n": len(d)})
        return pd.DataFrame(rows).round(4)

    def gradient_trend(self):
        """Linear slope of particle diameter vs depth per layer (µm per µm) with its sign."""
        out = {}
        for k in range(self.n_layers):
            sub = self.particles[self.particles["layer"] == k]
            if len(sub) >= 20:
                slope = np.polyfit(sub["y"] / self.params.px_per_um, sub["diameter_um"], 1)[0]
                out[f"Layer {k + 1}"] = slope
        return out


def _assign(xs_df, ys_df, col_x, curves):
    lab = np.zeros(len(xs_df), int)
    for c in curves:
        lab += ys_df >= np.interp(xs_df, col_x, c)
    return lab


def score_layer_model(coating, col_x, curves, floor, coat_h, p: Params):
    """BIC of a layer model on the individual particles, plus a physical plausibility check.

    Every particle is assigned to a layer by the wavy boundaries. Within each layer,
    log-diameters are Gaussian (own mean, common variance) and particle positions are
    a Poisson process (own density). BIC = -2 log L + k log(n_particles), with
    k = 2 per layer + `boundary_dof` per boundary.
    A model is "plausible" if every layer is at least `min_thickness_um` thick and
    neighbouring layers differ clearly in particle size or density.
    """
    W = len(floor)
    x, y = coating["x"].to_numpy(), coating["y"].to_numpy()
    v = np.log(coating["diameter_px"].to_numpy())
    n = len(v)
    lab = _assign(x, y, col_x, curves)
    m = len(curves) + 1
    xs = np.arange(W)
    upper = np.vstack([np.zeros(W)] + [np.interp(xs, col_x, c) for c in curves])
    lower = np.vstack([np.interp(xs, col_x, c) for c in curves] + [floor])
    area = np.clip(np.minimum(lower, floor) - np.minimum(upper, floor), 0, None).sum(axis=1)
    sse, ll_rate, med, dens = 0.0, 0.0, [], []
    for k in range(m):
        vk = v[lab == k]
        if len(vk) == 0 or area[k] <= 0:
            return np.inf, False
        sse += ((vk - vk.mean()) ** 2).sum()
        ll_rate += len(vk) * np.log(len(vk) / area[k]) - len(vk)
        med.append(np.median(vk)); dens.append(len(vk) / area[k])
    ll = -n / 2 * np.log(max(sse, 1e-12) / n) + ll_rate
    k_free = 2 * m + (m - 1) * p.boundary_dof
    bic = -2 * ll + k_free * np.log(n)
    thick_ok = all(a / W / p.px_per_um >= p.min_thickness_um for a in area)
    contrast_ok = all(abs(med[i + 1] - med[i]) >= np.log(p.min_size_contrast) or
                      abs(np.log(dens[i + 1] / dens[i])) >= np.log(p.min_density_contrast) for i in range(m - 1))
    return bic, thick_ok and contrast_ok


def analyze(gray, p: Params = None):
    """Full pipeline on a grayscale (uint8) cross-section image."""
    p = p or Params()
    H, W = gray.shape
    binary, df = segment_particles(gray, p)

    sub_x = sub_y = None
    coat_h, coating = H, df
    if p.detect_substrate:
        sub_x, sub_y = detect_substrate(gray, binary, p)
        below = df["y"].to_numpy() >= np.interp(df["x"], sub_x, sub_y)
        coating = df[~below]
        coat_h = int(np.ceil(sub_y.max()))

    floor = np.interp(np.arange(W), sub_x, sub_y) if sub_y is not None else np.full(W, float(H))
    row_width = (np.arange(coat_h)[:, None] < floor[None, :]).sum(axis=1)
    n_rows = max(12, int(round(coat_h / (p.row_bin_um * p.px_per_um))))
    profile, profile_y = depth_profile(coating, coat_h, n_rows, row_width)
    scores, fits = {}, {}
    for m in range(1, max(p.max_layers, p.n_layers or 0) + 1):
        fits[m] = wavy_boundaries(coating, coat_h, W, m, p, floor)[:2]
        bic, ok = score_layer_model(coating, *fits[m], floor, coat_h, p)
        scores[m] = bic if ok else np.nan
    valid = {m: s_ for m, s_ in scores.items() if np.isfinite(s_)}
    n_layers = p.n_layers or (min(valid, key=valid.get) if valid else 1)
    col_x, curves = fits[n_layers]

    # region id per pixel: 0..n_layers-1 coating layers, n_layers = substrate
    xs = np.arange(W)
    rows = np.arange(H)[:, None]
    layer_map = np.zeros((H, W), np.int16)
    for c in curves:
        layer_map += rows >= np.interp(xs, col_x, c)[None, :]
    if sub_y is not None:
        layer_map[rows >= np.interp(xs, sub_x, sub_y)[None, :]] = n_layers

    yi = np.clip(df["y"].round().astype(int), 0, H - 1)
    xi = np.clip(df["x"].round().astype(int), 0, W - 1)
    df = df.assign(layer=layer_map[yi, xi])
    return LayerResult(gray, binary, df, n_layers, col_x, curves, sub_x, sub_y, layer_map, scores,
                       profile, profile_y, p)


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img
