#!/usr/bin/env python3
"""
focus_assess.py
---------------
Rank IFU flat-field images by focus quality using peak-to-trough contrast,
normalised against upper (peak) and lower (gap) throughput envelopes.

Strategy
========
1. Trace fibres on a single reference image (or first file in the folder).
   Trace polynomials are reused for every image in the folder, on the
   assumption that defocus does not translate the traces.
2. For each image, sample several columns spanning the dispersion axis. At
   each sampled column:
     * read each trace's peak value as the max in a +/-1 pixel window around
       the polynomial-predicted y
     * read each inter-trace trough value as the min in a +/-1 pixel window
       around the midpoint between adjacent peaks
     * **auto-detect gap y-locations** at this column from the trace peaks:
       any consecutive peak pair with spacing > 1.8x the median pitch is
       considered an inter-bundle gap (centre = midpoint of the pair); two
       edge "gap" samples are always added near the top and bottom of the
       trace region.  --gaps can override with explicit values.
     * fit upper envelope through peak values with a smoothing spline,
       lower envelope through gap values with a low-order polynomial
     * normalise: (profile - lower_env) / (upper_env - lower_env)
     * "contrast" defined as 1 - Michelson = 1 - (P - T) / (P + T)
       = 2T / (P + T) at each peak / adjacent-trough.
       Range 0 (sharp, troughs at floor) to ~1 (blurry, troughs filled in).
3. Score = median of all contrast values across all peaks and all columns
   (more samples than median-of-medians, less noisy).  LOWER = SHARPER.
4. Outputs (in <folder>/focus_results/):
     * focus_scores.csv  -- rank, filename, contrast, scatter, FOCUS hdr
     * profile_<n>.png   -- multi-panel diagnostic, N_plot rows x 2 cols
     * topo_<n>.png      -- 2D heatmap of contrast across the frame
     * contrast_vs_focus.png -- summary plot (if FOCUS keyword present)

Usage
=====
    python focus_assess.py /path/to/flats --ext 1 --focus-key FOCUS
"""

import os
import glob
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline, griddata


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def _centroid(profile, peak_index, window=2):
    y = np.arange(peak_index - window, peak_index + window + 1)
    y = y[(y >= 0) & (y < len(profile))]
    w = profile[y]
    w = np.maximum(w - np.nanmin(w), 0)
    if np.sum(w) == 0:
        return float(peak_index)
    return float(np.sum(y * w) / np.sum(w))


def trace_fibres(data, n_steps=30, poly_order=3, verbose=True):
    ny, nx = data.shape
    cx = nx // 2
    prof = np.median(data[:, cx - 10: cx + 10], axis=1)
    prof_sm = gaussian_filter1d(prof, sigma=1.2)
    ref_peaks, _ = find_peaks(
        prof_sm, distance=4,
        prominence=np.nanpercentile(prof_sm, 5),
    )
    ref_peaks = np.sort(ref_peaks)
    n = len(ref_peaks)
    if verbose:
        print(f"  reference traces detected at x={cx}: {n}")

    trace = {i: {"x": [cx], "y": [float(p)]} for i, p in enumerate(ref_peaks)}
    x_locs = np.linspace(50, nx - 50, n_steps, dtype=int)

    for direction in (1, -1):
        cur = ref_peaks.astype(float).copy()
        if direction == 1:
            rel = x_locs[x_locs > cx]
        else:
            rel = x_locs[x_locs < cx][::-1]
        for x in rel:
            col = np.median(data[:, x - 2: x + 3], axis=1)
            col_sm = gaussian_filter1d(col, sigma=1.0)
            peaks, _ = find_peaks(
                col_sm, distance=4,
                prominence=np.nanpercentile(col_sm, 5),
            )
            if len(peaks) == 0:
                continue
            for i in range(n):
                d = np.abs(peaks - cur[i])
                k = np.argmin(d)
                if d[k] < 8:
                    sub_y = _centroid(col, peaks[k], window=2)
                    trace[i]["x"].append(int(x))
                    trace[i]["y"].append(sub_y)
                    cur[i] = sub_y

    coeffs = []
    for i in range(n):
        xs = np.asarray(trace[i]["x"])
        ys = np.asarray(trace[i]["y"])
        if len(xs) > poly_order + 2:
            coeffs.append(np.polyfit(xs, ys, poly_order))
    return np.asarray(coeffs)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _max_in_window(col, y, half=1):
    lo = int(max(0, np.floor(y - half)))
    hi = int(min(len(col), np.ceil(y + half) + 1))
    if hi <= lo:
        return np.nan
    return float(np.nanmax(col[lo:hi]))


def _min_in_window(col, y, half=1):
    lo = int(max(0, np.floor(y - half)))
    hi = int(min(len(col), np.ceil(y + half) + 1))
    if hi <= lo:
        return np.nan
    return float(np.nanmin(col[lo:hi]))


def _median_in_window(col, y, half=2):
    lo = int(max(0, np.floor(y - half)))
    hi = int(min(len(col), np.ceil(y + half) + 1))
    if hi <= lo:
        return np.nan
    return float(np.nanmedian(col[lo:hi]))


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def auto_gaps(peak_ys, ny, pitch_factor=1.8, edge_pad_pitches=1.5,
              edge_min_pad=4):
    """
    Detect gap y-locations from sorted peak positions in a single column.

    Parameters
    ----------
    pitch_factor : float
        Inter-peak spacing must exceed this multiple of the median pitch
        to count as an inter-bundle gap.
    edge_pad_pitches : float
        Distance from the outermost peak (in units of median pitch) at
        which to place the edge gap samples.
    edge_min_pad : float
        Hard floor on edge sample distance from the detector boundary.
    """
    peak_ys = np.sort(np.asarray(peak_ys, dtype=float))
    if len(peak_ys) < 3:
        return np.array([])
    pitches = np.diff(peak_ys)
    median_pitch = float(np.median(pitches))

    inter = []
    for i, p in enumerate(pitches):
        if p > pitch_factor * median_pitch:
            inter.append(0.5 * (peak_ys[i] + peak_ys[i + 1]))

    edge_low = max(edge_min_pad,
                   peak_ys[0] - edge_pad_pitches * median_pitch)
    edge_low = max(edge_low, 2.0)
    edge_high = min(ny - 1 - edge_min_pad,
                    peak_ys[-1] + edge_pad_pitches * median_pitch)
    edge_high = min(edge_high, ny - 3.0)

    return np.array(sorted([edge_low] + inter + [edge_high]))


# ---------------------------------------------------------------------------
# Envelope fits
# ---------------------------------------------------------------------------

def fit_upper_envelope(peak_ys, peak_vals, smooth_frac=0.025, k=3):
    peak_ys = np.asarray(peak_ys, dtype=float)
    peak_vals = np.asarray(peak_vals, dtype=float)
    good = np.isfinite(peak_ys) & np.isfinite(peak_vals)
    peak_ys = peak_ys[good]
    peak_vals = peak_vals[good]
    order = np.argsort(peak_ys)
    peak_ys = peak_ys[order]
    peak_vals = peak_vals[order]
    keep = np.concatenate([[True], np.diff(peak_ys) > 0])
    peak_ys = peak_ys[keep]
    peak_vals = peak_vals[keep]
    if len(peak_ys) < k + 2:
        deg = max(1, len(peak_ys) - 1)
        c = np.polyfit(peak_ys, peak_vals, deg)
        return lambda yy, _c=c: np.polyval(_c, yy)
    sigma = max(smooth_frac * np.median(peak_vals), 1e-6)
    s = len(peak_ys) * sigma ** 2
    spl = UnivariateSpline(peak_ys, peak_vals, k=k, s=s, ext=3)
    return spl


def fit_lower_envelope(gap_ys, gap_vals, order=2):
    gap_ys = np.asarray(gap_ys, dtype=float)
    gap_vals = np.asarray(gap_vals, dtype=float)
    good = np.isfinite(gap_ys) & np.isfinite(gap_vals)
    gap_ys = gap_ys[good]
    gap_vals = gap_vals[good]
    if len(gap_ys) < 2:
        # Degenerate: constant
        v = float(np.nanmedian(gap_vals)) if len(gap_vals) else 0.0
        return lambda yy, _v=v: np.full_like(np.atleast_1d(yy), _v,
                                             dtype=float)
    o = min(order, len(gap_ys) - 1)
    c = np.polyfit(gap_ys, gap_vals, o)
    return lambda yy, _c=c: np.polyval(_c, yy)


# ---------------------------------------------------------------------------
# Focus measurement
# ---------------------------------------------------------------------------

def measure_focus(data, trace_coeffs, eval_cols,
                  fixed_gap_ys=None,
                  upper_smooth_frac=0.025, lower_order=2,
                  col_window=2, peak_half=1, trough_half=1, gap_half=2,
                  gap_buffer=12.0, pitch_factor=1.5,
                  detect_pitch_factor=1.8):
    """
    fixed_gap_ys : array or None
        If given, used as gap y-locations at every column (override).
        Otherwise gaps are auto-detected per column from the trace peaks.
    """
    ny, nx = data.shape

    per_col_contrasts = []
    diagnostics = []

    for x in eval_cols:
        x = int(x)
        col = np.median(
            data[:, max(0, x - col_window): x + col_window + 1], axis=1
        ).astype(float)

        peak_ys = np.array([np.polyval(c, x) for c in trace_coeffs])
        keep = (peak_ys > peak_half + 1) & (peak_ys < ny - peak_half - 2)
        peak_ys = np.sort(peak_ys[keep])
        if len(peak_ys) < 5:
            continue

        peak_vals = np.array([_max_in_window(col, y, peak_half)
                              for y in peak_ys])
        trough_ys = 0.5 * (peak_ys[:-1] + peak_ys[1:])
        pitches = np.diff(peak_ys)
        trough_vals = np.array([_min_in_window(col, y, trough_half)
                                for y in trough_ys])

        # Determine gap y-locations for this column
        if fixed_gap_ys is not None:
            gap_ys = np.asarray(sorted(fixed_gap_ys), dtype=float)
        else:
            gap_ys = auto_gaps(peak_ys, ny,
                               pitch_factor=detect_pitch_factor)
        gap_vals = np.array([_median_in_window(col, y, gap_half)
                             for y in gap_ys])

        # Reject troughs near gaps or spanning unusually large pitches
        med_pitch = np.median(pitches)
        too_wide = pitches > pitch_factor * med_pitch
        if len(gap_ys):
            near_gap = np.array([
                np.any(np.abs(t - gap_ys) < gap_buffer) for t in trough_ys
            ])
        else:
            near_gap = np.zeros_like(trough_ys, dtype=bool)
        trough_mask = ~(too_wide | near_gap)

        up_fn = fit_upper_envelope(peak_ys, peak_vals,
                                   smooth_frac=upper_smooth_frac)
        lo_fn = fit_lower_envelope(gap_ys, gap_vals, order=lower_order)

        up_p = up_fn(peak_ys)
        lo_p = lo_fn(peak_ys)
        up_t = up_fn(trough_ys)
        lo_t = lo_fn(trough_ys)

        amp_p = up_p - lo_p
        amp_t = up_t - lo_t
        with np.errstate(divide="ignore", invalid="ignore"):
            P = (peak_vals - lo_p) / amp_p
            T = (trough_vals - lo_t) / amp_t

        P_pair = np.minimum(P[:-1], P[1:])
        denom = P_pair + T
        with np.errstate(divide="ignore", invalid="ignore"):
            # contrast = 1 - Michelson = 2T/(P+T); 0 = sharp, ~1 = blurry
            mich = np.where(
                np.isfinite(P_pair) & np.isfinite(T) & (denom > 0),
                1.0 - (P_pair - T) / denom,
                np.nan,
            )
        mich = np.where(trough_mask, mich, np.nan)
        per_col_contrasts.append(mich)
        diagnostics.append({
            "x": x, "col": col,
            "peak_ys": peak_ys, "peak_vals": peak_vals,
            "trough_ys": trough_ys, "trough_vals": trough_vals,
            "trough_mask": trough_mask,
            "gap_ys": gap_ys, "gap_vals": gap_vals,
            "up_fn": up_fn, "lo_fn": lo_fn,
            "P": P, "T": T, "contrast": mich,
        })

    if not per_col_contrasts:
        return {"score": np.nan, "scatter": np.nan,
                "diagnostics": []}

    # Median across all peak/trough samples (every column, every peak-pair)
    all_c = np.concatenate(per_col_contrasts)
    finite = all_c[np.isfinite(all_c)]
    if finite.size == 0:
        score = np.nan
        scatter = np.nan
    else:
        score = float(np.median(finite))
        # Robust scatter via MAD scaled to sigma
        scatter = float(1.4826 * np.median(np.abs(finite - score)))
    return {"score": score, "scatter": scatter,
            "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# I/O & plotting
# ---------------------------------------------------------------------------

def load_fits(path, ext):
    with fits.open(path) as hdul:
        return hdul[ext].data.astype(float)


def read_focus_keyword(path, keyword, ext):
    if not keyword:
        return None
    try:
        with fits.open(path) as hdul:
            for try_ext in (ext, 0):
                if try_ext < len(hdul) and keyword in hdul[try_ext].header:
                    return float(hdul[try_ext].header[keyword])
    except Exception:
        return None
    return None


def _subsample_diags(diag_list, n_keep):
    if not diag_list:
        return []
    if n_keep >= len(diag_list):
        return diag_list
    idxs = np.linspace(0, len(diag_list) - 1, n_keep, dtype=int)
    return [diag_list[i] for i in idxs]


def per_file_plot(name, diag_list, score, out_path, focus_val=None,
                  focus_key="FOCUS", plot_cols=5):
    """
    N rows x 2 cols.  Left: raw profile + envelopes; right: normalised.
    """
    diag_subset = _subsample_diags(diag_list, plot_cols)
    N = len(diag_subset)
    if N == 0:
        return

    fig_h = max(2.5 * N, 4.0)
    fig, axes = plt.subplots(N, 2, figsize=(15, fig_h),
                             sharex="col", squeeze=False)

    title = f"{name}   |   contrast = {score:.4f}"
    if focus_val is not None:
        title += f"   |   {focus_key} = {focus_val:g}"
    fig.suptitle(title, fontsize=11)

    for i, d in enumerate(diag_subset):
        ax_raw = axes[i, 0]
        ax_norm = axes[i, 1]
        col = d["col"]
        yy = np.arange(len(col))
        up_curve = d["up_fn"](yy)
        lo_curve = d["lo_fn"](yy)

        # Raw
        ax_raw.plot(yy, col, lw=0.4, color="0.3")
        ax_raw.plot(yy, up_curve, "r-", lw=0.9, label="upper (spline)")
        ax_raw.plot(yy, lo_curve, "g--", lw=0.9, label="lower (poly)")
        ax_raw.scatter(d["peak_ys"], d["peak_vals"], s=3, c="red",
                       alpha=0.6, zorder=4)
        ax_raw.scatter(d["gap_ys"], d["gap_vals"], s=35, c="green",
                       marker="x", zorder=5, label="auto-gaps")
        ax_raw.set_ylabel(f"x={d['x']}\ncounts", fontsize=9)
        ax_raw.grid(alpha=0.2)
        if i == 0:
            ax_raw.legend(loc="upper right", fontsize=7)
            ax_raw.set_title("raw + envelope fits", fontsize=10)

        # Normalised
        amp = up_curve - lo_curve
        safe = amp > 1e-6
        norm = np.full_like(col, np.nan, dtype=float)
        norm[safe] = (col[safe] - lo_curve[safe]) / amp[safe]
        ax_norm.plot(yy, norm, lw=0.4, color="0.3")
        ax_norm.axhline(1.0, color="r", ls=":", lw=0.7, alpha=0.6)
        ax_norm.axhline(0.0, color="g", ls=":", lw=0.7, alpha=0.6)
        ax_norm.scatter(d["peak_ys"], d["P"], s=3, c="red", alpha=0.6,
                        zorder=4)
        used = d["trough_mask"]
        ax_norm.scatter(d["trough_ys"][used], d["T"][used], s=6, c="blue",
                        alpha=0.7, zorder=4, label="troughs used")
        if (~used).any():
            ax_norm.scatter(d["trough_ys"][~used], d["T"][~used], s=6,
                            c="grey", alpha=0.4, zorder=3,
                            label="rejected")
        ax_norm.set_ylim(-0.2, 1.4)
        ax_norm.grid(alpha=0.2)
        if i == 0:
            ax_norm.legend(loc="upper right", fontsize=7)
            ax_norm.set_title("normalised", fontsize=10)
        if i == N - 1:
            ax_raw.set_xlabel("y (pixel)")
            ax_norm.set_xlabel("y (pixel)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def topographic_plot(name, diag_list, ny, nx, score, out_path,
                     focus_val=None, focus_key="FOCUS"):
    """
    2D heatmap of contrast across the frame.  Each (column_x, trough_y)
    sample contributes one contrast value; griddata interpolates to a
    regular grid for the topographic feel.
    """
    if not diag_list:
        return
    xs, ys, cs = [], [], []
    for d in diag_list:
        xc = d["x"]
        for ty, c, used in zip(d["trough_ys"], d["contrast"],
                               d["trough_mask"]):
            if used and np.isfinite(c):
                xs.append(xc)
                ys.append(ty)
                cs.append(c)
    xs = np.array(xs)
    ys = np.array(ys)
    cs = np.array(cs)
    if len(cs) < 10:
        return

    # Regular grid for interpolation
    grid_nx = 80
    grid_ny = 200
    xi = np.linspace(xs.min(), xs.max(), grid_nx)
    yi = np.linspace(ys.min(), ys.max(), grid_ny)
    GX, GY = np.meshgrid(xi, yi)
    GZ = griddata(np.column_stack([xs, ys]), cs, (GX, GY),
                  method="linear")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    extent = (xi.min(), xi.max(), yi.min(), yi.max())
    # Fixed scale [0, 1] so all topo plots are directly comparable.
    # 0 = troughs at floor (sharp); ~1 = troughs fill to peak (totally blurred).
    vmin, vmax = 0.0, 1.0
    # 'magma': dark = low contrast = sharp; bright = high = blurry.
    # Reads like a 'defocus thermometer'.
    im = ax.imshow(GZ, origin="lower", aspect="auto", extent=extent,
                   cmap="magma", vmin=vmin, vmax=vmax)
    ax.scatter(xs, ys, c=cs, s=4, cmap="magma", vmin=vmin, vmax=vmax,
               edgecolors="k", linewidths=0.15, alpha=0.7)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel("x (dispersion, pixel)")
    ax.set_ylabel("y (spatial, pixel)")
    title = f"{name}   |   median contrast = {score:.4f}"
    if focus_val is not None:
        title += f"   |   {focus_key} = {focus_val:g}"
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Contrast (1 - Michelson) -- lower = sharper")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def summary_focus_plot(rows, focus_key, out_path):
    pts = [(n, s, sc, fv) for (n, s, sc, fv) in rows
           if fv is not None and np.isfinite(s)]
    if len(pts) < 2:
        return False
    names = [p[0] for p in pts]
    scores = np.array([p[1] for p in pts])
    scatters = np.array([p[2] for p in pts])
    focus = np.array([p[3] for p in pts])
    order = np.argsort(focus)
    focus = focus[order]
    scores = scores[order]
    scatters = scatters[order]
    names = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(focus, scores, yerr=scatters, fmt="o-", lw=1, capsize=3,
                color="C0", markersize=5)
    best_i = int(np.argmin(scores))
    ax.scatter([focus[best_i]], [scores[best_i]], s=140, facecolor="none",
               edgecolor="red", lw=1.5, zorder=5, label="best (sharpest)")
    ax.set_xlabel(f"{focus_key} (header)")
    ax.set_ylabel("Contrast = 1 - Michelson  (lower = sharper)")
    ax.set_title(f"Focus sweep: contrast vs {focus_key}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.annotate(names[best_i], (focus[best_i], scores[best_i]),
                xytext=(8, -12), textcoords="offset points", fontsize=8,
                color="red")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", help="Folder of FITS images")
    p.add_argument("--reference", default=None,
                   help="Reference FITS for tracing")
    p.add_argument("--ext", type=int, default=1, help="FITS extension")
    p.add_argument("--gaps", default=None,
                   help="Optional comma-separated gap y-locations to use "
                        "for ALL columns (overrides auto-detection)")
    p.add_argument("--detect-pitch-factor", type=float, default=1.8,
                   help="Inter-peak spacing > this * median pitch counts "
                        "as an inter-bundle gap (default 1.8)")
    p.add_argument("--eval-cols", type=int, default=15,
                   help="Number of dispersion-axis columns sampled for "
                        "the metric")
    p.add_argument("--plot-cols", type=int, default=5,
                   help="How many of the eval columns appear in the "
                        "multi-panel per-file plot")
    p.add_argument("--order", type=int, default=3,
                   help="Polynomial order for trace fits")
    p.add_argument("--steps", type=int, default=30,
                   help="Tracing steps")
    p.add_argument("--upper-smooth-frac", type=float, default=0.025,
                   help="Spline smoothing for upper envelope, expressed "
                        "as expected fractional RMS scatter of peak "
                        "heights (default 0.025).")
    p.add_argument("--lower-order", type=int, default=2,
                   help="Polynomial order for lower (gap) envelope")
    p.add_argument("--focus-key", default="FOCUS",
                   help="FITS header keyword for focus value "
                        "(default FOCUS).  Empty = skip header read.")
    p.add_argument("--pattern", default="*.fits",
                   help="Glob pattern (default *.fits)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write CSV and plots "
                        "(default: <folder>/focus_results)")
    p.add_argument("--no-per-file-plots", action="store_true",
                   help="Skip per-file profile and topography plots")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not files:
        print(f"No files matching {args.pattern} in {args.folder}")
        return

    out_dir = args.output_dir or os.path.join(args.folder, "focus_results")
    os.makedirs(out_dir, exist_ok=True)

    ref_path = args.reference or files[0]
    print(f"Tracing reference: {ref_path}")
    ref = load_fits(ref_path, args.ext)
    coeffs = trace_fibres(ref, n_steps=args.steps, poly_order=args.order)
    print(f"  trace polynomials retained: {len(coeffs)}")

    ny, nx = ref.shape
    eval_cols = np.linspace(50, nx - 50, args.eval_cols, dtype=int)

    fixed_gap_ys = None
    if args.gaps:
        fixed_gap_ys = [float(g) for g in args.gaps.split(",")]
        print(f"  using fixed gap y-locations: {fixed_gap_ys}")
    else:
        # Show what auto-detection produces at the centre column for sanity
        cx = nx // 2
        peak_ys_ref = np.sort([np.polyval(c, cx) for c in coeffs])
        gaps_at_cx = auto_gaps(peak_ys_ref, ny,
                               pitch_factor=args.detect_pitch_factor)
        print(f"  auto-detected gaps at x={cx}: "
              f"{[f'{g:.1f}' for g in gaps_at_cx]}")

    print(f"\nMeasuring focus on {len(files)} files "
          f"({args.eval_cols} columns each)...")
    rows = []
    for f in files:
        name = os.path.basename(f)
        try:
            d = load_fits(f, args.ext)
        except Exception as e:
            print(f"  {name}: load failed ({e})")
            continue
        if d.shape != ref.shape:
            print(f"  {name}: shape mismatch, skipping")
            continue

        focus_val = read_focus_keyword(f, args.focus_key, args.ext)
        r = measure_focus(
            d, coeffs, eval_cols,
            fixed_gap_ys=fixed_gap_ys,
            upper_smooth_frac=args.upper_smooth_frac,
            lower_order=args.lower_order,
            detect_pitch_factor=args.detect_pitch_factor,
        )
        fv_str = f"{focus_val:.3f}" if focus_val is not None else "  --  "
        print(f"  {name:40s}  C = {r['score']:.4f}  "
              f"+/- {r['scatter']:.4f}   {args.focus_key} = {fv_str}")
        rows.append((name, r["score"], r["scatter"], focus_val))

        if not args.no_per_file_plots:
            stem = os.path.splitext(name)[0]
            per_file_plot(
                name, r["diagnostics"], r["score"],
                os.path.join(out_dir, f"profile_{stem}.png"),
                focus_val=focus_val, focus_key=args.focus_key,
                plot_cols=args.plot_cols,
            )
            topographic_plot(
                name, r["diagnostics"], ny, nx, r["score"],
                os.path.join(out_dir, f"topo_{stem}.png"),
                focus_val=focus_val, focus_key=args.focus_key,
            )

    # Best (sharpest) has the LOWEST contrast now -- sort ascending.
    rows.sort(key=lambda t: np.nan_to_num(t[1], nan=np.inf))
    print("\nRanking (best -> worst):")
    for i, (name, s, sc, fv) in enumerate(rows, 1):
        fv_str = f"{fv:.3f}" if fv is not None else "  --  "
        print(f"  {i:3d}. {name:40s}  C = {s:.4f}  +/- {sc:.4f}"
              f"   {args.focus_key} = {fv_str}")

    csv_path = os.path.join(out_dir, "focus_scores.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "filename", "contrast",
                    "scatter_mad_sigma", args.focus_key])
        for i, (name, s, sc, fv) in enumerate(rows, 1):
            fv_out = "" if fv is None else f"{fv:.6f}"
            w.writerow([i, name, f"{s:.6f}", f"{sc:.6f}", fv_out])
    print(f"\nWrote {csv_path}")

    summary_path = os.path.join(out_dir, "contrast_vs_focus.png")
    if summary_focus_plot(rows, args.focus_key, summary_path):
        print(f"Wrote {summary_path}")
    else:
        print(f"  (skipped contrast-vs-{args.focus_key} plot: "
              f"no/insufficient header values)")

    if not args.no_per_file_plots:
        print(f"Per-file plots in {out_dir}/profile_*.png and topo_*.png")


if __name__ == "__main__":
    main()
