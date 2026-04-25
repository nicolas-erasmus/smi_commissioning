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
     * read each gap value as a small median around the user-supplied gap y
     * fit upper envelope through the peak values with a smoothing spline
       (higher-resolution than a low-order polynomial), and a low-order
       polynomial through the gap values for the lower envelope
     * normalise: (profile - lower_env) / (upper_env - lower_env)
     * compute Michelson contrast (P - T) / (P + T) on the normalised
       peak / adjacent-trough pairs
3. Median across peak-pairs within a column; median across columns ->
   one focus score per image.  Higher = sharper.
4. Outputs (in <folder>/focus_results/ unless --output-dir given):
     * focus_scores.csv  -- rank, filename, contrast, scatter, FOCUS hdr
     * profile_<name>.png -- two-panel diagnostic per image
                             (raw profile + envelopes / normalised profile)
     * contrast_vs_focus.png -- summary plot (if FOCUS keyword present)

Usage
=====
    python focus_assess.py /path/to/flats \\
        --ext 1 --gaps 75,1070,2005,2945,3980 \\
        --focus-key FOCUS
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
from scipy.interpolate import UnivariateSpline


# ---------------------------------------------------------------------------
# Tracing (lifted with minor edits from trace_ifu_lines.py)
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
    return np.asarray(coeffs)  # (n_traces, poly_order+1)


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
# Envelope fits
# ---------------------------------------------------------------------------

def fit_upper_envelope(peak_ys, peak_vals, smooth_frac=0.025, k=3):
    """
    Smoothing-spline fit through (peak_ys, peak_vals).

    Parameters
    ----------
    smooth_frac : float
        Expected fractional RMS scatter of the peak heights about the
        true envelope.  Sets the spline's smoothing parameter
        s = N * (smooth_frac * median(peak_vals)) ** 2.
        Smaller -> more wiggle; larger -> more rigid.
    k : int
        Spline degree (3 = cubic).

    Returns
    -------
    callable spline(y)
    """
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
    """Polynomial through gap-region floor values."""
    o = min(order, len(gap_ys) - 1)
    c = np.polyfit(gap_ys, gap_vals, o)
    return lambda yy, _c=c: np.polyval(_c, yy)


# ---------------------------------------------------------------------------
# Focus measurement
# ---------------------------------------------------------------------------

def measure_focus(data, trace_coeffs, gap_ys, eval_cols,
                  upper_smooth_frac=0.025, lower_order=2,
                  col_window=2, peak_half=1, trough_half=1, gap_half=2,
                  gap_buffer=12.0, pitch_factor=1.5):
    ny, nx = data.shape
    gap_ys = np.asarray(sorted(gap_ys), dtype=float)

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
        gap_vals = np.array([_median_in_window(col, y, gap_half)
                             for y in gap_ys])

        med_pitch = np.median(pitches)
        too_wide = pitches > pitch_factor * med_pitch
        near_gap = np.array([
            np.any(np.abs(t - gap_ys) < gap_buffer) for t in trough_ys
        ])
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
            mich = np.where(
                np.isfinite(P_pair) & np.isfinite(T) & (denom > 0),
                (P_pair - T) / denom,
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
            "P": P, "T": T,
        })

    if not per_col_contrasts:
        return {"score": np.nan, "scatter": np.nan,
                "per_column": [], "diagnostics": []}

    per_col_med = np.array([np.nanmedian(c) for c in per_col_contrasts])
    return {
        "score": float(np.nanmedian(per_col_med)),
        "scatter": float(np.nanstd(per_col_med)),
        "per_column": per_col_med.tolist(),
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# I/O & plotting
# ---------------------------------------------------------------------------

def load_fits(path, ext):
    with fits.open(path) as hdul:
        return hdul[ext].data.astype(float)


def read_focus_keyword(path, keyword, ext):
    """Try the SCI extension first, then fall back to primary HDU."""
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


def per_file_plot(name, diag_list, score, out_path, focus_val=None,
                  focus_key="FOCUS"):
    """
    Two-panel plot for a single image (uses the median-x diagnostic).
    Top: raw profile + envelope fits + sample points.
    Bottom: normalised profile + peak/trough markers.
    """
    if not diag_list:
        return
    d = diag_list[len(diag_list) // 2]
    col = d["col"]
    yy = np.arange(len(col))

    up_curve = d["up_fn"](yy)
    lo_curve = d["lo_fn"](yy)

    amp = up_curve - lo_curve
    safe = amp > 1e-6
    norm = np.full_like(col, np.nan, dtype=float)
    norm[safe] = (col[safe] - lo_curve[safe]) / amp[safe]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    ax0, ax1 = axes

    ax0.plot(yy, col, lw=0.5, color="0.3", label="profile")
    ax0.plot(yy, up_curve, "r-", lw=1.0, label="upper envelope (spline)")
    ax0.plot(yy, lo_curve, "g--", lw=1.0, label="lower envelope (poly)")
    ax0.scatter(d["peak_ys"], d["peak_vals"], s=4, c="red", alpha=0.6,
                label="peak samples", zorder=4)
    ax0.scatter(d["gap_ys"], d["gap_vals"], s=45, c="green", marker="x",
                label="gap samples", zorder=5)
    ax0.set_ylabel("counts")
    title = f"{name}   |   contrast = {score:.4f}"
    if focus_val is not None:
        title += f"   |   {focus_key} = {focus_val:g}"
    title += f"   (col x = {d['x']})"
    ax0.set_title(title)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.2)

    ax1.plot(yy, norm, lw=0.5, color="0.3", label="normalised profile")
    ax1.axhline(1.0, color="r", ls=":", lw=0.8, alpha=0.6)
    ax1.axhline(0.0, color="g", ls=":", lw=0.8, alpha=0.6)
    ax1.scatter(d["peak_ys"], d["P"], s=4, c="red", alpha=0.6,
                label="peaks (norm)", zorder=4)
    used = d["trough_mask"]
    ax1.scatter(d["trough_ys"][used], d["T"][used], s=8, c="blue",
                alpha=0.7, label="troughs used", zorder=4)
    if (~used).any():
        ax1.scatter(d["trough_ys"][~used], d["T"][~used], s=8, c="grey",
                    alpha=0.4, label="troughs rejected", zorder=3)
    ax1.set_xlabel("y (pixel)")
    ax1.set_ylabel("normalised")
    ax1.set_ylim(-0.2, 1.4)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def summary_focus_plot(rows, focus_key, out_path):
    """rows is list of (name, score, scatter, focus_val)."""
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
    best_i = int(np.argmax(scores))
    ax.scatter([focus[best_i]], [scores[best_i]], s=140, facecolor="none",
               edgecolor="red", lw=1.5, zorder=5, label="best")
    ax.set_xlabel(f"{focus_key} (header)")
    ax.set_ylabel("Michelson contrast (median)")
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
                   help="Reference FITS for tracing "
                        "(default: first match in folder)")
    p.add_argument("--ext", type=int, default=1, help="FITS extension")
    p.add_argument("--gaps", default="75,1070,2005,2945,3980",
                   help="Comma-separated y-locations of inter-bundle gaps")
    p.add_argument("--eval-cols", type=int, default=15,
                   help="Number of dispersion-axis columns to sample")
    p.add_argument("--order", type=int, default=3,
                   help="Polynomial order for trace fits")
    p.add_argument("--steps", type=int, default=30,
                   help="Tracing steps")
    p.add_argument("--upper-smooth-frac", type=float, default=0.025,
                   help="Spline smoothing for upper envelope, expressed "
                        "as expected fractional RMS scatter of peak heights "
                        "(default 0.025).  Smaller -> more wiggle.")
    p.add_argument("--lower-order", type=int, default=2,
                   help="Polynomial order for lower (gap) envelope; "
                        "max sensible value with 5 gaps is 3")
    p.add_argument("--focus-key", default="FOCUS",
                   help="FITS header keyword for focus value "
                        "(default FOCUS).  Empty = skip header read.")
    p.add_argument("--pattern", default="*.fits",
                   help="Glob pattern (default *.fits)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write CSV and plots "
                        "(default: <folder>/focus_results)")
    p.add_argument("--no-per-file-plots", action="store_true",
                   help="Skip the two-panel per-file diagnostic PNGs")
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
    gap_ys = [float(g) for g in args.gaps.split(",")]

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
            d, coeffs, gap_ys, eval_cols,
            upper_smooth_frac=args.upper_smooth_frac,
            lower_order=args.lower_order,
        )
        fv_str = f"{focus_val:.3f}" if focus_val is not None else "  --  "
        print(f"  {name:40s}  C = {r['score']:.4f}  "
              f"+/- {r['scatter']:.4f}   {args.focus_key} = {fv_str}")
        rows.append((name, r["score"], r["scatter"], focus_val))

        if not args.no_per_file_plots:
            stem = os.path.splitext(name)[0]
            png = os.path.join(out_dir, f"profile_{stem}.png")
            per_file_plot(name, r["diagnostics"], r["score"], png,
                          focus_val=focus_val, focus_key=args.focus_key)

    rows.sort(key=lambda t: np.nan_to_num(t[1], nan=-np.inf), reverse=True)

    print("\nRanking (best -> worst):")
    for i, (name, s, sc, fv) in enumerate(rows, 1):
        fv_str = f"{fv:.3f}" if fv is not None else "  --  "
        print(f"  {i:3d}. {name:40s}  C = {s:.4f}  +/- {sc:.4f}"
              f"   {args.focus_key} = {fv_str}")

    csv_path = os.path.join(out_dir, "focus_scores.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "filename", "contrast",
                    "scatter_across_cols", args.focus_key])
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
        print(f"Per-file plots in {out_dir}/profile_*.png")


if __name__ == "__main__":
    main()
