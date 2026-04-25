#!/usr/bin/env python3
"""
focus_assess.py
---------------
Rank IFU flat-field images by focus quality using peak-to-trough contrast,
normalised against upper (peak) and lower (gap) throughput envelopes.

Strategy
========
1. Trace ~N fibres on a single reference image (or first file in the folder).
   Trace polynomials are reused for every image in the folder, on the
   assumption that defocus does not translate the traces.
2. For each image, sample several columns spanning the dispersion axis. At
   each sampled column:
     * read each trace's peak value as the max in a +/-1 pixel window around
       the polynomial-predicted y
     * read each inter-trace trough value as the min in a +/-1 pixel window
       around the midpoint between adjacent peaks
     * read each gap value as a small median around the user-supplied gap y
     * fit upper envelope (peaks vs y, low-order poly) and lower envelope
       (gaps vs y, low-order poly)
     * normalise: (profile - lower_env) / (upper_env - lower_env)
     * compute Michelson contrast (P - T) / (P + T) on the normalised
       peak / adjacent-trough pairs
3. Median across peak-pairs within a column; median across columns ->
   one focus score per image. Higher = sharper.

Usage
=====
    python focus_assess.py /path/to/flats \
        --ext 1 --gaps 75,1070,2005,2945,3980 --plot
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
    """Max of col in [y-half, y+half], clipped to bounds."""
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
# Focus measurement
# ---------------------------------------------------------------------------

def measure_focus(data, trace_coeffs, gap_ys, eval_cols,
                  upper_order=4, lower_order=2,
                  col_window=2, peak_half=1, trough_half=1, gap_half=2,
                  gap_buffer=12.0, pitch_factor=1.5):
    """
    Parameters of note
    ------------------
    gap_buffer : float
        Reject any trough whose midpoint lies within this many pixels of any
        user-supplied gap y-location.  Such midpoints fall inside the gap
        rather than between two real traces and would pollute the metric.
    pitch_factor : float
        Reject any trough whose flanking peaks are separated by more than
        pitch_factor * median_pitch.  Backstop for unflagged dropouts.
    """
    ny, nx = data.shape
    gap_ys = np.asarray(sorted(gap_ys), dtype=float)

    per_col_contrasts = []
    diagnostics = []  # for optional plotting

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

        # Mask trough samples that land in / near gap regions, or span
        # an unusually large pitch (= unflagged dropout).
        med_pitch = np.median(pitches)
        too_wide = pitches > pitch_factor * med_pitch
        near_gap = np.array([
            np.any(np.abs(t - gap_ys) < gap_buffer) for t in trough_ys
        ])
        trough_mask = ~(too_wide | near_gap)

        # Envelopes
        up_o = min(upper_order, len(peak_ys) - 1)
        lo_o = min(lower_order, len(gap_ys) - 1)
        up_coef = np.polyfit(peak_ys, peak_vals, up_o)
        lo_coef = np.polyfit(gap_ys, gap_vals, lo_o)

        up_p = np.polyval(up_coef, peak_ys)
        lo_p = np.polyval(lo_coef, peak_ys)
        up_t = np.polyval(up_coef, trough_ys)
        lo_t = np.polyval(lo_coef, trough_ys)

        amp_p = up_p - lo_p
        amp_t = up_t - lo_t
        with np.errstate(divide="ignore", invalid="ignore"):
            P = (peak_vals - lo_p) / amp_p
            T = (trough_vals - lo_t) / amp_t

        # Pair each trough with the dimmer of its two flanking peaks
        # (defensive: if envelope wobbles a peak above 1, this avoids
        # rewarding that with extra contrast).
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
            "x": x, "col": col, "peak_ys": peak_ys, "peak_vals": peak_vals,
            "trough_ys": trough_ys, "trough_vals": trough_vals,
            "gap_ys": gap_ys, "gap_vals": gap_vals,
            "up_coef": up_coef, "lo_coef": lo_coef,
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
# I/O & driver
# ---------------------------------------------------------------------------

def load_fits(path, ext):
    with fits.open(path) as hdul:
        return hdul[ext].data.astype(float)


def diagnostic_plot(name, diag, out_path):
    if not diag:
        return
    # Pick the middle column for the plot
    d = diag[len(diag) // 2]
    col = d["col"]
    yy = np.arange(len(col))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(yy, col, lw=0.5, color="0.3", label="profile")
    ax.plot(yy, np.polyval(d["up_coef"], yy), "r--", lw=1,
            label="upper envelope")
    ax.plot(yy, np.polyval(d["lo_coef"], yy), "g--", lw=1,
            label="lower envelope")
    ax.scatter(d["peak_ys"], d["peak_vals"], s=4, c="red", alpha=0.6,
               label="peak samples")
    ax.scatter(d["gap_ys"], d["gap_vals"], s=40, c="green", marker="x",
               label="gap samples")
    ax.set_title(f"{name}  (column x={d['x']})")
    ax.set_xlabel("y (pixel)")
    ax.set_ylabel("counts")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="Folder of FITS images")
    p.add_argument("--reference", default=None,
                   help="Reference FITS for tracing "
                        "(default: first match in folder)")
    p.add_argument("--ext", type=int, default=1,
                   help="FITS extension (default 1)")
    p.add_argument("--gaps", default="75,1070,2005,2945,3980",
                   help="Comma-separated y-locations of inter-bundle gaps")
    p.add_argument("--eval-cols", type=int, default=15,
                   help="Number of dispersion-axis columns to sample")
    p.add_argument("--order", type=int, default=3,
                   help="Polynomial order for trace fits")
    p.add_argument("--steps", type=int, default=30,
                   help="Tracing steps")
    p.add_argument("--upper-order", type=int, default=4,
                   help="Polynomial order for upper (peak) envelope fit")
    p.add_argument("--lower-order", type=int, default=2,
                   help="Polynomial order for lower (gap) envelope fit; "
                        "with 5 gaps, 2 is safe, 3 max")
    p.add_argument("--pattern", default="*.fits",
                   help="Glob pattern (default *.fits)")
    p.add_argument("--output", default="focus_scores.csv")
    p.add_argument("--plot", action="store_true",
                   help="Save normalised-profile diagnostic PNG for "
                        "best and worst image")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not files:
        print(f"No files matching {args.pattern} in {args.folder}")
        return

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
    results = []
    for f in files:
        try:
            d = load_fits(f, args.ext)
        except Exception as e:
            print(f"  {os.path.basename(f)}: load failed ({e})")
            continue
        if d.shape != ref.shape:
            print(f"  {os.path.basename(f)}: shape mismatch, skipping")
            continue
        r = measure_focus(d, coeffs, gap_ys, eval_cols,
                          upper_order=args.upper_order,
                          lower_order=args.lower_order)
        print(f"  {os.path.basename(f):40s}  C = {r['score']:.4f}  "
              f"+/- {r['scatter']:.4f}")
        results.append((os.path.basename(f), r))

    # Rank: best (highest contrast) first
    results.sort(key=lambda t: np.nan_to_num(t[1]["score"], nan=-np.inf),
                 reverse=True)

    print("\nRanking (best -> worst):")
    for i, (name, r) in enumerate(results, 1):
        print(f"  {i:3d}. {name:40s}  C = {r['score']:.4f}  "
              f"+/- {r['scatter']:.4f}")

    out_csv = args.output
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "filename", "contrast", "scatter_across_cols",
                    "n_columns_used"])
        for i, (name, r) in enumerate(results, 1):
            w.writerow([i, name, f"{r['score']:.6f}",
                        f"{r['scatter']:.6f}", len(r["per_column"])])
    print(f"\nWrote {out_csv}")

    if args.plot and results:
        best_name, best_r = results[0]
        worst_name, worst_r = results[-1]
        diagnostic_plot(best_name, best_r["diagnostics"],
                        os.path.join(args.folder, "focus_best.png"))
        diagnostic_plot(worst_name, worst_r["diagnostics"],
                        os.path.join(args.folder, "focus_worst.png"))
        print(f"Wrote focus_best.png and focus_worst.png in {args.folder}")


if __name__ == "__main__":
    main()
