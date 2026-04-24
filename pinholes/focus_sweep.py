"""
focus_sweep.py
--------------
Pinhole-trace FWHM focus sweep.

For each FITS frame in a folder:
    1. Locate 3 x-positions along the dispersion direction (avoiding CCD
       gaps by default).
    2. At each position, extract a vertical cut, find the 7 brightest peaks
       (the 7 pinhole traces), and fit a 1-D Gaussian to each.
    3. Median per-trace FWHM from the 3 samples  -> one number per pinhole.
    4. Median of those 7 numbers                 -> one frame-level FWHM.

Per-frame diagnostic PNGs are written to <folder>/diagnostics/:
    <stem>_trace.png      - image with the 7x3 fit locations marked
    <stem>_gaussfits.png  - 7-panel plot: each panel overlays the profile
                            data and Gaussian fits at all 3 x-positions

Across the folder the FOCUS header keyword is read and frame FWHM is
plotted vs focus, optionally with a parabola fit to estimate best focus.

Usage:
    python focus_sweep.py /path/to/folder
    python focus_sweep.py /path/to/folder --fit-parabola
    python focus_sweep.py /path/to/folder --n-positions 5
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # ~2.3548


# ===========================================================================
# FITS I/O
# ===========================================================================

def load_image(fits_path, ext=0):
    """Return 2-D image, falling back to the first 2-D HDU if ext is empty."""
    with fits.open(fits_path) as hdul:
        data = hdul[ext].data
        if data is None or np.ndim(data) != 2:
            for h in hdul:
                if h.data is not None and np.ndim(h.data) == 2:
                    return np.asarray(h.data, dtype=float)
            raise ValueError(f"No 2-D image in {fits_path}")
        return np.asarray(data, dtype=float)


def read_focus(fits_path, keyword="FOCUS"):
    """First occurrence of `keyword` across all HDUs, or None."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if keyword in hdu.header:
                try:
                    return float(hdu.header[keyword])
                except (TypeError, ValueError):
                    return None
    return None


# ===========================================================================
# X-position selection (gap-aware)
# ===========================================================================

def find_x_positions(data, n=3, threshold_frac=0.1):
    """
    Return n x-positions for FWHM measurement.

    Defaults to the centres of n equal segments of the image width
    (so n=3 => 1/6, 1/2, 5/6 of width). If any of these land in a CCD gap
    (identified by `median_along_y < threshold_frac * 95th-percentile`), the
    position is nudged to the nearest good column.
    """
    nx = data.shape[1]
    xprof = np.median(data, axis=0)
    threshold = threshold_frac * np.percentile(xprof, 95)
    good = xprof > threshold

    positions = []
    for i in range(n):
        target = int((i + 0.5) * nx / n)
        if 0 <= target < nx and good[target]:
            positions.append(target)
            continue
        # Walk outward to the nearest good column
        found = None
        for d in range(1, nx):
            if target + d < nx and good[target + d]:
                found = target + d
                break
            if target - d >= 0 and good[target - d]:
                found = target - d
                break
        positions.append(found if found is not None else target)
    return np.array(positions)


# ===========================================================================
# Peak detection + Gaussian fit
# ===========================================================================

def gaussian(x, amp, mu, sigma, offset):
    return offset + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def detect_trace_peaks(cut, expected_traces=7, min_distance=30):
    """
    Find the `expected_traces` most prominent peaks in a 1-D vertical cut.
    Returns the peak y-integer-indices, sorted bottom-to-top.
    """
    med = np.median(cut)
    mad = np.median(np.abs(cut - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(cut)
    prominence = max(3.0 * sigma, 1e-6)

    peaks, props = find_peaks(cut, distance=min_distance, prominence=prominence)
    if len(peaks) > expected_traces:
        order = np.argsort(props["prominences"])[::-1]
        peaks = peaks[order[:expected_traces]]
    return np.sort(peaks)


def fit_gaussian(cut, y_guess, half_window=10):
    """
    Fit Gaussian + constant offset to cut around y_guess.
    Returns a dict: y_guess, yy, prof, popt, fwhm.
    popt is kept even on failure for diagnostic plotting; fwhm is NaN if
    sigma falls outside a plausible range.
    """
    y_int = int(round(y_guess))
    y0 = max(0, y_int - half_window)
    y1 = min(len(cut), y_int + half_window + 1)
    yy = np.arange(y0, y1)
    prof = cut[y0:y1].astype(float)

    result = dict(y_guess=float(y_guess), yy=yy, prof=prof,
                  popt=None, fwhm=np.nan)

    if len(yy) < 5:
        return result
    offset0 = float(np.median(prof))
    amp0 = float(prof.max() - offset0)
    if amp0 <= 0:
        return result
    sigma0 = max(1.5, 0.1 * (yy[-1] - yy[0]))
    try:
        popt, _ = curve_fit(
            gaussian, yy, prof,
            p0=[amp0, y_guess, sigma0, offset0],
            maxfev=1000,
        )
    except (RuntimeError, ValueError):
        return result
    result["popt"] = popt
    sigma = abs(popt[2])
    halfw = 0.5 * (yy[-1] - yy[0])
    if 0.3 <= sigma <= halfw:
        result["fwhm"] = SIGMA_TO_FWHM * sigma
    return result


# ===========================================================================
# Per-frame processing
# ===========================================================================

def _safe_nanmedian(arr):
    a = np.asarray(arr, dtype=float)
    if not np.any(np.isfinite(a)):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(a))


def process_frame(fits_path, n_traces=7, n_positions=3,
                  col_halfwidth=2, fwhm_window=10, min_distance=30,
                  focus_kw="FOCUS"):
    data = load_image(fits_path)
    focus = read_focus(fits_path, focus_kw)
    x_positions = find_x_positions(data, n=n_positions)

    # For each x-position: cut, find peaks, fit each peak
    fits_by_position = []
    for x in x_positions:
        x0 = max(0, x - col_halfwidth)
        x1 = min(data.shape[1], x + col_halfwidth + 1)
        cut = np.median(data[:, x0:x1], axis=1)
        peaks = detect_trace_peaks(cut, expected_traces=n_traces,
                                   min_distance=min_distance)
        row = []
        for y in peaks:
            entry = fit_gaussian(cut, y, half_window=fwhm_window)
            entry["x"] = int(x)
            row.append(entry)
        # Pad if we got fewer peaks than expected (keeps array shapes consistent)
        while len(row) < n_traces:
            row.append(dict(x=int(x), y_guess=np.nan, yy=None, prof=None,
                            popt=None, fwhm=np.nan))
        fits_by_position.append(row)

    # Reshape: per_trace_fits[i] = list of n_positions entries for trace i
    # We rely on the peaks being y-sorted at each x, so the i-th peak is the
    # i-th pinhole across positions.
    per_trace_fits = [
        [fits_by_position[j][i] for j in range(n_positions)]
        for i in range(n_traces)
    ]

    per_trace_median = np.array([
        _safe_nanmedian([e["fwhm"] for e in row]) for row in per_trace_fits
    ])
    frame_median = _safe_nanmedian(per_trace_median)
    n_found = [sum(1 for e in fp if np.isfinite(e["y_guess"]))
               for fp in fits_by_position]

    return dict(
        path=Path(fits_path),
        focus=focus,
        x_positions=x_positions,
        per_trace_fits=per_trace_fits,
        per_trace_median=per_trace_median,
        frame_median=frame_median,
        n_traces_found=n_found,
        data=data,
    )


# ===========================================================================
# Plots
# ===========================================================================

def plot_trace_diagnostic(data, x_positions, per_trace_fits, path, out_png,
                          fwhm_window=10, box_halfwidth=40):
    """Image with fit locations marked as rectangles.

    Green rectangle = successful Gaussian fit
    Orange rectangle = fit returned but sigma was out-of-range
    Red cross = no fit attempted (no peak detected)
    """
    ny, nx = data.shape
    vmin, vmax = np.percentile(data, [5, 99])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto")
    for x in x_positions:
        ax.axvline(x, color="yellow", lw=0.8, alpha=0.4)

    for row in per_trace_fits:
        for entry in row:
            x = entry["x"]
            if entry["popt"] is not None:
                y = entry["popt"][1]
                color = "lime" if np.isfinite(entry["fwhm"]) else "orange"
            elif np.isfinite(entry["y_guess"]):
                y = entry["y_guess"]
                color = "red"
            else:
                continue
            rect = Rectangle(
                (x - box_halfwidth, y - fwhm_window),
                2 * box_halfwidth, 2 * fwhm_window,
                linewidth=1.2, edgecolor=color, facecolor="none",
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.plot(x, y, "+", color=color, ms=8, mew=1.2)

    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_xlabel("x [pix]"); ax.set_ylabel("y [pix]")
    ax.set_title(f"{path.name}: FWHM fit locations "
                 "(lime=good, orange=bad sigma, red=no fit)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_gaussian_fits(per_trace_fits, per_trace_median, path, out_png,
                       expected_traces=7):
    """One panel per trace; 3 profiles + Gaussian fits overlaid per panel,
    shifted so each fitted centre sits at 0 on the x-axis."""
    n_traces = len(per_trace_fits)
    n_panels = max(n_traces, expected_traces)
    ncols = 4 if n_panels >= 4 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for i in range(n_panels):
        ax = axes[i]
        if i >= n_traces:
            ax.set_visible(False)
            continue
        row = per_trace_fits[i]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(row)))
        for entry, color in zip(row, colors):
            if entry["yy"] is None or entry["prof"] is None:
                continue
            if entry["popt"] is not None:
                y0 = entry["popt"][1]
            else:
                y0 = entry["y_guess"]
            if not np.isfinite(y0):
                y0 = 0.5 * (entry["yy"][0] + entry["yy"][-1])

            fwhm_str = (f"FWHM={entry['fwhm']:.2f}"
                        if np.isfinite(entry["fwhm"]) else "FWHM=nan")
            ax.plot(entry["yy"] - y0, entry["prof"], "o",
                    color=color, ms=4, label=f"x={entry['x']}  {fwhm_str}")
            if entry["popt"] is not None:
                yy_fine = np.linspace(entry["yy"][0], entry["yy"][-1], 200)
                ax.plot(yy_fine - y0, gaussian(yy_fine, *entry["popt"]),
                        "-", color=color, lw=1.2, alpha=0.9)

        med = per_trace_median[i]
        med_str = f"{med:.3f}" if np.isfinite(med) else "nan"
        ax.set_title(f"Trace {i + 1}  (median FWHM = {med_str} pix)",
                     fontsize=10)
        ax.set_xlabel(r"$y - y_{\mathrm{centre}}$ [pix]")
        ax.set_ylabel("counts")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Gaussian fits: {path.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_focus_curve(results, out_png, fit_parabola=False, focus_kw="FOCUS"):
    good = [r for r in results
            if r["focus"] is not None and np.isfinite(r["frame_median"])]
    if not good:
        print("No frames have both FOCUS and a finite FWHM; skipping plot.")
        return

    focus = np.array([r["focus"] for r in good])
    frame_med = np.array([r["frame_median"] for r in good])
    order = np.argsort(focus)
    focus = focus[order]
    frame_med = frame_med[order]
    good = [good[i] for i in order]

    # (n_frames, n_traces) matrix of per-trace medians
    per_trace = np.array([r["per_trace_median"] for r in good])
    n_traces = per_trace.shape[1]
    colors = plt.cm.tab10(np.arange(n_traces) % 10)

    fig, ax = plt.subplots(figsize=(10, 6))

    # One coloured line per trace, skipping NaN samples
    for i in range(n_traces):
        vals = per_trace[:, i]
        mask = np.isfinite(vals)
        if not np.any(mask):
            continue
        style = "o-" if np.sum(mask) >= 2 else "o"
        ax.plot(focus[mask], vals[mask], style, color=colors[i],
                ms=5, lw=1.0, alpha=0.8, label=f"Trace {i + 1}")

    # Frame median on top
    ax.plot(focus, frame_med, "o-", color="black", ms=9, lw=2.2,
            label="Frame median", zorder=10)

    if fit_parabola and len(focus) >= 3:
        c = np.polyfit(focus, frame_med, 2)
        if c[0] > 0:
            xx = np.linspace(focus.min(), focus.max(), 300)
            yy = np.polyval(c, xx)
            best = -c[1] / (2 * c[0])
            best_fwhm = np.polyval(c, best)
            ax.plot(xx, yy, "--", color="red", lw=1.5,
                    label=f"Parabola: best {focus_kw}={best:.4g}, "
                          f"FWHM={best_fwhm:.3f} pix",
                    zorder=11)
            ax.axvline(best, color="red", lw=0.6, ls=":", alpha=0.6)
        else:
            print("Parabola has negative curvature; skipping overlay.")

    ax.set_xlabel(focus_kw)
    ax.set_ylabel("Median FWHM [pix]")
    ax.set_title(f"Focus sweep ({len(good)} frames)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote focus curve -> {out_png}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", type=Path, help="Folder of FITS frames")
    p.add_argument("--pattern", default="*.fits", help="Glob pattern")
    p.add_argument("--focus-kw", default="FOCUS", help="Focus header keyword")
    p.add_argument("--n-traces", type=int, default=7,
                   help="Expected number of pinhole traces")
    p.add_argument("--n-positions", type=int, default=3,
                   help="Number of x-positions for FWHM measurement")
    p.add_argument("--col-halfwidth", type=int, default=2,
                   help="Half-width of column median for each cut")
    p.add_argument("--fwhm-window", type=int, default=10,
                   help="Half-width in y for Gaussian fit [pix]")
    p.add_argument("--min-distance", type=int, default=30,
                   help="Min y-separation between peaks")
    p.add_argument("--fit-parabola", action="store_true",
                   help="Fit parabola to FWHM vs focus, report best focus")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Skip per-frame diagnostic PNGs")
    p.add_argument("--diag-dir", type=Path, default=None,
                   help="Where diagnostic PNGs go (default: <folder>/diagnostics)")
    args = p.parse_args()

    files = sorted(args.folder.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.folder}")

    diag_dir = args.diag_dir or (args.folder / "diagnostics")
    if not args.no_diagnostics:
        diag_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        print(f"\n=== {f.name} ===")
        try:
            r = process_frame(
                f,
                n_traces=args.n_traces,
                n_positions=args.n_positions,
                col_halfwidth=args.col_halfwidth,
                fwhm_window=args.fwhm_window,
                min_distance=args.min_distance,
                focus_kw=args.focus_kw,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        focus_str = f"{r['focus']:.4g}" if r["focus"] is not None else "None"
        print(f"  {args.focus_kw}={focus_str}")
        print(f"  x-positions: {list(r['x_positions'])}")
        print(f"  peaks found per x-position: {r['n_traces_found']}")
        per_str = ", ".join(
            f"{v:.3f}" if np.isfinite(v) else "nan"
            for v in r["per_trace_median"]
        )
        print(f"  per-trace median FWHM [pix]: {per_str}")
        print(f"  frame median FWHM [pix]: {r['frame_median']:.3f}")

        if not args.no_diagnostics:
            stem = f.stem
            plot_trace_diagnostic(
                r["data"], r["x_positions"], r["per_trace_fits"],
                f, diag_dir / f"{stem}_trace.png",
                fwhm_window=args.fwhm_window,
            )
            plot_gaussian_fits(
                r["per_trace_fits"], r["per_trace_median"],
                f, diag_dir / f"{stem}_gaussfits.png",
                expected_traces=args.n_traces,
            )
            print(f"  wrote {stem}_trace.png, {stem}_gaussfits.png")

        r.pop("data", None)         # drop large arrays before keeping
        results.append(r)

    out_npz = args.folder / "focus_sweep.npz"
    np.savez(
        out_npz,
        filenames=np.array([str(r["path"]) for r in results]),
        focus=np.array([r["focus"] if r["focus"] is not None else np.nan
                        for r in results]),
        per_trace_median=np.array([r["per_trace_median"] for r in results],
                                  dtype=object),
        frame_median=np.array([r["frame_median"] for r in results]),
        x_positions=np.array([r["x_positions"] for r in results], dtype=object),
    )
    print(f"\nSaved results -> {out_npz}")

    plot_focus_curve(results, args.folder / "focus_sweep.png",
                     fit_parabola=args.fit_parabola, focus_kw=args.focus_kw)


if __name__ == "__main__":
    main()
