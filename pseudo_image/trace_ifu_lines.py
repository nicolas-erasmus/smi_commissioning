"""
trace_ifu_lines.py
------------------
Trace horizontal spectral lines in an IFU flat-field FITS image.

Strategy
    1. Take vertical "cuts" every `step` pixels along x (optionally median-
       collapsing a small window of columns for SNR).
    2. Run scipy.signal.find_peaks on each cut to locate the ~250 lines.
    3. Refine each peak to sub-pixel accuracy with a 3-point parabolic fit.
    4. Link peaks across cuts into traces (one trace per fibre/line) using
       nearest-neighbour matching from a central reference cut, walking
       outward in both directions.
    5. Fit a polynomial (default order 2) to each trace: y(x) = a0 + a1*x + a2*x^2
    6. Plot the image with fitted traces overlaid for a quick visual check.

Usage
    python trace_ifu_lines.py flat.fits
    python trace_ifu_lines.py flat.fits --step 100 --order 2 --min-distance 6

Outputs
    * <stem>_traces.npz   - saved trace data + polynomial coefficients
    * <stem>_traces.png   - diagnostic plot
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Peak detection helpers
# ---------------------------------------------------------------------------

def refine_peak_parabolic(y, idx):
    """
    Refine an integer-pixel peak to sub-pixel accuracy using a 3-point
    parabolic (quadratic) interpolation around the maximum.

    Returns the refined y-coordinate. Falls back to the integer index if
    the peak is on the array edge.
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    ym1, y0, yp1 = y[idx - 1], y[idx], y[idx + 1]
    denom = (ym1 - 2.0 * y0 + yp1)
    if denom == 0:
        return float(idx)
    delta = 0.5 * (ym1 - yp1) / denom
    # Guard against nonsense (e.g. very noisy triplets)
    if not np.isfinite(delta) or abs(delta) > 1.0:
        return float(idx)
    return idx + delta


def find_peaks_in_cut(cut, min_distance=6, prominence=None, height=None,
                      detrend_width=51, nsigma=3.0, return_detrended=False):
    """
    Find peaks in a 1-D vertical cut.

    If `detrend_width` > 0, subtract a median-filtered baseline first. This
    is essential when the image has strong large-scale structure (bright/dark
    bands) that would otherwise dominate any global noise estimate and push
    the prominence threshold sky-high.

    The prominence threshold (if not set by hand) is `nsigma` times the
    MAD-based sigma of the detrended cut.

    Returns sub-pixel-refined y-positions (float array). If return_detrended
    is True, also returns (detrended_cut, threshold) for debug plotting.
    """
    if detrend_width > 0:
        # median_filter with mode='reflect' avoids edge artefacts. Size must
        # span several line spacings so the filter sits on the background
        # between lines, not on the lines themselves.
        baseline = median_filter(cut, size=detrend_width, mode="reflect")
        work = cut - baseline
    else:
        work = cut

    if prominence is None:
        med = np.median(work)
        mad = np.median(np.abs(work - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(work)
        prominence = max(nsigma * sigma, 1e-6)

    peaks, _ = find_peaks(
        work,
        distance=min_distance,
        prominence=prominence,
        height=height,
    )
    refined = np.array([refine_peak_parabolic(work, p) for p in peaks])
    if return_detrended:
        return refined, work, prominence
    return refined


# ---------------------------------------------------------------------------
# Trace linking
# ---------------------------------------------------------------------------

def link_traces(peaks_per_cut, x_samples, match_tol=4.0):
    """
    Link peaks across cuts into continuous traces.

    Parameters
    ----------
    peaks_per_cut : list of 1-D arrays of float
        peaks_per_cut[i] = sub-pixel y-positions found in cut at x_samples[i].
    x_samples : 1-D array of int
        The x-coordinate of each cut.
    match_tol : float
        Max allowed |dy| between adjacent cuts to be considered the same trace.

    Returns
    -------
    traces : list of dict
        Each trace has keys 'x' (array) and 'y' (array) of matched samples.
    """
    n_cuts = len(peaks_per_cut)
    if n_cuts == 0:
        return []

    # Start from the cut with the most peaks as the reference — this maximises
    # the chance of seeding every real trace.
    ref_idx = int(np.argmax([len(p) for p in peaks_per_cut]))

    # Seed: one trace per peak in the reference cut
    traces = [
        {"x": [x_samples[ref_idx]], "y": [y]}
        for y in peaks_per_cut[ref_idx]
    ]

    def extend(direction):
        """direction = +1 (rightwards) or -1 (leftwards)"""
        start = ref_idx + direction
        stop = n_cuts if direction == 1 else -1
        for i in range(start, stop, direction):
            available = list(peaks_per_cut[i])
            if not available:
                continue
            # Greedy nearest-neighbour match. Each available peak is used at
            # most once per cut.
            used = np.zeros(len(available), dtype=bool)
            # For each trace, predict y using the last point (could upgrade to
            # a local linear extrapolation, but this is fine for well-sampled
            # flats).
            for tr in traces:
                last_y = tr["y"][-1] if direction == 1 else tr["y"][0]
                dy = np.abs(np.array(available) - last_y)
                dy[used] = np.inf
                j = int(np.argmin(dy))
                if dy[j] <= match_tol:
                    used[j] = True
                    if direction == 1:
                        tr["x"].append(x_samples[i])
                        tr["y"].append(available[j])
                    else:
                        tr["x"].insert(0, x_samples[i])
                        tr["y"].insert(0, available[j])

    extend(+1)
    extend(-1)

    # Convert to arrays
    for tr in traces:
        tr["x"] = np.asarray(tr["x"], dtype=float)
        tr["y"] = np.asarray(tr["y"], dtype=float)
    return traces


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def trace_ifu_flat(
    fits_path,
    ext=0,
    step=100,
    median_width=5,
    min_distance=6,
    prominence=None,
    detrend_width=51,
    nsigma=3.0,
    match_tol=4.0,
    poly_order=2,
    min_points=None,
    debug_cut=None,
    debug_png=None,
):
    """
    Full pipeline. Returns (data, traces, coeffs, x_samples).

    coeffs[i] is the polynomial coefficient array for traces[i], as returned
    by np.polyfit (highest power first), or None if the trace was too short.
    """
    with fits.open(fits_path) as hdul:
        hdu = hdul[ext]
        data = hdu.data
        # Some instruments put the image in extension 1 (or later) and leave
        # the primary HDU header-only. If the requested extension has no
        # usable 2-D image, fall back to the first one that does.
        if data is None or np.ndim(data) != 2:
            found = None
            for i, h in enumerate(hdul):
                if h.data is not None and np.ndim(h.data) == 2:
                    found = i
                    break
            if found is None:
                raise ValueError(
                    f"No 2-D image found in {fits_path}. HDU shapes: "
                    + ", ".join(
                        f"{i}:{np.shape(h.data)}" for i, h in enumerate(hdul)
                    )
                )
            if found != ext:
                print(f"Note: extension {ext} has no 2-D image; "
                      f"using extension {found} instead.")
            data = hdul[found].data
        data = np.asarray(data, dtype=float)

    ny, nx = data.shape

    # --- 1. Build cut locations ------------------------------------------------
    half = median_width // 2
    x_samples = np.arange(step // 2, nx, step)

    # --- 2. Peak detection in each cut ----------------------------------------
    peaks_per_cut = []
    for xi in x_samples:
        x0 = max(0, xi - half)
        x1 = min(nx, xi + half + 1)
        cut = np.median(data[:, x0:x1], axis=1)
        peaks = find_peaks_in_cut(
            cut, min_distance=min_distance, prominence=prominence,
            detrend_width=detrend_width, nsigma=nsigma,
        )
        peaks_per_cut.append(peaks)

    counts = [len(p) for p in peaks_per_cut]
    print(f"Peaks per cut: min={min(counts)}, median={int(np.median(counts))}, "
          f"max={max(counts)} across {len(counts)} cuts")

    # Optional: dump a diagnostic of one cut so you can tune parameters.
    if debug_cut is not None:
        xi = int(debug_cut)
        x0 = max(0, xi - half)
        x1 = min(nx, xi + half + 1)
        cut = np.median(data[:, x0:x1], axis=1)
        peaks, detrended, thr = find_peaks_in_cut(
            cut, min_distance=min_distance, prominence=prominence,
            detrend_width=detrend_width, nsigma=nsigma,
            return_detrended=True,
        )
        yy = np.arange(ny)
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(yy, cut, "k-", lw=0.6, label="raw cut")
        if detrend_width > 0:
            baseline = median_filter(cut, size=detrend_width, mode="reflect")
            axes[0].plot(yy, baseline, "r-", lw=0.8, label=f"median baseline (w={detrend_width})")
        axes[0].set_ylabel("counts")
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"Cut at x={xi}: {len(peaks)} peaks found")

        axes[1].plot(yy, detrended, "k-", lw=0.6, label="detrended")
        axes[1].axhline(thr, color="orange", lw=0.8, ls="--",
                        label=f"prominence threshold = {thr:.2f}")
        for p in peaks:
            axes[1].axvline(p, color="cyan", lw=0.4, alpha=0.7)
        axes[1].set_xlabel("y [pix]")
        axes[1].set_ylabel("counts - baseline")
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(debug_png, dpi=150)
        plt.close(fig)
        print(f"Wrote cut diagnostic -> {debug_png}")

    # --- 3. Link into traces --------------------------------------------------
    traces = link_traces(peaks_per_cut, x_samples, match_tol=match_tol)
    print(f"Linked {len(traces)} candidate traces")

    # --- 4. Polynomial fits ---------------------------------------------------
    if min_points is None:
        min_points = poly_order + 2  # need at least order+1, +1 for slack
    coeffs = []
    keep_traces = []
    for tr in traces:
        if len(tr["x"]) >= min_points:
            c = np.polyfit(tr["x"], tr["y"], poly_order)
            coeffs.append(c)
            keep_traces.append(tr)
    print(f"Kept {len(keep_traces)} traces with >= {min_points} samples "
          f"(polynomial order {poly_order})")

    return data, keep_traces, coeffs, x_samples


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_result(data, traces, coeffs, x_samples, out_png):
    ny, nx = data.shape
    vmin, vmax = np.percentile(data, [5, 99])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto")

    xx = np.arange(nx)
    for tr, c in zip(traces, coeffs):
        yy = np.polyval(c, xx)
        ax.plot(xx, yy, "-", color="red", lw=0.6, alpha=0.8)
        ax.plot(tr["x"], tr["y"], ".", color="cyan", ms=2.5)

    # Mark where the cuts were taken
    for xi in x_samples:
        ax.axvline(xi, color="yellow", lw=0.3, alpha=0.25)

    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"{len(traces)} traces (red = poly fit, cyan = peaks)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote diagnostic plot -> {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fits_path", type=Path, help="Input flat-field FITS file")
    p.add_argument("--ext", type=int, default=0, help="FITS extension (default 0)")
    p.add_argument("--step", type=int, default=100, help="Cut spacing in pixels")
    p.add_argument("--median-width", type=int, default=5,
                   help="Width of median-combined column window at each cut")
    p.add_argument("--min-distance", type=int, default=6,
                   help="Min separation between peaks (pixels)")
    p.add_argument("--prominence", type=float, default=None,
                   help="Peak prominence threshold (default: auto = nsigma*sigma_MAD)")
    p.add_argument("--detrend-width", type=int, default=51,
                   help="Median-filter width for baseline subtraction. "
                        "Should be wider than the line spacing but narrower "
                        "than the macro structure. 0 disables detrending.")
    p.add_argument("--nsigma", type=float, default=3.0,
                   help="Auto-prominence threshold = nsigma * MAD-sigma")
    p.add_argument("--match-tol", type=float, default=4.0,
                   help="Max |dy| between adjacent cuts for trace linking")
    p.add_argument("--order", type=int, default=2,
                   help="Polynomial order for trace fit (default 2)")
    p.add_argument("--debug-cut", type=int, default=None,
                   help="If given, also save a diagnostic plot of the cut "
                        "at this x-coordinate (useful for tuning).")
    args = p.parse_args()

    stem = args.fits_path.with_suffix("")
    npz_path = Path(str(stem) + "_traces.npz")
    png_path = Path(str(stem) + "_traces.png")
    debug_png = Path(str(stem) + f"_cut_x{args.debug_cut}.png") \
        if args.debug_cut is not None else None

    data, traces, coeffs, x_samples = trace_ifu_flat(
        args.fits_path,
        ext=args.ext,
        step=args.step,
        median_width=args.median_width,
        min_distance=args.min_distance,
        prominence=args.prominence,
        detrend_width=args.detrend_width,
        nsigma=args.nsigma,
        match_tol=args.match_tol,
        poly_order=args.order,
        debug_cut=args.debug_cut,
        debug_png=debug_png,
    )

    # Save traces. Use object dtype because traces have different lengths.
    np.savez(
        npz_path,
        coeffs=np.array(coeffs),                       # (N, order+1)
        trace_x=np.array([t["x"] for t in traces], dtype=object),
        trace_y=np.array([t["y"] for t in traces], dtype=object),
        x_samples=x_samples,
        poly_order=args.order,
    )
    print(f"Wrote trace data -> {npz_path}")

    plot_result(data, traces, coeffs, x_samples, png_path)


if __name__ == "__main__":
    main()
