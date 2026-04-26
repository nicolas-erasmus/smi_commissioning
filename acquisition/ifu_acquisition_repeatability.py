#!/usr/bin/env python3
"""
IFU fiber throughput from imaging-mode flats.

Image 1 = IFU + spectrograph in imaging mode (dispersion element OUT) under a
flat lamp. Produces a vertical line of ~244 dots, one per fiber. We use it to
locate each fiber dot and measure each fiber's FWHM.

Image 2 = a second image (e.g. on-sky or alternative illumination). Forced
circular-aperture photometry is performed at Image 1's (x, y) positions, with
each aperture's diameter set to APERTURE_SCALE * FWHM_i (default 1.0; i.e. the
aperture diameter equals the per-fiber FWHM measured from Image 1).

Pipeline
--------
1. Sum Image 1 along Y -> 1D x-profile. Argmax gives the column where the
   "vertical dotted line" sits.
2. Take a median Y-profile in a narrow band around that column.
3. Detect fibers. Two backends:
     peaks (default): scipy.signal.find_peaks on the column profile +
       per-source 2D Gaussian fit. Robust for closely-packed lines of dots.
     sep            : sep.extract on a band-cropped, background-subtracted
       image. Tweak --deblend-nthresh / --deblend-cont / --filter-off
       to fight blending of close neighbours.
4. Per-source FWHM:
     peaks: 2.355 * sqrt(sigma_x * sigma_y) from the 2D fit.
     sep  : 2.355 * sqrt(a * b) from second-moment axes.
5. Validate N == 244 (warn + continue otherwise; --strict to abort).
6. Sort sources by Y, sort final_data.csv by slit_x, zip -> fiber IDs.
   This matches the convention in extract_flux_throughput.py.
7. sep.sum_circle on Image 2 at each (x_i, y_i) with r_i = scale * fwhm_i / 2.
8. relflux = flux2 / max(flux2).
9. Three-panel broken-axis sky map (style from plot_throughput_map() in
   extract_flux_throughput.py): black panel face, gray cmap, white-edged
   markers, white ID/trace labels.

Usage
-----
    python ifu_imaging_throughput.py image1.fits image2.fits \\
        --csv final_data.csv --aperture-scale 1.0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

try:
    import sep
except ImportError:
    sys.exit("Could not import `sep`. Install with `pip install sep`.")


N_FIBERS_EXPECTED = 244


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_fits(path):
    """Load a FITS image as a contiguous, native-byte-order float32 array.
    SEP rejects big-endian buffers (FITS native), so we normalise here."""
    data = fits.getdata(path)
    if data.dtype.byteorder not in ("=", "|"):
        data = data.astype(data.dtype.newbyteorder("="))
    return np.ascontiguousarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 1-2: locate the dot column
# ---------------------------------------------------------------------------

def find_dot_column(data, smooth=5):
    """Sum along Y; return (x_peak, y_collapsed). x_peak is the column index
    where the vertical line of fiber dots sits."""
    y_collapsed = data.sum(axis=0)
    if smooth > 1:
        kern = np.ones(smooth) / smooth
        smoothed = np.convolve(y_collapsed, kern, mode="same")
    else:
        smoothed = y_collapsed
    x_peak = int(np.argmax(smoothed))
    return x_peak, y_collapsed


def column_profile(data, x_center, half_band=15, mode="sum"):
    """1D Y profile from a vertical band of columns around x_center.

    Modes
    -----
    sum    : per-row sum across the band. Robust to tilt up to ±half_band:
             whichever column the dot lands in at row Y, its flux is
             added in. Default.
    max    : per-row max. Even more contrasty, slightly noisier.
    median : per-row median. ONLY useful when the line is truly vertical;
             with tilt, most of the band is background at any given row
             and the median washes out the peaks.
    """
    nx = data.shape[1]
    a = max(0, x_center - half_band)
    b = min(nx, x_center + half_band + 1)
    band = data[:, a:b]
    if mode == "median":
        return np.median(band, axis=1)
    if mode == "max":
        return np.max(band, axis=1)
    return np.sum(band, axis=1)


def refine_x_at_peak(data, x_center, y_peak, half_band=15, y_half=1):
    """For a detected Y peak, find the actual X column of the fiber dot
    inside the X band. Handles tilt: the source can be anywhere in the
    band [x_center-half_band, x_center+half_band].

    y_half=1 averages 3 rows around y_peak before taking argmax."""
    ny, nx = data.shape
    ya = max(0, int(round(y_peak)) - y_half)
    yb = min(ny, int(round(y_peak)) + y_half + 1)
    xa = max(0, int(x_center) - half_band)
    xb = min(nx, int(x_center) + half_band + 1)
    if yb <= ya or xb <= xa:
        return float(x_center)
    row_band = data[ya:yb, xa:xb].sum(axis=0)
    return float(xa + int(np.argmax(row_band)))


# ---------------------------------------------------------------------------
# Step 3-4: detect sources and measure FWHM
#
# Two backends are available and selected via --detect-method:
#
#  peaks (default) : scipy.signal.find_peaks on the 1D column profile, then
#                    a 2D Gaussian fit per peak for sub-pixel centroid and
#                    FWHM. Robust for closely-packed lines of dots where
#                    sep.extract tends to merge neighbours at the
#                    detection threshold.
#
#  sep             : sep.extract on a band-cropped, background-subtracted
#                    image. FWHM = 2.355 * sqrt(a * b) from second moments.
#                    Use --deblend-nthresh / --deblend-cont / --filter-off
#                    to fight blending of close pairs.
# ---------------------------------------------------------------------------

def _gauss2d(coords, A, mux, muy, sx, sy, C):
    """Axis-aligned 2D Gaussian + constant. Used by curve_fit only."""
    x, y = coords
    return A * np.exp(-0.5 * ((x - mux) ** 2 / sx ** 2
                              + (y - muy) ** 2 / sy ** 2)) + C


def fit_gaussian_2d(data, x0_init, y0_init, x_window=5, y_window=4):
    """Fit an axis-aligned 2D Gaussian-with-constant on a stamp around
    (x0_init, y0_init).

    Stamp size is (2*x_window+1) x (2*y_window+1). Tight y_window keeps
    closely-packed neighbours out of the fit; wide x_window helps the
    centroid since the dot column is the only feature in X.

    Returns
    -------
    (x_centroid, y_centroid, fwhm, peak_amp, ok)
    """
    ny, nx = data.shape
    xc, yc = int(round(x0_init)), int(round(y0_init))
    xa = max(0, xc - x_window)
    xb = min(nx, xc + x_window + 1)
    ya = max(0, yc - y_window)
    yb = min(ny, yc + y_window + 1)
    stamp = data[ya:yb, xa:xb]
    if stamp.size < 9:
        return float(x0_init), float(y0_init), float("nan"), \
               float(np.nanmax(stamp) if stamp.size else 0.0), False

    yy, xx = np.mgrid[ya:yb, xa:xb]
    coords = (xx.ravel().astype(float), yy.ravel().astype(float))
    z = stamp.ravel().astype(float)

    A0 = float(np.nanmax(stamp) - np.nanmin(stamp))
    if A0 <= 0:
        A0 = 1.0
    p0 = [A0, float(x0_init), float(y0_init), 1.5, 1.5,
          float(np.nanmin(stamp))]
    s_lo = 0.3
    s_hi_x = max(1.5, float(x_window))
    s_hi_y = max(1.5, float(y_window))
    bounds = (
        [0.0,    xa - 0.5, ya - 0.5, s_lo, s_lo, -np.inf],
        [np.inf, xb - 0.5, yb - 0.5, s_hi_x, s_hi_y, np.inf],
    )

    try:
        popt, _ = curve_fit(_gauss2d, coords, z, p0=p0, bounds=bounds,
                            maxfev=600)
        A, mux, muy, sx, sy, _C = popt
        fwhm = 2.355 * np.sqrt(abs(sx) * abs(sy))
        return float(mux), float(muy), float(fwhm), float(A), True
    except Exception:
        return float(x0_init), float(y0_init), float("nan"), \
               float(np.nanmax(stamp)), False


def detect_fibers_peaks(data, x_center, smooth_sigma=1.2, peak_distance=4,
                        prom_pct=10, x_window=5, y_window=4,
                        column_half_band=15, profile_mode="sum"):
    """Find fiber dots via 1D peak detection on a column profile, then
    refine each peak with a 2D Gaussian fit.

    Tilt handling: the column profile sums across a ±column_half_band
    band, so the dotted line can drift in X by up to that much without
    the peaks washing out. Each detected peak is then re-located in X
    (per-row argmax inside the band) before the 2D fit, so a tilted
    slit is handled fiber-by-fiber.

    Designed for a vertical line of closely-packed point sources where
    sep.extract tends to merge neighbours.
    """
    yprof = column_profile(data, x_center, half_band=column_half_band,
                           mode=profile_mode)
    smoothed = gaussian_filter1d(yprof, smooth_sigma)
    prominence = float(np.nanpercentile(yprof, prom_pct))

    peaks_y, _ = find_peaks(smoothed, distance=peak_distance,
                            prominence=prominence)

    rows = []
    n_failed = 0
    for py in peaks_y:
        # Refine X for this peak — handles slit tilt
        x_init = refine_x_at_peak(data, x_center, py,
                                  half_band=column_half_band, y_half=1)
        x_fit, y_fit, fwhm, amp, ok = fit_gaussian_2d(
            data, x_init, py, x_window=x_window, y_window=y_window
        )
        if not ok:
            n_failed += 1
        rows.append({
            "x":     x_fit,
            "y":     y_fit,
            "x_init": x_init,
            "fwhm":  fwhm,
            "peak":  amp,
            "fit_ok": ok,
            # Schema-compatible placeholders so the downstream pipeline
            # doesn't care which detection backend was used.
            "a":     (fwhm / 2.355) if np.isfinite(fwhm) else float("nan"),
            "b":     (fwhm / 2.355) if np.isfinite(fwhm) else float("nan"),
            "theta": 0.0,
            "flux_sep": float("nan"),
        })
    src = pd.DataFrame(rows).reset_index(drop=True)

    # Don't drop failed fits — that would shift the cross-match by one.
    # Replace their FWHM with the median so forced photometry still works.
    if not src.empty:
        med = float(np.nanmedian(src["fwhm"]))
        src["fwhm"] = src["fwhm"].fillna(med)
        if n_failed:
            print(f"  WARNING: {n_failed} of {len(src)} 2D fits failed; "
                  f"used median FWHM ({med:.2f} px) for those rows")
        x_drift = float(src["x"].max() - src["x"].min())
        print(f"  X drift across fibers (slit tilt): {x_drift:.2f} px "
              f"(min x={src['x'].min():.2f}, max x={src['x'].max():.2f})")

    return src


def detect_fibers_sep(data, x_center, x_half_band=30, thresh=5.0, minarea=3,
                      deblend_cont=0.005, deblend_nthresh=32,
                      filter_off=False):
    """Run sep.extract on a vertical band centred on x_center.

    For closely-packed sources, try `--filter-off`, raise
    `--deblend-nthresh` (e.g. 64) and lower `--deblend-cont` (e.g. 1e-5).
    """
    nx = data.shape[1]
    crop_x0 = max(0, x_center - x_half_band)
    crop_x1 = min(nx, x_center + x_half_band + 1)
    sub = np.ascontiguousarray(data[:, crop_x0:crop_x1])

    bkg = sep.Background(sub)
    sub_bgsub = sub - bkg.back()

    extract_kwargs = dict(
        thresh=thresh,
        err=bkg.globalrms,
        minarea=minarea,
        deblend_cont=deblend_cont,
        deblend_nthresh=deblend_nthresh,
    )
    if filter_off:
        extract_kwargs["filter_kernel"] = None
    objs = sep.extract(sub_bgsub, **extract_kwargs)

    a = objs["a"]
    b = objs["b"]
    fwhm = 2.355 * np.sqrt(a * b)

    src = pd.DataFrame({
        "x":      objs["x"] + crop_x0,
        "y":      objs["y"],
        "fwhm":   fwhm,
        "peak":   objs["peak"],
        "fit_ok": np.ones(len(objs), dtype=bool),
        "a":      a,
        "b":      b,
        "theta":  objs["theta"],
        "flux_sep": objs["flux"],
    }).reset_index(drop=True)

    return src


# ---------------------------------------------------------------------------
# Step 6: cross-match to fiber IDs
# ---------------------------------------------------------------------------

def match_to_fibers(sources, fiber_csv_path):
    """Sort sources by Y, sort fiber_df by slit_x, then zip by index. Same
    convention as extract_flux_throughput.py.

    Returns the joined DataFrame and the sorted fiber_df."""
    fiber_df = pd.read_csv(fiber_csv_path)
    required = {"slit_x", "sky_x", "sky_y", "ID"}
    missing = required - set(fiber_df.columns)
    if missing:
        raise ValueError(
            f"{fiber_csv_path} is missing required columns: {sorted(missing)}. "
            f"Found: {list(fiber_df.columns)}"
        )

    fiber_df_sorted = fiber_df.sort_values("slit_x").reset_index(drop=True)
    src_sorted = sources.sort_values("y").reset_index(drop=True)

    n = min(len(src_sorted), len(fiber_df_sorted))
    out = src_sorted.iloc[:n].copy().reset_index(drop=True)
    out["ID"]    = fiber_df_sorted["ID"].iloc[:n].astype(int).values
    out["trace"] = np.arange(1, n + 1, dtype=int)
    out["sky_x"] = fiber_df_sorted["sky_x"].iloc[:n].values
    out["sky_y"] = fiber_df_sorted["sky_y"].iloc[:n].values
    return out, fiber_df_sorted


# ---------------------------------------------------------------------------
# Step 7: forced photometry
# ---------------------------------------------------------------------------

def forced_aperture_photometry(data, x, y, fwhm, aperture_scale=1.0):
    """sep.sum_circle at (x, y) with radius = aperture_scale * fwhm / 2.

    Returns flux, fluxerr, flag, radii (per-source), background object."""
    bkg = sep.Background(data)
    data_bgsub = data - bkg.back()
    r = aperture_scale * fwhm / 2.0
    flux, fluxerr, flag = sep.sum_circle(
        data_bgsub, x, y, r, err=bkg.globalrms, gain=1.0,
    )
    return flux, fluxerr, flag, r, bkg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_xprofile(y_collapsed, x_peak, out_path):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y_collapsed, lw=0.7, color="k")
    ax.axvline(x_peak, color="crimson", lw=0.8, ls="--",
               label=f"peak at x = {x_peak}")
    ax.set_xlabel("X (column)")
    ax.set_ylabel("Sum along Y")
    ax.set_title("Image 1 — Y-collapsed X profile")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_yprofile(yprof, sources, out_path):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(yprof, lw=0.6, color="k")
    ys = sources["y"].astype(int).clip(0, len(yprof) - 1).values
    ax.plot(sources["y"].values, yprof[ys], "rx", ms=4, mew=0.6,
            label=f"{len(sources)} sep sources")
    ax.set_xlabel("Y (row)")
    ax.set_ylabel("Median column flux")
    ax.set_title("Image 1 — column profile at the dot column")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_image1_detection(data, sources, radii, out_path):
    """Image 1 with sep apertures overlaid."""
    fig, ax = plt.subplots(figsize=(8, 16))
    pos = data[data > 0]
    if pos.size:
        vmin, vmax = np.nanpercentile(pos, [5, 99.5])
    else:
        vmin, vmax = 1, 100
    ax.imshow(data, origin="lower", cmap="magma", aspect="auto",
              norm=LogNorm(vmin=max(vmin, 1e-3), vmax=vmax))
    for (_, row), r in zip(sources.iterrows(), radii):
        circ = plt.Circle((row["x"], row["y"]), r, color="cyan",
                          fill=False, lw=0.4)
        ax.add_patch(circ)
    ax.set_title(
        f"Image 1 detections — N={len(sources)},  "
        f"<FWHM>={sources['fwhm'].median():.2f} px,  "
        f"FWHM range [{sources['fwhm'].min():.2f}, {sources['fwhm'].max():.2f}]",
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_throughput_map(df, out_path, title, ratio_col="relflux",
                        cbar_label="Relative flux (flux / max)",
                        cmap="gray"):
    """Three-panel broken-axis sky map.

    Layout copied from plot_throughput_map() in extract_flux_throughput.py:
    same xlims, widths, panel ratios, break marks, ID/trace annotations,
    p2-p98 robust color limits. Visual twist: panel facecolor is BLACK and
    markers have WHITE edges with a grayscale-shaded fill keyed to relflux.
    """
    xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
    widths = [b - a for a, b in xlims]
    ymin = df["sky_y"].min() - 1
    ymax = df["sky_y"].max() + 1

    vmin, vmax = np.nanpercentile(df[ratio_col], [2, 98])

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 8), sharey=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.05},
    )
    norm = plt.Normalize(vmin, vmax)

    sc = None
    for i, ax in enumerate(axes):
        ax.set_facecolor("black")
        a, b = xlims[i]
        sub = df[(df["sky_x"] >= a) & (df["sky_x"] <= b)]
        sc = ax.scatter(
            sub["sky_x"], sub["sky_y"], c=sub[ratio_col],
            s=260, cmap=cmap, edgecolors="white", linewidths=0.7,
            norm=norm, zorder=2,
        )
        for _, row in sub.iterrows():
            ax.annotate(str(int(row["ID"])),
                        xy=(row["sky_x"], row["sky_y"]),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=5.5, ha="center", va="center",
                        fontweight="bold", color="white")
            ax.annotate(f"t{int(row['trace'])}",
                        xy=(row["sky_x"], row["sky_y"]),
                        xytext=(0, -3.5), textcoords="offset points",
                        fontsize=4.5, ha="center", va="center",
                        color="white", alpha=0.7)

        ax.set_xlim(a, b)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, ls="--", lw=0.5, color="gray")
        ax.tick_params(axis="both", labelsize=9)

        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", which="both", left=False)
        if i < 2:
            ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Sky Y", fontsize=11)
    axes[1].set_xlabel("Sky X", fontsize=11)

    # Break marks on the inner spines
    d = 0.012
    kw = dict(color="k", clip_on=False, lw=1.0)
    for i in (0, 1):  # right side of axes 0 and 1
        ax = axes[i]
        tk = dict(kw, transform=ax.transAxes)
        ax.plot([1 - d, 1 + d], [-d, +d], **tk)
        ax.plot([1 - d, 1 + d], [1 - d, 1 + d], **tk)
    for i in (1, 2):  # left side of axes 1 and 2
        ax = axes[i]
        tk = dict(kw, transform=ax.transAxes)
        ax.plot([-d, +d], [-d, +d], **tk)
        ax.plot([-d, +d], [1 - d, 1 + d], **tk)

    cbar = fig.colorbar(sc, ax=axes, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=11)

    med = np.nanmedian(df[ratio_col])
    fig.suptitle(
        f"{title}   "
        f"(N={len(df)}, median={med:.3f}, "
        f"p2-98=[{vmin:.3f}, {vmax:.3f}])",
        fontsize=13, y=0.96,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image1", help="IFU imaging-mode flat (FITS)")
    parser.add_argument("image2",
                        help="Second image for forced photometry (FITS)")
    parser.add_argument("--csv", default="final_data.csv",
                        help="fiber metadata table (default: final_data.csv)")
    parser.add_argument("--aperture-scale", type=float, default=1.0,
                        help="forced aperture DIAMETER in units of FWHM "
                             "(default 1.0 -> radius = 0.5 * FWHM)")
    parser.add_argument("--detect-method", choices=("peaks", "sep"),
                        default="peaks",
                        help="source detection backend. 'peaks' (default): "
                             "1D find_peaks on column profile + per-source "
                             "2D Gaussian fit. Robust for closely-packed "
                             "lines of dots. 'sep': sep.extract; tweak "
                             "deblend params if neighbours blend.")
    parser.add_argument("--detect-thresh", type=float, default=5.0,
                        help="(sep mode) sep.extract detection threshold "
                             "in sigma (default 5.0)")
    parser.add_argument("--detect-minarea", type=int, default=3,
                        help="(sep mode) sep.extract minarea (default 3)")
    parser.add_argument("--deblend-cont", type=float, default=0.005,
                        help="(sep mode) sep.extract deblend_cont "
                             "(default 0.005). Lower = more aggressive "
                             "deblending of close pairs.")
    parser.add_argument("--deblend-nthresh", type=int, default=32,
                        help="(sep mode) sep.extract deblend_nthresh "
                             "(default 32). Higher = more thresholds for "
                             "the deblender.")
    parser.add_argument("--filter-off", action="store_true",
                        help="(sep mode) disable the convolution filter in "
                             "sep.extract. Helps separate closely-packed "
                             "neighbours.")
    parser.add_argument("--peak-distance", type=int, default=4,
                        help="(peaks mode) min peak-to-peak separation in "
                             "pixels for find_peaks (default 4; should be "
                             "well below the actual fiber spacing)")
    parser.add_argument("--peak-prominence-pct", type=float, default=10.0,
                        help="(peaks mode) percentile of the column profile "
                             "used as the prominence threshold (default 10)")
    parser.add_argument("--peak-smooth-sigma", type=float, default=1.2,
                        help="(peaks mode) Gaussian smoothing sigma applied "
                             "to the profile before find_peaks (default 1.2)")
    parser.add_argument("--column-band", type=int, default=15,
                        help="(peaks mode) half-width (px) of the X band "
                             "used to build the Y profile and to refine "
                             "each peak's X position. Must be >= the slit "
                             "tilt (max-X minus min-X across fibers, "
                             "halved). Default 15.")
    parser.add_argument("--profile-mode", choices=("sum", "max", "median"),
                        default="sum",
                        help="(peaks mode) how to collapse the X band into "
                             "a 1D Y profile. 'sum' (default) and 'max' are "
                             "robust to slit tilt; 'median' is only safe if "
                             "the line is truly vertical.")
    parser.add_argument("--y-window", type=int, default=4,
                        help="(peaks mode) half-height of the 2D fit stamp "
                             "in pixels (default 4 -> 9 px tall stamps; "
                             "must be < half the fiber peak-to-peak spacing)")
    parser.add_argument("--x-window", type=int, default=5,
                        help="(peaks mode) half-width of the 2D fit stamp "
                             "in pixels (default 5)")
    parser.add_argument("--x-band", type=int, default=30,
                        help="(sep mode) half-width (px) of the vertical "
                             "band cropped around the dot column for sep "
                             "extraction (default 30)")
    parser.add_argument("--n-expected", type=int, default=N_FIBERS_EXPECTED,
                        help=f"expected fiber count (default {N_FIBERS_EXPECTED})")
    parser.add_argument("--strict", action="store_true",
                        help="abort if detected source count != --n-expected")
    parser.add_argument("--outdir", default=".",
                        help="output directory (default cwd)")
    parser.add_argument("--cmap", default="gray",
                        help="matplotlib colormap for the throughput map "
                             "(default 'gray'; try 'inferno' or 'magma' for "
                             "more visible low-flux fibers)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image1))[0]
    def out(name):
        return os.path.join(args.outdir, f"{stem}_{name}")

    # ---- Load images
    print(f"[load] {args.image1}")
    data1 = load_fits(args.image1)
    print(f"[load] {args.image2}")
    data2 = load_fits(args.image2)
    print(f"  shape image1 = {data1.shape}")
    print(f"  shape image2 = {data2.shape}")
    if data1.shape != data2.shape:
        print("  WARNING: image shapes differ; forced photometry assumes a "
              "shared pixel coordinate system.")

    # ---- Step 1: locate the dot column
    x_peak, y_collapsed = find_dot_column(data1)
    print(f"[step 1] dot column at x = {x_peak}")
    plot_xprofile(y_collapsed, x_peak, out("xprofile.pdf"))

    # ---- Step 2: column profile (same band/mode as detection so the
    # yprofile diagnostic matches what the peak finder actually saw)
    yprof = column_profile(data1, x_peak,
                           half_band=args.column_band,
                           mode=args.profile_mode)

    # ---- Step 3-4: source detection
    if args.detect_method == "peaks":
        print(f"[step 3] peak-finding mode: column-profile find_peaks "
              f"(distance={args.peak_distance}, "
              f"prom_pct={args.peak_prominence_pct}, "
              f"smooth_sigma={args.peak_smooth_sigma}, "
              f"band=±{args.column_band}, profile={args.profile_mode}) "
              f"+ 2D Gaussian fit "
              f"(x_window=±{args.x_window}, y_window=±{args.y_window})")
        sources = detect_fibers_peaks(
            data1, x_peak,
            smooth_sigma=args.peak_smooth_sigma,
            peak_distance=args.peak_distance,
            prom_pct=args.peak_prominence_pct,
            x_window=args.x_window,
            y_window=args.y_window,
            column_half_band=args.column_band,
            profile_mode=args.profile_mode,
        )
    else:  # sep
        print(f"[step 3] sep.extract on band x = [{x_peak - args.x_band}, "
              f"{x_peak + args.x_band}], thresh={args.detect_thresh}σ, "
              f"minarea={args.detect_minarea}, "
              f"deblend_cont={args.deblend_cont}, "
              f"deblend_nthresh={args.deblend_nthresh}, "
              f"filter={'OFF' if args.filter_off else 'on'}")
        sources = detect_fibers_sep(
            data1, x_peak,
            x_half_band=args.x_band,
            thresh=args.detect_thresh,
            minarea=args.detect_minarea,
            deblend_cont=args.deblend_cont,
            deblend_nthresh=args.deblend_nthresh,
            filter_off=args.filter_off,
        )

    n_det = len(sources)
    print(f"  detected {n_det} sources")
    print(f"  median FWHM = {sources['fwhm'].median():.3f} px "
          f"(min={sources['fwhm'].min():.2f}, max={sources['fwhm'].max():.2f})")

    if n_det != args.n_expected:
        if args.detect_method == "peaks":
            hint = ("Tweak --peak-distance, --peak-prominence-pct, or "
                    "--peak-smooth-sigma. Inspect <stem>_yprofile.pdf.")
        else:
            hint = ("Try --filter-off, raise --deblend-nthresh (e.g. 64), "
                    "lower --deblend-cont (e.g. 1e-5), or switch to "
                    "--detect-method peaks.")
        msg = (f"detected {n_det} sources but expected {args.n_expected}. "
               + hint)
        if args.strict:
            sys.exit(f"[abort] {msg}")
        print(f"  WARNING: {msg}")

    plot_yprofile(yprof, sources, out("yprofile.pdf"))

    # ---- Step 5-6: cross-match to fiber IDs
    print(f"[step 6] cross-matching to {args.csv}")
    sources, _ = match_to_fibers(sources, args.csv)

    # ---- Self-photometry on Image 1 (for sanity)
    flux1, fluxerr1, flag1, radii, _ = forced_aperture_photometry(
        data1, sources["x"].values, sources["y"].values,
        sources["fwhm"].values, aperture_scale=args.aperture_scale,
    )
    sources["flux_image1"] = flux1
    sources["flux_image1_err"] = fluxerr1
    sources["flag_image1"] = flag1

    plot_image1_detection(data1, sources, radii, out("image1_detection.pdf"))

    # ---- Step 7: forced photometry on Image 2
    print(f"[step 7] forced photometry on Image 2 with "
          f"r_i = {args.aperture_scale:.2f} * FWHM_i / 2")
    flux2, fluxerr2, flag2, _, _ = forced_aperture_photometry(
        data2, sources["x"].values, sources["y"].values,
        sources["fwhm"].values, aperture_scale=args.aperture_scale,
    )
    sources["flux_image2"] = flux2
    sources["flux_image2_err"] = fluxerr2
    sources["flag_image2"] = flag2

    # ---- Step 8: relative flux
    fmax = float(np.nanmax(flux2))
    if not np.isfinite(fmax) or fmax <= 0:
        print("  WARNING: max(flux2) is not positive — relflux undefined.")
        sources["relflux"] = np.nan
    else:
        sources["relflux"] = flux2 / fmax
    print(f"  max image2 flux = {fmax:.3e}; "
          f"median relflux = {np.nanmedian(sources['relflux']):.3f}")

    # ---- Save table
    csv_out = out("fiber_positions.csv")
    sources.to_csv(csv_out, index=False)
    print(f"[write] {csv_out}")

    # ---- Step 9: 3-panel broken-axis sky map
    pdf_out = out("throughput_map.pdf")
    plot_throughput_map(
        sources, pdf_out,
        title=f"IFU forced-photometry throughput  ({stem})",
        ratio_col="relflux",
        cbar_label=(f"Relative flux  (flux / max,  "
                    f"D = {args.aperture_scale:.2f} × FWHM)"),
        cmap=args.cmap,
    )
    print(f"[write] {pdf_out}")
    print("[done]")


if __name__ == "__main__":
    main()
