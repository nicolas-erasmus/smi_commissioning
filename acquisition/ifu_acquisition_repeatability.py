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
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

try:
    import sep
except ImportError:
    sys.exit("Could not import `sep`. Install with `pip install sep`.")

try:
    from PIL import Image as PILImage
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


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


def plot_image1_detection(data, sources, radii, out_path,
                          x_band=50, vmin=None, vmax=None):
    """Image 1 with sep apertures overlaid.

    EQUAL aspect so apertures render as round circles, not ellipses.
    LINEAR stretch with min/max (or 1-99.5 percentiles by default to
    keep hot pixels from washing out the dots) -- this matches the
    "scale image min/max and you can see them" behaviour.
    The X axis is cropped to a band around the dot column so the
    apertures are visible at a sensible scale on the page.
    """
    ny, nx = data.shape
    x_center = int(np.median(sources["x"]))
    xa = max(0, x_center - x_band)
    xb = min(nx, x_center + x_band + 1)
    crop = data[:, xa:xb]

    if vmin is None:
        vmin = float(np.nanpercentile(crop, 1.0))
    if vmax is None:
        vmax = float(np.nanpercentile(crop, 99.5))

    crop_w = xb - xa
    fig_h = 16.0
    fig_w = max(2.5, min(8.0, fig_h * crop_w / ny))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(
        crop, origin="lower", cmap="magma", aspect="equal",
        vmin=vmin, vmax=vmax,
        extent=[xa - 0.5, xb - 0.5, -0.5, ny - 0.5],
        interpolation="nearest",
    )
    for (_, row), r in zip(sources.iterrows(), radii):
        circ = plt.Circle((row["x"], row["y"]), r, color="cyan",
                          fill=False, lw=0.5)
        ax.add_patch(circ)

    ax.set_xlim(xa - 0.5, xb - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title(
        f"Image 1 detections — N={len(sources)},  "
        f"<FWHM>={sources['fwhm'].median():.2f} px,  "
        f"FWHM range [{sources['fwhm'].min():.2f}, {sources['fwhm'].max():.2f}],  "
        f"X drift={sources['x'].max() - sources['x'].min():.2f} px",
        fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_throughput_map(df, out_path, title, ratio_col="relflux",
                        cbar_label="Relative flux (flux / max)",
                        cmap="gray", vmin=None, vmax=None, dpi=150,
                        frame_label=None):
    """Three-panel broken-axis sky map.

    Layout copied from plot_throughput_map() in extract_flux_throughput.py:
    same xlims, widths, panel ratios, break marks, ID/trace annotations,
    p2-p98 robust color limits. Visual twist: panel facecolor is BLACK and
    markers have WHITE edges with a grayscale-shaded fill keyed to relflux.

    For folder/GIF mode, pass explicit vmin/vmax to keep the colorbar
    consistent across frames; pass frame_label to caption the frame.
    """
    xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
    widths = [b - a for a, b in xlims]
    ymin = df["sky_y"].min() - 1
    ymax = df["sky_y"].max() + 1

    auto_lo, auto_hi = np.nanpercentile(df[ratio_col], [2, 98])
    if vmin is None:
        vmin = auto_lo
    if vmax is None:
        vmax = auto_hi

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
    suptitle = (f"{title}   "
                f"(N={len(df)}, median={med:.3f}, "
                f"p2-98=[{vmin:.3f}, {vmax:.3f}])")
    if frame_label:
        suptitle = f"{frame_label}\n{suptitle}"
    fig.suptitle(suptitle, fontsize=13, y=0.97)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Folder mode: per-frame processing, repeatability, centroid-trail plot, GIF
# ---------------------------------------------------------------------------

def collect_image2_paths(image2_arg):
    """If image2_arg is a directory, glob FITS files in it. Otherwise
    return [image2_arg]."""
    if os.path.isdir(image2_arg):
        patterns = ("*.fits", "*.fit", "*.fts",
                    "*.FITS", "*.FIT", "*.FTS",
                    "*.fits.gz", "*.fit.gz")
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(image2_arg, pat)))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No FITS files in directory: {image2_arg}")
        return paths
    if not os.path.isfile(image2_arg):
        raise FileNotFoundError(image2_arg)
    return [image2_arg]


def process_image2(data2, sources_template, aperture_scale, frame_name=""):
    """Forced sep.sum_circle photometry on data2 using sources_template
    positions and per-fiber FWHMs. Returns a copy of sources_template with
    flux_image2, flux_image2_err, flag_image2, relflux columns added."""
    flux2, fluxerr2, flag2, _, _ = forced_aperture_photometry(
        data2, sources_template["x"].values, sources_template["y"].values,
        sources_template["fwhm"].values, aperture_scale=aperture_scale,
    )
    out = sources_template.copy()
    out["flux_image2"] = flux2
    out["flux_image2_err"] = fluxerr2
    out["flag_image2"] = flag2
    out["frame"] = frame_name
    fmax = float(np.nanmax(flux2))
    if np.isfinite(fmax) and fmax > 0:
        out["relflux"] = flux2 / fmax
    else:
        out["relflux"] = np.nan
    return out, fmax


def per_frame_metrics(df, frame_name):
    """Compute the per-frame summary used for repeatability statistics."""
    flux = df["flux_image2"].values.astype(float)
    valid = np.isfinite(flux)
    f = np.where(valid & (flux > 0), flux, 0.0)
    total = float(np.sum(f))
    if total <= 0:
        return None

    sx = df["sky_x"].values
    sy = df["sky_y"].values
    cx = float(np.sum(f * sx) / total)
    cy = float(np.sum(f * sy) / total)

    idx_max = int(np.nanargmax(flux))
    best_id = int(df["ID"].iloc[idx_max])
    f_max = float(flux[idx_max])

    # Top-N concentration metrics (independent of WHICH fibers are
    # brightest -- just measures how peaked the light distribution is).
    sorted_f = np.sort(f)[::-1]
    top1 = float(sorted_f[0] / total)
    top5 = float(np.sum(sorted_f[:5]) / total)
    top10 = float(np.sum(sorted_f[:10]) / total)

    # Number of fibers carrying >=10% of the brightest fiber's flux --
    # a rough proxy for "spot size on the IFU face".
    n_above_10pct = int(np.sum(flux >= 0.1 * f_max))

    return {
        "frame":                  frame_name,
        "centroid_sky_x":         cx,
        "centroid_sky_y":         cy,
        "brightest_fiber":        best_id,
        "brightest_fiber_flux":   f_max,
        "total_flux":             total,
        "top1_fraction":          top1,
        "top5_fraction":          top5,
        "top10_fraction":         top10,
        "n_fibers_above_10pct":   n_above_10pct,
    }


def aggregate_repeatability(per_frame_df):
    """Reduce the per-frame summary table to a few headline numbers.

    Headline metric: centroid_rms = root-mean-square distance of each
    frame's flux-weighted centroid from the mean centroid. Lower is
    better. In the same units as sky_x / sky_y.
    """
    n = len(per_frame_df)
    if n == 0:
        return {}

    cx = per_frame_df["centroid_sky_x"].values
    cy = per_frame_df["centroid_sky_y"].values
    cx_m, cy_m = float(np.mean(cx)), float(np.mean(cy))
    cx_s = float(np.std(cx, ddof=1)) if n > 1 else 0.0
    cy_s = float(np.std(cy, ddof=1)) if n > 1 else 0.0
    rms = float(np.sqrt(np.mean((cx - cx_m) ** 2 + (cy - cy_m) ** 2)))

    fiber_counts = per_frame_df["brightest_fiber"].value_counts()
    mode_id = int(fiber_counts.index[0])
    mode_count = int(fiber_counts.iloc[0])

    tf = per_frame_df["total_flux"].values.astype(float)
    tf_rel_std = float(np.std(tf, ddof=1) / np.mean(tf)) if n > 1 and np.mean(tf) > 0 else 0.0

    top5 = per_frame_df["top5_fraction"].values
    top5_mean = float(np.mean(top5))
    top5_std  = float(np.std(top5, ddof=1)) if n > 1 else 0.0

    return {
        "n_frames":                 n,
        "centroid_x_mean":          cx_m,
        "centroid_y_mean":          cy_m,
        "centroid_x_std":           cx_s,
        "centroid_y_std":           cy_s,
        "centroid_rms":             rms,
        "brightest_fiber_mode":     mode_id,
        "brightest_fiber_consistency": float(mode_count) / n,
        "brightest_fiber_unique":   int(fiber_counts.size),
        "total_flux_relative_std":  tf_rel_std,
        "top5_fraction_mean":       top5_mean,
        "top5_fraction_std":        top5_std,
    }


def print_repeatability(agg, fiber_counts):
    """Pretty-print the headline numbers."""
    print("\n" + "=" * 70)
    print(f"REPEATABILITY SUMMARY  ({agg['n_frames']} frames)")
    print("=" * 70)
    print(f"  Centroid (sky_x, sky_y):  "
          f"({agg['centroid_x_mean']:+.4f}, {agg['centroid_y_mean']:+.4f})  "
          f"± ({agg['centroid_x_std']:.4f}, {agg['centroid_y_std']:.4f})")
    print(f"  Centroid RMS scatter:     {agg['centroid_rms']:.4f}  "
          "(headline number; sky-coord units)")
    print(f"  Brightest fiber:          ID {agg['brightest_fiber_mode']} "
          f"in {int(round(agg['brightest_fiber_consistency'] * agg['n_frames']))}/"
          f"{agg['n_frames']} frames "
          f"(unique winners: {agg['brightest_fiber_unique']})")
    print(f"  Total flux scatter:       "
          f"{100 * agg['total_flux_relative_std']:.2f}% "
          f"(seeing/throughput proxy)")
    print(f"  Top-5 flux concentration: "
          f"{100 * agg['top5_fraction_mean']:.1f}% ± "
          f"{100 * agg['top5_fraction_std']:.1f}% "
          "(spot size proxy)")
    if agg['brightest_fiber_unique'] > 1:
        print("\n  Brightest-fiber distribution:")
        for fid, cnt in fiber_counts.head(5).items():
            print(f"    ID {int(fid):>4d}:  {cnt:>3d} / {agg['n_frames']} frames")
    print("=" * 70 + "\n")


def plot_centroid_trail(per_frame_df, sources_template, agg, out_path):
    """Two panels: full-IFU broken-axis layout + zoomed centroid scatter.

    Left/middle/right (broken axes): all 244 fibers as faint gray dots so
    the centroid trail can be located on the IFU. The flux-weighted
    centroid for each frame is overplotted as a colored dot, with frame
    index as the color.
    Bottom panel: a zoom on the centroid cloud with mean point and 1-sigma
    error ellipse. This is the actual repeatability picture.
    """
    fig = plt.figure(figsize=(15, 11))
    # --- Top: full IFU broken-axis with centroid trail
    xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
    widths = [b - a for a, b in xlims]
    ymin = sources_template["sky_y"].min() - 1
    ymax = sources_template["sky_y"].max() + 1

    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.4],
                          width_ratios=widths, hspace=0.25, wspace=0.05)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]

    cx_all = per_frame_df["centroid_sky_x"].values
    cy_all = per_frame_df["centroid_sky_y"].values
    cmap = plt.get_cmap("viridis")
    n = len(per_frame_df)
    colors = cmap(np.linspace(0, 1, max(n, 2)))

    sc_top = None
    for i, ax in enumerate(axes_top):
        a, b = xlims[i]
        sub = sources_template[(sources_template["sky_x"] >= a) &
                               (sources_template["sky_x"] <= b)]
        ax.set_facecolor("black")
        ax.scatter(sub["sky_x"], sub["sky_y"], s=20,
                   c="0.35", edgecolors="0.6", linewidths=0.3, zorder=1)
        m = (cx_all >= a) & (cx_all <= b)
        if np.any(m):
            sc_top = ax.scatter(cx_all[m], cy_all[m],
                                c=np.arange(n)[m], cmap="viridis",
                                s=70, edgecolors="white", linewidths=0.7,
                                zorder=4)
        # Mean centroid
        ax.scatter([agg["centroid_x_mean"]], [agg["centroid_y_mean"]],
                   marker="*", s=320, c="red", edgecolors="white",
                   linewidths=0.7, zorder=5)
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

    axes_top[0].set_ylabel("Sky Y", fontsize=11)
    axes_top[1].set_xlabel("Sky X", fontsize=11)

    # Break marks
    d = 0.012
    kw = dict(color="k", clip_on=False, lw=1.0)
    for i in (0, 1):
        ax = axes_top[i]
        tk = dict(kw, transform=ax.transAxes)
        ax.plot([1 - d, 1 + d], [-d, +d], **tk)
        ax.plot([1 - d, 1 + d], [1 - d, 1 + d], **tk)
    for i in (1, 2):
        ax = axes_top[i]
        tk = dict(kw, transform=ax.transAxes)
        ax.plot([-d, +d], [-d, +d], **tk)
        ax.plot([-d, +d], [1 - d, 1 + d], **tk)

    if sc_top is not None:
        cb = fig.colorbar(sc_top, ax=axes_top, fraction=0.025, pad=0.02,
                          shrink=0.85)
        cb.set_label("Frame index", fontsize=10)

    # --- Bottom: zoomed centroid scatter
    ax_zoom = fig.add_subplot(gs[1, :])
    ax_zoom.set_facecolor("black")
    # Plot fibers near the cloud as context
    pad = 4 * max(agg["centroid_x_std"], agg["centroid_y_std"], 0.1)
    xz_lo = agg["centroid_x_mean"] - pad
    xz_hi = agg["centroid_x_mean"] + pad
    yz_lo = agg["centroid_y_mean"] - pad
    yz_hi = agg["centroid_y_mean"] + pad
    near = sources_template[
        (sources_template["sky_x"] >= xz_lo) & (sources_template["sky_x"] <= xz_hi) &
        (sources_template["sky_y"] >= yz_lo) & (sources_template["sky_y"] <= yz_hi)
    ]
    ax_zoom.scatter(near["sky_x"], near["sky_y"], s=120,
                    c="0.18", edgecolors="0.55", linewidths=0.4, zorder=1)
    for _, row in near.iterrows():
        ax_zoom.annotate(str(int(row["ID"])),
                         xy=(row["sky_x"], row["sky_y"]),
                         xytext=(0, 0), textcoords="offset points",
                         fontsize=7, ha="center", va="center",
                         color="0.7", zorder=2)

    # Centroid trail with frame indices
    sc = ax_zoom.scatter(cx_all, cy_all, c=np.arange(n), cmap="viridis",
                         s=80, edgecolors="white", linewidths=0.8, zorder=4)
    for i, (x, y) in enumerate(zip(cx_all, cy_all)):
        ax_zoom.annotate(f"{i}", xy=(x, y), xytext=(4, 4),
                         textcoords="offset points",
                         fontsize=7, color="white", zorder=5)

    # Mean + 1-sigma ellipse
    ax_zoom.scatter([agg["centroid_x_mean"]], [agg["centroid_y_mean"]],
                    marker="*", s=420, c="red", edgecolors="white",
                    linewidths=0.8, zorder=6,
                    label=f"mean ({agg['centroid_x_mean']:+.3f}, "
                          f"{agg['centroid_y_mean']:+.3f})")
    if agg["n_frames"] > 1:
        ell = Ellipse(
            (agg["centroid_x_mean"], agg["centroid_y_mean"]),
            2 * agg["centroid_x_std"], 2 * agg["centroid_y_std"],
            fill=False, edgecolor="red", lw=1.0, ls="--", zorder=5,
            label=f"1σ ellipse (RMS={agg['centroid_rms']:.3f})",
        )
        ax_zoom.add_patch(ell)

    ax_zoom.set_xlim(xz_lo, xz_hi)
    ax_zoom.set_ylim(yz_lo, yz_hi)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("Sky X")
    ax_zoom.set_ylabel("Sky Y")
    ax_zoom.grid(True, alpha=0.25, ls="--", lw=0.5, color="gray")
    ax_zoom.legend(fontsize=9, loc="upper right",
                   facecolor="white", framealpha=0.85)

    fig.suptitle(
        f"Star centroid repeatability — N={agg['n_frames']} frames    "
        f"RMS={agg['centroid_rms']:.4f}    "
        f"brightest fiber {agg['brightest_fiber_mode']} in "
        f"{int(round(agg['brightest_fiber_consistency'] * agg['n_frames']))}/"
        f"{agg['n_frames']} frames",
        fontsize=13, y=0.995,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_gif(image_paths, out_path, duration_ms=500):
    """Stitch a list of PNGs into an animated GIF. Uses PIL."""
    if not image_paths:
        return False
    if not HAVE_PIL:
        print("  WARNING: Pillow not installed; skipping GIF. "
              "`pip install Pillow` to enable.")
        return False
    frames = []
    for p in image_paths:
        try:
            frames.append(PILImage.open(p).convert("RGB"))
        except Exception as e:
            print(f"  WARNING: skipping {p} in GIF: {e}")
    if not frames:
        return False
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )
    return True


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
                        help="FITS file OR a directory of FITS files for "
                             "forced photometry. If a directory, all FITS "
                             "files inside are processed against the same "
                             "fiber template from image1 and assembled "
                             "into a GIF + repeatability summary.")
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
    parser.add_argument("--gif-duration", type=int, default=500,
                        help="(folder mode) milliseconds per GIF frame "
                             "(default 500)")
    parser.add_argument("--no-gif", action="store_true",
                        help="(folder mode) skip GIF assembly even if "
                             "Pillow is available")
    parser.add_argument("--fixed-color-limits", action="store_true",
                        help="(folder mode) force vmin=0, vmax=1 for the "
                             "throughput map color scale. Default behavior "
                             "in folder mode is pooled p2-p98 across all "
                             "frames (so the colorbar is the same on every "
                             "plot but doesn't waste range on outliers). "
                             "Use this flag instead if you want strict "
                             "[0, 1] across runs.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image1))[0]
    def out(name):
        return os.path.join(args.outdir, f"{stem}_{name}")

    # ---- Resolve image2 path(s)
    image2_paths = collect_image2_paths(args.image2)
    folder_mode = len(image2_paths) > 1
    print(f"[setup] image1={args.image1}, "
          f"{'folder' if folder_mode else 'single'} mode "
          f"({len(image2_paths)} image2 file{'s' if folder_mode else ''})")

    # ---- Load image1
    print(f"[load] {args.image1}")
    data1 = load_fits(args.image1)
    print(f"  shape image1 = {data1.shape}")

    # ---- Step 1: locate the dot column
    x_peak, y_collapsed = find_dot_column(data1)
    print(f"[step 1] dot column at x = {x_peak}")
    plot_xprofile(y_collapsed, x_peak, out("xprofile.pdf"))

    # ---- Step 2: column profile
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

    # ---- Self-photometry on Image 1 (sanity)
    flux1, fluxerr1, flag1, radii, _ = forced_aperture_photometry(
        data1, sources["x"].values, sources["y"].values,
        sources["fwhm"].values, aperture_scale=args.aperture_scale,
    )
    sources["flux_image1"] = flux1
    sources["flux_image1_err"] = fluxerr1
    sources["flag_image1"] = flag1

    plot_image1_detection(data1, sources, radii, out("image1_detection.pdf"))

    # Save the fiber template -- positions, FWHMs, IDs. Same for every
    # image2 frame in folder mode.
    template_csv = out("fiber_template.csv")
    sources.to_csv(template_csv, index=False)
    print(f"[write] {template_csv}")

    # ---- Step 7-9: forced photometry on each image2 frame
    if folder_mode:
        per_frame_dir = os.path.join(args.outdir, "per_frame")
        os.makedirs(per_frame_dir, exist_ok=True)

    per_frame_summary = []
    png_paths = []   # for the GIF

    cbar_label = (f"Relative flux  (flux / max,  "
                  f"D = {args.aperture_scale:.2f} × FWHM)")

    # ---- Pass 1: photometry + per-frame CSVs + repeatability metrics.
    # Keep DataFrames in memory so pass 2 can render with shared color
    # limits. ~244 rows per frame is trivial for any reasonable run.
    frame_records = []  # list of (fi, frame_name, df, csv_path, pdf_path, png_path)

    for fi, p2 in enumerate(image2_paths):
        frame_name = os.path.splitext(os.path.basename(p2))[0]
        print(f"\n[frame {fi + 1}/{len(image2_paths)}] {p2}")
        data2 = load_fits(p2)
        if data2.shape != data1.shape:
            print(f"  WARNING: {frame_name} shape {data2.shape} differs "
                  f"from image1 {data1.shape}; forced photometry assumes "
                  "shared pixel coordinates.")

        df, fmax = process_image2(data2, sources, args.aperture_scale,
                                  frame_name=frame_name)
        print(f"  max flux = {fmax:.3e}, "
              f"median relflux = {np.nanmedian(df['relflux']):.3f}")

        if folder_mode:
            csv_p = os.path.join(per_frame_dir, f"{frame_name}_fluxes.csv")
            pdf_p = os.path.join(per_frame_dir, f"{frame_name}_throughput_map.pdf")
            png_p = os.path.join(per_frame_dir, f"{frame_name}_throughput_map.png")
        else:
            csv_p = out("fiber_positions.csv")
            pdf_p = out("throughput_map.pdf")
            png_p = None

        df.to_csv(csv_p, index=False)
        frame_records.append((fi, frame_name, df, csv_p, pdf_p, png_p))

        m = per_frame_metrics(df, frame_name)
        if m is not None:
            per_frame_summary.append(m)

    # ---- Determine the shared color scale across ALL frames so the
    # GIF (and per-frame PDFs) use a consistent colorbar.
    if args.fixed_color_limits:
        map_vmin, map_vmax = 0.0, 1.0
        scale_descr = "fixed [0, 1]"
    else:
        all_relflux = np.concatenate([r[2]["relflux"].values
                                      for r in frame_records])
        all_relflux = all_relflux[np.isfinite(all_relflux)]
        if all_relflux.size:
            map_vmin = float(np.nanpercentile(all_relflux, 2))
            map_vmax = float(np.nanpercentile(all_relflux, 98))
            # Guarantee a visible range even when relflux is uniform
            if map_vmax - map_vmin < 1e-3:
                map_vmin = float(np.nanmin(all_relflux))
                map_vmax = float(np.nanmax(all_relflux))
        else:
            map_vmin, map_vmax = 0.0, 1.0
        scale_descr = (f"pooled p2-p98 across {len(frame_records)} frames"
                       if folder_mode else "p2-p98")
    if folder_mode:
        print(f"\n[scale] shared colormap range: "
              f"vmin={map_vmin:.4f}, vmax={map_vmax:.4f} ({scale_descr})")

    # ---- Pass 2: render PDFs and PNGs with the shared scale
    for fi, frame_name, df, csv_p, pdf_p, png_p in frame_records:
        plot_throughput_map(
            df, pdf_p,
            title=f"IFU forced-photometry throughput",
            ratio_col="relflux", cbar_label=cbar_label,
            cmap=args.cmap, vmin=map_vmin, vmax=map_vmax,
            frame_label=(f"frame {fi + 1}/{len(image2_paths)}: {frame_name}"
                         if folder_mode else frame_name),
        )
        if png_p:
            plot_throughput_map(
                df, png_p,
                title=f"IFU forced-photometry throughput",
                ratio_col="relflux", cbar_label=cbar_label,
                cmap=args.cmap, vmin=map_vmin, vmax=map_vmax,
                dpi=120,
                frame_label=f"frame {fi + 1}/{len(image2_paths)}: {frame_name}",
            )
            png_paths.append(png_p)

    # ---- Folder-mode aggregates
    if folder_mode and per_frame_summary:
        sum_df = pd.DataFrame(per_frame_summary)
        sum_csv = out("repeatability_summary.csv")
        sum_df.to_csv(sum_csv, index=False)
        print(f"\n[write] {sum_csv}")

        agg = aggregate_repeatability(sum_df)
        fiber_counts = sum_df["brightest_fiber"].value_counts()
        print_repeatability(agg, fiber_counts)

        agg_txt = out("repeatability_aggregate.txt")
        with open(agg_txt, "w") as f:
            f.write(f"# Repeatability aggregate from {agg['n_frames']} frames\n")
            f.write(f"# Source: {args.image2}\n\n")
            for k, v in agg.items():
                f.write(f"{k}: {v}\n")
            f.write("\n# Brightest-fiber distribution:\n")
            for fid, cnt in fiber_counts.items():
                f.write(f"#   ID {int(fid):>4d}: {cnt} / {agg['n_frames']} frames\n")
        print(f"[write] {agg_txt}")

        trail_pdf = out("centroid_trail.pdf")
        plot_centroid_trail(sum_df, sources, agg, trail_pdf)
        print(f"[write] {trail_pdf}")

        if not args.no_gif:
            gif_path = out("throughput_animation.gif")
            ok = make_gif(png_paths, gif_path,
                          duration_ms=args.gif_duration)
            if ok:
                print(f"[write] {gif_path}  "
                      f"({len(png_paths)} frames @ {args.gif_duration} ms)")

    print("\n[done]")


if __name__ == "__main__":
    main()
