"""
Extract IFU/Slit flux ratio per fiber and produce a broken-axis throughput map.

Key changes vs. the previous version
------------------------------------
1. Trace following now walks OUTWARD from the center column and updates curr_y
   at every step. The old code anchored curr_y at the central peak for every
   x-step, which let the search box slide off curved traces near the chip
   edges.

2. Centroid/FWHM estimator uses a Gaussian-with-constant fit (curve_fit).
   This is robust to asymmetric/truncated windows at the array boundaries,
   which is what was biasing the first and last fibers. Falls back to
   intensity moments only when the window is *not* truncated and the fit
   has failed for some other reason.

3. Polynomial fit is iteratively sigma-clipped, so a single bad centroid
   (cosmic ray, neighbor contamination) cannot drag the trace.

4. FWHM-outlier rejection inside each trace before the polyfit, plus a
   peak-count-vs-CSV-length sanity check at startup.

5. Optional simple background subtraction in extraction (--bg-subtract).

6. Broken-axis plot: equal aspect, robust 2nd-98th percentile color limits,
   explicit break marks on the inner spines, single shared colorbar,
   median ratio in the title.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Slit aperture is slightly larger than a fiber, so the per-trace slit flux
# is inflated by this factor relative to what a fiber-sized aperture would
# collect. The central-normalised ratio is divided by this to compensate.
SLIT_FIBER_AREA_RATIO = 1.152


# ---------------------------------------------------------------------------
# Centroiding
# ---------------------------------------------------------------------------

def _gauss_const(y, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((y - mu) / sigma) ** 2) + C


def fit_centroid_fwhm(profile, peak_index, window=5, fwhm_default=3.0):
    """Sub-pixel centroid + FWHM via Gaussian-plus-constant fit.

    Robust at the chip edges because the model does not assume the window
    is symmetric about the centroid — moments-based centroiding does, and
    that is what biased the first/last fibers in the old script.

    Returns
    -------
    (centroid, fwhm, ok) : tuple
        ok is False if the fit failed and no trustworthy fallback is
        available (i.e. the window was truncated). The caller should drop
        such samples rather than feed them into the polyfit.
    """
    ny = len(profile)
    y_lo = max(0, peak_index - window)
    y_hi = min(ny, peak_index + window + 1)
    y = np.arange(y_lo, y_hi)
    f = profile[y_lo:y_hi].astype(float)

    if len(y) < 4:
        return float(peak_index), fwhm_default, False

    p0 = [
        max(np.nanmax(f) - np.nanmin(f), 1.0),  # amplitude
        float(peak_index),                       # centroid
        1.5,                                     # sigma
        float(np.nanmin(f)),                     # constant background
    ]
    bounds = (
        [0.0, y_lo - 0.5, 0.3,        -np.inf],
        [np.inf, y_hi - 0.5, float(window), np.inf],
    )

    try:
        popt, _ = curve_fit(_gauss_const, y, f, p0=p0, bounds=bounds, maxfev=400)
        _, mu, sigma, _ = popt
        if not (y_lo - 0.5 <= mu <= y_hi - 0.5):
            raise ValueError("centroid escaped window")
        return float(mu), float(2.355 * abs(sigma)), True
    except Exception:
        truncated = (peak_index - window < 0) or (peak_index + window >= ny)
        if truncated:
            # Don't trust moments on a one-sided window — caller should skip.
            return float(peak_index), fwhm_default, False
        weights = np.maximum(f - np.nanmin(f), 0)
        if np.sum(weights) <= 0:
            return float(peak_index), fwhm_default, False
        c = np.sum(y * weights) / np.sum(weights)
        var = np.sum(((y - c) ** 2) * weights) / np.sum(weights)
        return float(c), float(2.355 * np.sqrt(max(0.1, var))), True


# ---------------------------------------------------------------------------
# Trace following + robust polyfit
# ---------------------------------------------------------------------------

def follow_trace(data, p_start, x_steps, search=6, centroid_window=5):
    """Walk outward from the central column, updating curr_y each step.

    Returns arrays of (x, y_centroid, fwhm), sorted by x. Steps where the
    centroid fit failed are dropped, so the polyfit only sees clean data.
    """
    ny = data.shape[0]
    mid = len(x_steps) // 2
    xs, ys, fws = [], [], []

    def _step(x, curr_y):
        col = np.median(data[:, max(0, x - 2):x + 3], axis=1)
        y_min = int(max(0, curr_y - search))
        y_max = int(min(ny, curr_y + search))
        if y_max <= y_min + 1:
            return None
        local_peak = int(np.argmax(col[y_min:y_max])) + y_min
        cy, fwhm, ok = fit_centroid_fwhm(col, local_peak, window=centroid_window)
        if not ok:
            return None
        return cy, fwhm

    # Walk right (and across the center)
    curr_y = float(p_start)
    for x in x_steps[mid:]:
        out = _step(x, curr_y)
        if out is None:
            continue
        cy, fwhm = out
        xs.append(x); ys.append(cy); fws.append(fwhm)
        curr_y = cy

    # Walk left
    curr_y = float(p_start)
    for x in reversed(x_steps[:mid]):
        out = _step(x, curr_y)
        if out is None:
            continue
        cy, fwhm = out
        xs.append(x); ys.append(cy); fws.append(fwhm)
        curr_y = cy

    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order], np.array(fws)[order]


def robust_polyfit(x, y, deg=3, sigma=3.0, n_iter=3):
    """Iteratively sigma-clipped polynomial fit. Returns (coeffs, mask)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.ones_like(x, dtype=bool)
    coeffs = np.polyfit(x, y, deg)
    for _ in range(n_iter):
        if mask.sum() < deg + 2:
            break
        coeffs = np.polyfit(x[mask], y[mask], deg)
        resid = y - np.polyval(coeffs, x)
        rms = np.std(resid[mask])
        if rms <= 0:
            break
        new_mask = np.abs(resid) < sigma * rms
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return coeffs, mask


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_flux(data, coeffs, fwhm_val, k=0.3,
                 bg_subtract=False, bg_gap=2.0, bg_width=3.0):
    """Top-hat sum within ±k*FWHM of the trace.

    If bg_subtract is True, subtract a per-column median estimated from
    bands sitting `bg_gap` pixels outside the aperture, with thickness
    `bg_width`. Useful when fibers are well-separated; harmful in densely
    packed regions where the background bands hit neighbors. Off by default.
    """
    ny, nx = data.shape
    x_pixels = np.arange(nx)
    y_centers = np.polyval(coeffs, x_pixels)
    h_width = k * fwhm_val
    spectrum = np.zeros(nx)

    for x in x_pixels:
        y0 = y_centers[x]
        y_start = int(np.floor(y0 - h_width))
        y_end = int(np.ceil(y0 + h_width))
        ya, yb = max(0, y_start), min(ny, y_end + 1)
        ap = data[ya:yb, x]
        flux = float(np.sum(ap))
        n_ap = yb - ya

        if bg_subtract:
            bg_lo_a = max(0, int(np.floor(y0 - h_width - bg_gap - bg_width)))
            bg_lo_b = max(0, int(np.floor(y0 - h_width - bg_gap)))
            bg_hi_a = min(ny, int(np.ceil(y0 + h_width + bg_gap)))
            bg_hi_b = min(ny, int(np.ceil(y0 + h_width + bg_gap + bg_width)))
            bg_pix = np.concatenate([data[bg_lo_a:bg_lo_b, x],
                                     data[bg_hi_a:bg_hi_b, x]])
            if bg_pix.size >= 3:
                flux -= float(np.median(bg_pix)) * n_ap

        spectrum[x] = flux
    return spectrum


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_diagnostic_pdf(data, trace_data, folder_name, title):
    suffix = "ifu_apertures" if "ifu" in title.lower() else "slit_apertures"
    output_name = f"{folder_name}_{suffix}_diagnostic.pdf"
    ny, nx = data.shape

    fig, ax = plt.subplots(figsize=(25, 15))
    pos = data[data > 0]
    if pos.size:
        vmin, vmax = np.nanpercentile(pos, [5, 99.5])
    else:
        vmin, vmax = 1e-3, 1.0

    ax.imshow(data, origin='lower', cmap='magma', aspect='auto',
              norm=LogNorm(vmin=max(vmin, 1e-3), vmax=vmax),
              interpolation='nearest')

    x_plot = np.linspace(0, nx - 1, 1000)
    n = len(trace_data)
    for i, tr in enumerate(trace_data):
        y_mid = np.polyval(tr['coeffs'], x_plot)
        h_width = tr['fwhm'] * 0.6
        ax.plot(x_plot, y_mid, color='cyan', lw=0.2, ls=':', alpha=0.8)
        ax.plot(x_plot, y_mid - h_width, color='white', lw=0.15, alpha=0.5)
        ax.plot(x_plot, y_mid + h_width, color='white', lw=0.15, alpha=0.5)
        # always label first and last fibers (they're the ones that fail)
        if i == 0 or i == n - 1 or i % 10 == 0:
            ax.text(nx * 0.005, y_mid[0], f"#{i}", color='yellow', fontsize=4,
                    va='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.3, pad=0.1))

    ax.set_title(f"{title} | Dynamic FWHM Apertures", fontsize=16)
    plt.savefig(output_name, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")


def plot_throughput_map(df, folder_name, ratio_col='ratio',
                        cbar_label='Flux Ratio (IFU / Slit)',
                        title_prefix='Flux Throughput Ratio',
                        vmin=None, vmax=None, cmap='RdYlBu_r'):
    """Three-panel broken-axis fiber map with break marks and equal aspect.

    vmin/vmax default to the 2nd-98th percentile of the data (robust auto-
    scaling); pass explicit values to fix the colorbar range.
    """
    xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
    widths = [b - a for a, b in xlims]
    ymin = df['sky_y'].min() - 1
    ymax = df['sky_y'].max() + 1

    auto_lo, auto_hi = np.nanpercentile(df[ratio_col], [2, 98])
    if vmin is None:
        vmin = auto_lo
    if vmax is None:
        vmax = auto_hi

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 8), sharey=True,
        gridspec_kw={'width_ratios': widths, 'wspace': 0.05},
    )
    norm = plt.Normalize(vmin, vmax)

    sc = None
    for i, ax in enumerate(axes):
        a, b = xlims[i]
        sub = df[(df['sky_x'] >= a) & (df['sky_x'] <= b)]
        sc = ax.scatter(sub['sky_x'], sub['sky_y'], c=sub[ratio_col],
                        s=260, cmap=cmap, edgecolors='k', linewidths=0.5,
                        norm=norm, zorder=2)
        for _, row in sub.iterrows():
            ax.annotate(str(int(row['ID'])),
                        xy=(row['sky_x'], row['sky_y']),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=5.5, ha='center', va='center',
                        fontweight='bold')
            ax.annotate(f"t{int(row['trace'])}",
                        xy=(row['sky_x'], row['sky_y']),
                        xytext=(0, -3.5), textcoords='offset points',
                        fontsize=4.5, ha='center', va='center',
                        color='black', alpha=0.6)

        ax.set_xlim(a, b)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, ls='--', lw=0.5)
        ax.tick_params(axis='both', labelsize=9)

        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False)
        if i < 2:
            ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Sky Y', fontsize=11)
    axes[1].set_xlabel('Sky X', fontsize=11)

    # Break marks on the inner spines
    d = 0.012
    kw = dict(color='k', clip_on=False, lw=1.0)
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

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical',
                        fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=11)

    med = np.nanmedian(df[ratio_col])
    fig.suptitle(
        f"{title_prefix}: {folder_name}   "
        f"(N={len(df)}, median={med:.3f}, "
        f"p2-98=[{vmin:.3f}, {vmax:.3f}])",
        fontsize=13, y=0.96,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", help="IFU reference flat (fits)")
    parser.add_argument("image2", help="Comparison slit flat (fits)")
    parser.add_argument("--csv", default="final_data.csv")
    parser.add_argument("--bg-subtract", action="store_true",
                        help="Subtract a median sky-band background per column")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't pop up the matplotlib window")
    args = parser.parse_args()

    path_parts = args.image1.split(os.sep)
    folder_name = path_parts[-2] if len(path_parts) > 1 else "output"

    data1 = fits.getdata(args.image1).astype(float)
    data2 = fits.getdata(args.image2).astype(float)
    fiber_df = pd.read_csv(args.csv)
    ny, nx = data1.shape

    # 1. Detect peaks at the central column
    center_prof = np.median(data1[:, nx // 2 - 10:nx // 2 + 10], axis=1)
    peaks, _ = find_peaks(
        gaussian_filter1d(center_prof, 1.2),
        distance=4,
        prominence=np.nanpercentile(center_prof, 10),
    )
    peaks = np.sort(peaks)
    n_peaks, n_fibers = len(peaks), len(fiber_df)
    print(f"Detected {n_peaks} traces; CSV has {n_fibers} fibers.")
    if n_peaks != n_fibers:
        print(f"  WARNING: peak count differs from CSV by {n_peaks - n_fibers}. "
              "Check the diagnostic PDFs before trusting the ratio map.")

    # 2. Follow each trace and fit a robust polynomial
    x_steps = np.linspace(50, nx - 50, 25, dtype=int)
    trace_info = []
    for p_start in peaks:
        xs, ys, fws = follow_trace(data1, p_start, x_steps)
        if len(xs) < 5:
            print(f"  skipped trace at y={p_start}: only {len(xs)} good centroids")
            continue

        # FWHM-outlier rejection (likely contamination from a neighbor)
        med_f = float(np.median(fws))
        f_mask = (fws > 0.5 * med_f) & (fws < 2.0 * med_f)
        if f_mask.sum() < 5:
            f_mask = np.ones_like(fws, dtype=bool)

        coeffs, _ = robust_polyfit(xs[f_mask], ys[f_mask], deg=3, sigma=3.0)
        trace_info.append({
            'coeffs': coeffs,
            'fwhm': float(np.median(fws[f_mask])),
            'median_y': float(np.median(ys[f_mask])),
        })

    trace_info = sorted(trace_info, key=lambda t: t['median_y'])
    print(f"Kept {len(trace_info)} usable traces.")

    # 3. Diagnostics
    save_diagnostic_pdf(data1, trace_info, folder_name, "IFU Reference Traces")
    save_diagnostic_pdf(data2, trace_info, folder_name, "Slit Flat Blind Extraction")

    # 4. Extract and ratio
    fiber_df_sorted = fiber_df.sort_values('slit_x').reset_index(drop=True)
    records = []
    for i, tr in enumerate(trace_info):
        if i >= len(fiber_df_sorted):
            break
        s1 = extract_flux(data1, tr['coeffs'], tr['fwhm'], bg_subtract=args.bg_subtract)
        s2 = extract_flux(data2, tr['coeffs'], tr['fwhm'], bg_subtract=args.bg_subtract)
        records.append({
            'sky_x':     fiber_df_sorted.iloc[i]['sky_x'],
            'sky_y':     fiber_df_sorted.iloc[i]['sky_y'],
            'ID':        int(fiber_df_sorted.iloc[i]['ID']),
            'trace':     i + 1,
            'ifu_flux':  float(np.sum(s1)),
            'slit_flux': float(np.sum(s2)),
        })
    df = pd.DataFrame(records)

    # Per-fiber ratio (IFU / Slit at the same aperture)
    df['ratio'] = np.where(df['slit_flux'] > 0,
                           df['ifu_flux'] / df['slit_flux'], np.nan)

    # Central-trace reference: mean slit flux over the middle ~10 traces.
    # Used as a single normalisation scalar so the slit-lamp profile along
    # the slit doesn't get divided into the result.
    n = len(df)
    half = 5
    mid = n // 2
    lo, hi = max(0, mid - half), min(n, mid + half)
    central = df.iloc[lo:hi]
    ref_slit = float(np.nanmean(central['slit_flux']))
    print(f"Central reference: mean slit flux over traces "
          f"{int(central['trace'].min())}-{int(central['trace'].max())} "
          f"({len(central)} traces) = {ref_slit:.3e}")

    df['ratio_central'] = (df['ifu_flux'] / ref_slit
                           if ref_slit > 0 else np.nan)

    # Slit aperture is slightly larger than a fiber — correct so the ratio
    # represents fiber-to-fiber throughput rather than fiber-to-slit collection.
    df['ratio_central_corr'] = df['ratio_central'] / SLIT_FIBER_AREA_RATIO

    out_csv = f"{folder_name}_throughput.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote per-fiber fluxes and ratios to: {out_csv}")

    # 5. Final maps — one per ratio definition
    fig1 = plot_throughput_map(
        df, folder_name,
        ratio_col='ratio',
        cbar_label='Flux Ratio (IFU / Slit, per aperture)',
        title_prefix='Flux Throughput Ratio',
    )
    pdf1 = f"{folder_name}_flux_throughput_map.pdf"
    fig1.savefig(pdf1, bbox_inches='tight')
    print(f"Saved: {pdf1}")

    fig2 = plot_throughput_map(
        df, folder_name,
        ratio_col='ratio_central_corr',
        cbar_label=(f'(IFU / mean slit, central {len(central)} traces)'
                    f' / {SLIT_FIBER_AREA_RATIO}'),
        title_prefix='IFU Throughput vs Central-Slit Mean (corrected)',
        vmin=0.0, vmax=0.6, cmap='RdYlGn',
    )
    pdf2 = f"{folder_name}_flux_throughput_map_centralnorm.pdf"
    fig2.savefig(pdf2, bbox_inches='tight')
    print(f"Saved: {pdf2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
