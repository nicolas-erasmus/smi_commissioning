"""
Build a low-resolution RGB image of an IFU galaxy frame, one "pixel" per fiber.

Pipeline (mirrors the throughput-map script you sent me):
  1. Trace-fit on a flat (image1) — peak detect at the central column,
     follow each trace outward, robust polyfit, FWHM per trace.
  2. Re-use those same trace coefficients + FWHMs as fixed apertures and
     extract a 1-D spectrum from the science frame (image2 = the galaxy).
  3. Split each fiber spectrum into THREE equal wavelength blocks along
     the dispersion axis, sum each block -> (B, G, R) flux triplet.
     (Convention: dispersion runs blue->red along +x. Use --reverse-wavelength
      if your detector is flipped.)
  4. Normalize the three channels across all fibers (linear or asinh) and
     plot each fiber as a filled circle whose facecolor IS the RGB triplet.
     Result: a coarse color picture of the galaxy on the IFU footprint.

Reuses follow_trace / robust_polyfit / extract_flux from the original logic.
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


# ---------------------------------------------------------------------------
# Centroiding (unchanged from the throughput script)
# ---------------------------------------------------------------------------

def _gauss_const(y, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((y - mu) / sigma) ** 2) + C


def fit_centroid_fwhm(profile, peak_index, window=5, fwhm_default=3.0):
    ny = len(profile)
    y_lo = max(0, peak_index - window)
    y_hi = min(ny, peak_index + window + 1)
    y = np.arange(y_lo, y_hi)
    f = profile[y_lo:y_hi].astype(float)

    if len(y) < 4:
        return float(peak_index), fwhm_default, False

    p0 = [
        max(np.nanmax(f) - np.nanmin(f), 1.0),
        float(peak_index),
        1.5,
        float(np.nanmin(f)),
    ]
    bounds = (
        [0.0, y_lo - 0.5, 0.3, -np.inf],
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
            return float(peak_index), fwhm_default, False
        weights = np.maximum(f - np.nanmin(f), 0)
        if np.sum(weights) <= 0:
            return float(peak_index), fwhm_default, False
        c = np.sum(y * weights) / np.sum(weights)
        var = np.sum(((y - c) ** 2) * weights) / np.sum(weights)
        return float(c), float(2.355 * np.sqrt(max(0.1, var))), True


# ---------------------------------------------------------------------------
# Trace following + robust polyfit (unchanged)
# ---------------------------------------------------------------------------

def follow_trace(data, p_start, x_steps, search=6, centroid_window=5):
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

    curr_y = float(p_start)
    for x in x_steps[mid:]:
        out = _step(x, curr_y)
        if out is None:
            continue
        cy, fwhm = out
        xs.append(x); ys.append(cy); fws.append(fwhm)
        curr_y = cy

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
# Extraction (unchanged)
# ---------------------------------------------------------------------------

def extract_flux(data, coeffs, fwhm_val, k=0.6,
                 bg_subtract=False, bg_gap=2.0, bg_width=3.0):
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
# Spectrum -> RGB
# ---------------------------------------------------------------------------

def spectrum_to_bgr_block_sums(spectrum, reverse_wavelength=False):
    """Split a 1-D spectrum into 3 equal blocks along x and sum each block.

    Returns (b_sum, g_sum, r_sum), assuming dispersion runs blue -> red
    along increasing x. If your detector is flipped, pass reverse_wavelength=True.
    """
    n = len(spectrum)
    edge1 = n // 3
    edge2 = 2 * n // 3
    block0 = float(np.sum(spectrum[:edge1]))
    block1 = float(np.sum(spectrum[edge1:edge2]))
    block2 = float(np.sum(spectrum[edge2:]))
    if reverse_wavelength:
        # detector reads red -> blue, swap so blue=lowest x in our convention
        return block2, block1, block0
    return block0, block1, block2


def normalize_rgb(rgb_array, stretch='linear', percentile=99.0,
                  Q=8.0, scale=(1.0, 1.0, 1.0), per_channel=False):
    """Map raw (B,G,R)-summed fluxes to [0,1] color tuples.

    Parameters
    ----------
    rgb_array : (N,3) array of (R,G,B) raw fluxes (note column order is RGB,
                we hand-build it that way upstream so matplotlib gets RGB).
    stretch   : 'linear' | 'asinh' | 'log'
    percentile: clip max at this percentile of the brightest channel
                (per_channel=False) or each channel independently (True).
    Q         : softening for asinh stretch — larger Q = more compression.
    scale     : per-channel multiplicative gain applied BEFORE stretching,
                to balance white point if your bands are unequal.
    per_channel : if True, normalize each channel by its own percentile.
                  Punchier color but distorts true relative brightness.
    """
    arr = np.asarray(rgb_array, dtype=float).copy()
    arr = np.clip(arr, 0.0, None)  # negative pixels look ugly; drop them
    arr *= np.asarray(scale)[None, :]

    if per_channel:
        denom = np.array([np.nanpercentile(arr[:, k], percentile) or 1.0
                          for k in range(3)])
    else:
        global_hi = np.nanpercentile(arr, percentile)
        denom = np.array([global_hi, global_hi, global_hi])
    denom = np.where(denom > 0, denom, 1.0)
    x = arr / denom[None, :]

    if stretch == 'linear':
        out = x
    elif stretch == 'log':
        out = np.log1p(np.maximum(x, 0) * 9.0) / np.log(10.0)
    elif stretch == 'asinh':
        # Lupton-style: stretch by intensity, preserve hue
        I = np.mean(x, axis=1)  # average across channels
        I_safe = np.where(I > 0, I, 1.0)
        f = np.arcsinh(Q * I) / (Q * I_safe + 1e-12)
        out = x * f[:, None]
    else:
        raise ValueError(f"unknown stretch: {stretch}")

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_diagnostic_pdf(data, trace_data, folder_name, title):
    suffix = title.lower().replace(' ', '_')
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
        if i == 0 or i == n - 1 or i % 10 == 0:
            ax.text(nx * 0.005, y_mid[0], f"#{i}", color='yellow', fontsize=4,
                    va='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.3, pad=0.1))

    ax.set_title(f"{title} | Apertures used for RGB extraction", fontsize=16)
    plt.savefig(output_name, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")


def plot_rgb_galaxy(df, rgb_norm, folder_name, single_panel=False,
                    marker_size=260, label_fibers=False,
                    bg_color='black', title_suffix=''):
    """Plot fibers at sky_x/sky_y, facecolor = the per-fiber RGB triplet.

    df       : dataframe with sky_x, sky_y, ID, trace columns.
    rgb_norm : (N, 3) array of normalized R,G,B values in [0,1], aligned to df.
    """
    n_total = len(df)

    if single_panel:
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        axes = [ax]
        # auto-zoom to where the fibers actually are
        pad = 1.0
        ax.set_xlim(df['sky_x'].min() - pad, df['sky_x'].max() + pad)
        ax.set_ylim(df['sky_y'].min() - pad, df['sky_y'].max() + pad)
        ax.set_aspect('equal')
        ax.set_facecolor(bg_color)
        ax.grid(True, alpha=0.2, ls='--', lw=0.5, color='gray')
        ax.set_xlabel('Sky X', fontsize=11)
        ax.set_ylabel('Sky Y', fontsize=11)

        ax.scatter(df['sky_x'], df['sky_y'], c=rgb_norm,
                   s=marker_size, edgecolors='none', zorder=2)

        if label_fibers:
            for i, row in df.iterrows():
                ax.annotate(str(int(row['ID'])),
                            xy=(row['sky_x'], row['sky_y']),
                            xytext=(0, 0), textcoords='offset points',
                            fontsize=4.5, ha='center', va='center',
                            color='white', alpha=0.7)
    else:
        # Broken-axis 3-panel layout, matching the original throughput map
        xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
        widths = [b - a for a, b in xlims]
        ymin = df['sky_y'].min() - 1
        ymax = df['sky_y'].max() + 1

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 8), sharey=True,
            gridspec_kw={'width_ratios': widths, 'wspace': 0.05},
        )

        for i, ax in enumerate(axes):
            a, b = xlims[i]
            mask = (df['sky_x'] >= a) & (df['sky_x'] <= b)
            sub = df[mask]
            sub_rgb = rgb_norm[mask.values]

            ax.set_facecolor(bg_color)
            ax.scatter(sub['sky_x'], sub['sky_y'], c=sub_rgb,
                       s=marker_size, edgecolors='none', zorder=2)

            if label_fibers:
                for (_, row), col in zip(sub.iterrows(), sub_rgb):
                    # pick contrasting label color
                    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                    txtc = 'black' if lum > 0.55 else 'white'
                    ax.annotate(str(int(row['ID'])),
                                xy=(row['sky_x'], row['sky_y']),
                                xytext=(0, 3), textcoords='offset points',
                                fontsize=5.5, ha='center', va='center',
                                fontweight='bold', color=txtc)
                    ax.annotate(f"t{int(row['trace'])}",
                                xy=(row['sky_x'], row['sky_y']),
                                xytext=(0, -3.5), textcoords='offset points',
                                fontsize=4.5, ha='center', va='center',
                                color=txtc, alpha=0.7)

            ax.set_xlim(a, b)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, ls='--', lw=0.5, color='gray')
            ax.tick_params(axis='both', labelsize=9)
            if i > 0:
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', which='both', left=False)
            if i < 2:
                ax.spines['right'].set_visible(False)

        axes[0].set_ylabel('Sky Y', fontsize=11)
        axes[1].set_xlabel('Sky X', fontsize=11)

        # break marks
        d = 0.012
        kw = dict(color='k', clip_on=False, lw=1.0)
        for i in (0, 1):
            ax = axes[i]
            tk = dict(kw, transform=ax.transAxes)
            ax.plot([1 - d, 1 + d], [-d, +d], **tk)
            ax.plot([1 - d, 1 + d], [1 - d, 1 + d], **tk)
        for i in (1, 2):
            ax = axes[i]
            tk = dict(kw, transform=ax.transAxes)
            ax.plot([-d, +d], [-d, +d], **tk)
            ax.plot([-d, +d], [1 - d, 1 + d], **tk)

    fig.suptitle(
        f"IFU RGB reconstruction: {folder_name}   "
        f"(N={n_total} fibers){title_suffix}",
        fontsize=13, y=0.96,
    )
    return fig


def plot_grayscale_map(df, folder_name, flux_col='total_flux',
                       cmap='gray', single_panel=False,
                       marker_size=260, label_fibers=False,
                       stretch='linear', Q=8.0, percentile=(2, 98),
                       title_suffix=''):
    """Throughput-map-style fiber plot, but colored by total flux.

    This is the diagnostic you want when the RGB picture looks blank — if
    the galaxy doesn't show up here either, the problem is upstream of the
    color mapping (bad traces, wrong file, bg dominating signal).
    """
    f = df[flux_col].to_numpy(dtype=float)

    # Apply the same stretches we use for RGB, so the visual brightness
    # behavior matches between modes.
    if stretch == 'linear':
        f_disp = f
    elif stretch == 'log':
        f_safe = np.clip(f - np.nanmin(f) + 1.0, 1.0, None)
        f_disp = np.log10(f_safe)
    elif stretch == 'asinh':
        scale = np.nanpercentile(np.clip(f, 0, None), 99) or 1.0
        f_disp = np.arcsinh(Q * np.clip(f, 0, None) / scale) / Q
    else:
        raise ValueError(f"unknown stretch: {stretch}")

    p_lo, p_hi = np.nanpercentile(f_disp, percentile)
    norm = plt.Normalize(p_lo, p_hi)

    if single_panel:
        fig, ax = plt.subplots(figsize=(9.5, 8.5))
        pad = 1.0
        ax.set_xlim(df['sky_x'].min() - pad, df['sky_x'].max() + pad)
        ax.set_ylim(df['sky_y'].min() - pad, df['sky_y'].max() + pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, ls='--', lw=0.5)
        ax.set_xlabel('Sky X', fontsize=11)
        ax.set_ylabel('Sky Y', fontsize=11)

        sc = ax.scatter(df['sky_x'], df['sky_y'], c=f_disp, cmap=cmap,
                        s=marker_size, edgecolors='k', linewidths=0.4,
                        norm=norm, zorder=2)
        if label_fibers:
            for _, row in df.iterrows():
                ax.annotate(str(int(row['ID'])),
                            xy=(row['sky_x'], row['sky_y']),
                            xytext=(0, 3), textcoords='offset points',
                            fontsize=5.5, ha='center', va='center',
                            fontweight='bold', color='red')
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    else:
        xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
        widths = [b - a for a, b in xlims]
        ymin = df['sky_y'].min() - 1
        ymax = df['sky_y'].max() + 1

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 8), sharey=True,
            gridspec_kw={'width_ratios': widths, 'wspace': 0.05},
        )

        sc = None
        for i, ax in enumerate(axes):
            a, b = xlims[i]
            mask = (df['sky_x'] >= a) & (df['sky_x'] <= b)
            sub = df[mask]
            sub_f = f_disp[mask.values]
            sc = ax.scatter(sub['sky_x'], sub['sky_y'], c=sub_f, cmap=cmap,
                            s=marker_size, edgecolors='k', linewidths=0.4,
                            norm=norm, zorder=2)
            if label_fibers:
                for _, row in sub.iterrows():
                    ax.annotate(str(int(row['ID'])),
                                xy=(row['sky_x'], row['sky_y']),
                                xytext=(0, 3), textcoords='offset points',
                                fontsize=5.5, ha='center', va='center',
                                fontweight='bold', color='red')
                    ax.annotate(f"t{int(row['trace'])}",
                                xy=(row['sky_x'], row['sky_y']),
                                xytext=(0, -3.5), textcoords='offset points',
                                fontsize=4.5, ha='center', va='center',
                                color='red', alpha=0.6)
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

        d = 0.012
        kw = dict(color='k', clip_on=False, lw=1.0)
        for i in (0, 1):
            tk = dict(kw, transform=axes[i].transAxes)
            axes[i].plot([1 - d, 1 + d], [-d, +d], **tk)
            axes[i].plot([1 - d, 1 + d], [1 - d, 1 + d], **tk)
        for i in (1, 2):
            tk = dict(kw, transform=axes[i].transAxes)
            axes[i].plot([-d, +d], [-d, +d], **tk)
            axes[i].plot([-d, +d], [1 - d, 1 + d], **tk)

        cbar = fig.colorbar(sc, ax=axes, orientation='vertical',
                            fraction=0.025, pad=0.02, shrink=0.85)

    cbar_lab = f'Total flux ({stretch})' if stretch != 'linear' else 'Total flux'
    cbar.set_label(cbar_lab, fontsize=11)

    med = np.nanmedian(f)
    fig.suptitle(
        f"IFU total-flux map: {folder_name}   "
        f"(N={len(df)}, median flux={med:.3e}){title_suffix}",
        fontsize=13, y=0.96,
    )
    return fig


def plot_rgb_legend_panel(folder_name, title_suffix=''):
    """Tiny standalone figure showing which third of the spectrum became which channel."""
    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for k, (col, lab) in enumerate([
            ((0.15, 0.35, 1.0), 'Blue\n(1st 1/3 of spectrum)'),
            ((0.2, 1.0, 0.3),  'Green\n(middle 1/3)'),
            ((1.0, 0.3, 0.25), 'Red\n(last 1/3)')]):
        ax.add_patch(plt.Rectangle((k + 0.05, 0.2), 0.9, 0.6,
                                   facecolor=col, edgecolor='k', lw=0.5))
        ax.text(k + 0.5, -0.1, lab, ha='center', va='top', fontsize=8)
    ax.set_title(f"RGB channel mapping{title_suffix}", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct an RGB galaxy image from an IFU science frame.")
    parser.add_argument("flat",    help="IFU flat (used for trace fitting)")
    parser.add_argument("science", help="IFU science frame of the galaxy")
    parser.add_argument("--csv", default="final_data.csv",
                        help="Fiber position CSV (must have sky_x, sky_y, ID, slit_x)")
    parser.add_argument("--bg-subtract", action="store_true",
                        help="Per-column background subtraction in extract_flux")
    parser.add_argument("--reverse-wavelength", action="store_true",
                        help="Use if your dispersion runs red->blue along +x")
    parser.add_argument("--stretch", choices=['linear', 'asinh', 'log'],
                        default='asinh', help="Brightness stretch (default: asinh)")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="Clip channels at this percentile (default: 99)")
    parser.add_argument("--Q", type=float, default=8.0,
                        help="Softening for asinh stretch (default: 8)")
    parser.add_argument("--scale-r", type=float, default=1.0)
    parser.add_argument("--scale-g", type=float, default=1.0)
    parser.add_argument("--scale-b", type=float, default=1.0)
    parser.add_argument("--per-channel-norm", action="store_true",
                        help="Normalize each channel independently (more saturated)")
    parser.add_argument("--single-panel", action="store_true",
                        help="One panel zoomed to galaxy extent (vs 3-panel broken axis)")
    parser.add_argument("--label-fibers", action="store_true",
                        help="Overplot fiber IDs / trace numbers")
    parser.add_argument("--marker-size", type=float, default=260.0)
    parser.add_argument("--bg-color", default='black',
                        help="Background of the RGB plot (default: black)")
    parser.add_argument("--xrange", nargs=2, type=int, metavar=('LO', 'HI'),
                        default=None,
                        help="X-pixel range to sum over, inclusive LO, exclusive HI. "
                             "Use this to isolate an emission line or any sub-band. "
                             "Affects both grayscale (total_flux = sum over range) "
                             "and RGB (the range is what gets split into 3 blocks). "
                             "Default: full spectrum.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-spectra", action="store_true",
                        help="Also dump every fiber's 1-D spectrum to a .npy file")
    parser.add_argument("--grayscale", action="store_true",
                        help="Make a total-flux grayscale map instead of RGB. "
                             "Useful as a sanity check when the RGB image looks blank.")
    parser.add_argument("--grayscale-cmap", default='gray',
                        help="Matplotlib cmap for grayscale mode "
                             "(default: 'gray'; try 'inferno' or 'magma' for astro look)")
    args = parser.parse_args()

    path_parts = args.flat.split(os.sep)
    folder_name = path_parts[-2] if len(path_parts) > 1 else "rgb_output"

    flat_data = fits.getdata(args.flat).astype(float)
    sci_data  = fits.getdata(args.science).astype(float)
    if flat_data.shape != sci_data.shape:
        raise SystemExit(f"Flat shape {flat_data.shape} != science shape {sci_data.shape}; "
                         "the trace solution from the flat won't apply to the science frame.")

    fiber_df = pd.read_csv(args.csv)
    ny, nx = flat_data.shape

    # ---- 1. Detect peaks at the central column of the FLAT
    center_prof = np.median(flat_data[:, nx // 2 - 10:nx // 2 + 10], axis=1)
    peaks, _ = find_peaks(
        gaussian_filter1d(center_prof, 1.2),
        distance=4,
        prominence=np.nanpercentile(center_prof, 10),
    )
    peaks = np.sort(peaks)
    n_peaks, n_fibers = len(peaks), len(fiber_df)
    print(f"Detected {n_peaks} traces on the flat; CSV has {n_fibers} fibers.")
    if n_peaks != n_fibers:
        print(f"  WARNING: peak count differs from CSV by {n_peaks - n_fibers}. "
              "Inspect the diagnostic PDFs.")

    # ---- 2. Trace-fit each peak on the FLAT
    x_steps = np.linspace(50, nx - 50, 25, dtype=int)
    trace_info = []
    for p_start in peaks:
        xs, ys, fws = follow_trace(flat_data, p_start, x_steps)
        if len(xs) < 5:
            print(f"  skipped trace at y={p_start}: only {len(xs)} good centroids")
            continue
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

    # Diagnostic plots: same apertures laid over both frames
    save_diagnostic_pdf(flat_data, trace_info, folder_name, "Flat traces")
    save_diagnostic_pdf(sci_data,  trace_info, folder_name, "Science traces")

    # Validate optional x-range
    xrange_lo, xrange_hi = None, None
    if args.xrange is not None:
        xrange_lo, xrange_hi = args.xrange
        if xrange_lo < 0:
            xrange_lo = 0
        if xrange_hi > nx:
            xrange_hi = nx
        if xrange_hi <= xrange_lo:
            raise SystemExit(f"--xrange: LO ({xrange_lo}) must be < HI ({xrange_hi}) "
                             f"and within [0, {nx}].")
        n_used = xrange_hi - xrange_lo
        print(f"Summing/splitting over x = [{xrange_lo}, {xrange_hi})  "
              f"({n_used} pixels of {nx})")

    # ---- 3. Extract spectrum from SCIENCE per trace, split into 3 blocks
    fiber_df_sorted = fiber_df.sort_values('slit_x').reset_index(drop=True)
    records = []
    spectra = []  # optional, kept aligned to records
    for i, tr in enumerate(trace_info):
        if i >= len(fiber_df_sorted):
            break
        spec = extract_flux(sci_data, tr['coeffs'], tr['fwhm'],
                            bg_subtract=args.bg_subtract)
        # Restrict to the user-specified band, if any. Done after extraction
        # so we still benefit from the full-length aperture trace, just sum
        # over a sub-range.
        if xrange_lo is not None:
            spec_used = spec[xrange_lo:xrange_hi]
        else:
            spec_used = spec
        b_sum, g_sum, r_sum = spectrum_to_bgr_block_sums(
            spec_used, reverse_wavelength=args.reverse_wavelength)

        records.append({
            'sky_x':      fiber_df_sorted.iloc[i]['sky_x'],
            'sky_y':      fiber_df_sorted.iloc[i]['sky_y'],
            'ID':         int(fiber_df_sorted.iloc[i]['ID']),
            'trace':      i + 1,
            'b_flux':     b_sum,
            'g_flux':     g_sum,
            'r_flux':     r_sum,
            'total_flux': float(np.sum(spec_used)),
        })
        if args.save_spectra:
            spectra.append(spec)  # save the FULL spectrum, not the sliced one

    df = pd.DataFrame(records)

    out_csv = f"{folder_name}_rgb_fluxes.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote per-fiber RGB block fluxes to: {out_csv}")

    if args.save_spectra:
        np.save(f"{folder_name}_fiber_spectra.npy", np.asarray(spectra))
        print(f"Wrote raw spectra to: {folder_name}_fiber_spectra.npy   "
              f"shape=({len(spectra)},{nx})")

    # ---- 4. Plot
    range_tag = f"_x{xrange_lo}-{xrange_hi}" if xrange_lo is not None else ""
    range_label = (f", x=[{xrange_lo},{xrange_hi})"
                   if xrange_lo is not None else "")

    if args.grayscale:
        title_suffix = (f"  [stretch={args.stretch}"
                        + (f", Q={args.Q}" if args.stretch == 'asinh' else "")
                        + f", cmap={args.grayscale_cmap}"
                        + range_label + "]")
        fig = plot_grayscale_map(
            df, folder_name,
            flux_col='total_flux',
            cmap=args.grayscale_cmap,
            single_panel=args.single_panel,
            marker_size=args.marker_size,
            label_fibers=args.label_fibers,
            stretch=args.stretch,
            Q=args.Q,
            title_suffix=title_suffix,
        )
        out_pdf = f"{folder_name}_grayscale_galaxy{range_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out_pdf}")

        if not args.no_show:
            plt.show()
        return

    # ---- 4b. RGB path: normalize -> RGB triplet per fiber
    raw_rgb = df[['r_flux', 'g_flux', 'b_flux']].to_numpy()
    rgb_norm = normalize_rgb(
        raw_rgb,
        stretch=args.stretch,
        percentile=args.percentile,
        Q=args.Q,
        scale=(args.scale_r, args.scale_g, args.scale_b),
        per_channel=args.per_channel_norm,
    )

    print(f"RGB stretch={args.stretch}, percentile={args.percentile}, "
          f"per_channel={args.per_channel_norm}")
    print(f"  raw channel medians  R={np.median(df.r_flux):.3e}  "
          f"G={np.median(df.g_flux):.3e}  B={np.median(df.b_flux):.3e}")
    print(f"  norm channel medians R={np.median(rgb_norm[:,0]):.3f} "
          f"G={np.median(rgb_norm[:,1]):.3f} B={np.median(rgb_norm[:,2]):.3f}")

    # ---- 5. Plot
    title_suffix = (f"  [stretch={args.stretch}"
                    + (f", Q={args.Q}" if args.stretch == 'asinh' else "")
                    + f", p{args.percentile:g}"
                    + range_label + "]")
    fig = plot_rgb_galaxy(
        df, rgb_norm, folder_name,
        single_panel=args.single_panel,
        marker_size=args.marker_size,
        label_fibers=args.label_fibers,
        bg_color=args.bg_color,
        title_suffix=title_suffix,
    )
    out_pdf = f"{folder_name}_rgb_galaxy{range_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_pdf}")

    fig_legend = plot_rgb_legend_panel(folder_name, title_suffix=title_suffix)
    out_legend = f"{folder_name}_rgb_legend{range_tag}.pdf"
    fig_legend.savefig(out_legend, bbox_inches='tight')
    print(f"Saved: {out_legend}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
