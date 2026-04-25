"""
Throughput map for already-extracted spectra.

Companion to extract_flux_throughput.py. Use this when both input FITS
files are reduced products in which each pixel ROW is already one
extracted trace (the spectrum of one fiber along the dispersion axis).
No trace following or aperture extraction is needed — the rows are the
extracted spectra.

Pipeline
--------
1. Build a 1-D reference spectrum: mean of the central 10 rows of image1.
2. For every row i of image2, sum the row and divide by the sum of the 
   reference spectrum to get one scalar per fiber.
3. Divide by SLIT_FIBER_AREA_RATIO (= 1.152) so the result represents
   fiber-to-fiber throughput rather than fiber-to-slit collection.
4. Map row index → fiber by sorting final_data.csv on slit_x, then plot
   with the same broken-axis routine as the trace-following script.

Usage
-----
    python extract_throughput_reduced.py reduced1.fits reduced2.fits \
        --csv final_data.csv [--no-show]
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits


# Slit aperture is slightly larger than a fiber, so the per-trace slit flux
# is inflated by this factor relative to a fiber-sized aperture would
# collect. The throughput value is divided by this to compensate.
SLIT_FIBER_AREA_RATIO = 1.152

# Number of central rows of image1 averaged to form the reference spectrum.
N_CENTRAL_REF_ROWS = 10


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_throughput_map(df, folder_name, ratio_col='throughput',
                        cbar_label='Throughput',
                        title_prefix='Throughput',
                        vmin=0.0, vmax=0.6, cmap='RdYlGn'):
    """Three-panel broken-axis fiber map with break marks and equal aspect.
    
    Defaulting scale to 0-60% (0.0 to 0.6) and using RdYlGn colormap.
    """
    xlims = [(-19.5, -15), (-4, 4), (15, 19.5)]
    widths = [b - a for a, b in xlims]
    ymin = df['sky_y'].min() - 1
    ymax = df['sky_y'].max() + 1

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

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical',
                        fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=11)

    med = np.nanmedian(df[ratio_col])
    fig.suptitle(
        f"{title_prefix}: {folder_name}   "
        f"(N={len(df)}, median={med:.3f}, "
        f"scale=[{vmin:.1f}, {vmax:.1f}])",
        fontsize=13, y=0.96,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Throughput map from two reduced (already-extracted) FITS files."
    )
    parser.add_argument("image1", help="Reference reduced FITS (each row = one trace)")
    parser.add_argument("image2", help="Comparison reduced FITS (each row = one trace)")
    parser.add_argument("--csv", default="final_data.csv",
                        help="Fiber position CSV (must contain ID, sky_x, sky_y, slit_x)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't pop up the matplotlib window")
    args = parser.parse_args()

    path_parts = args.image2.split(os.sep)
    folder_name = path_parts[-2] if len(path_parts) > 1 else "output"

    # ---------- Load -----------------------------------------------------
    data1 = fits.getdata(args.image1).astype(float)
    data2 = fits.getdata(args.image2).astype(float)
    fiber_df = pd.read_csv(args.csv)

    if data1.shape != data2.shape:
        raise ValueError(f"Image shapes differ: {data1.shape} vs {data2.shape}.")

    n_traces, nx = data1.shape
    print(f"Loaded {args.image1}: shape {data1.shape}")
    print(f"Loaded {args.image2}: shape {data2.shape}")

    # ---------- Reference spectrum --------------------------------------
    half = N_CENTRAL_REF_ROWS // 2
    mid = n_traces // 2
    lo = max(0, mid - half)
    hi = min(n_traces, mid + (N_CENTRAL_REF_ROWS - half))
    ref = np.mean(data1[lo:hi, :], axis=0)
    
    bad = ~np.isfinite(ref) | (ref <= 0)
    good = ~bad

    # ---------- Throughput per row (Sum then Ratio) ---------------------
    ref_sum = np.nansum(ref[good])
    image2_sums = np.nansum(data2[:, good], axis=1)
    
    raw_sum = image2_sums / ref_sum
    throughput = raw_sum / SLIT_FIBER_AREA_RATIO
    
    print(f"Throughput stats: median={np.nanmedian(throughput):.3e}")

    # ---------- Map row → fiber -----------------------------------------
    fiber_df_sorted = fiber_df.sort_values('slit_x').reset_index(drop=True)
    n_match = min(n_traces, len(fiber_df_sorted))

    records = []
    for i in range(n_match):
        records.append({
            'sky_x':      fiber_df_sorted.iloc[i]['sky_x'],
            'sky_y':      fiber_df_sorted.iloc[i]['sky_y'],
            'ID':         int(fiber_df_sorted.iloc[i]['ID']),
            'trace':      i + 1,
            'raw_sum':    float(raw_sum[i]),
            'throughput': float(throughput[i]),
        })
    df = pd.DataFrame(records)

    out_csv = f"{folder_name}_throughput_reduced.csv"
    df.to_csv(out_csv, index=False)

    # ---------- Plot -----------------------------------------------------
    # vmin=0.0 and vmax=0.6 sets the 0-60% scale.
    # cmap='RdYlGn' sets the Red-Yellow-Green colormap.
    fig = plot_throughput_map(
        df, folder_name,
        ratio_col='throughput',
        cbar_label=f'Σ(image2) / Σ(ref) / {SLIT_FIBER_AREA_RATIO}',
        title_prefix='Reduced-File Throughput (Sum-then-Ratio)',
        vmin=0.0,
        vmax=0.6,
        cmap='RdYlGn'
    )
    pdf = f"{folder_name}_throughput_reduced_map.pdf"
    fig.savefig(pdf, bbox_inches='tight')
    print(f"Saved: {pdf}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()