"""
wing_profile.py
---------------
Fit a Gaussian-core + Lorentzian-wing profile at 21 pinhole locations
(3 x-positions x 7 traces) in a single FITS frame, to characterise how
severe the PSF wings are.

Model (independent widths):
    f(y) = A_G * exp(-0.5*((y-mu)/sigma_G)^2)
         + A_L / (1 + ((y-mu)/gamma)^2)
         + offset

Suggested wing-severity parameter — Lorentzian flux fraction:
    eta = F_L / (F_G + F_L)
where F_G = A_G*sigma_G*sqrt(2*pi) and F_L = A_L*gamma*pi.
    eta = 0     -> pure Gaussian (no wings)
    eta ~ 0.1   -> mild wings
    eta >= 0.3  -> noticeable wings
    eta >= 0.5  -> severe (more flux in wings than core)
    eta = 1     -> pure Lorentzian

Usage:
    python wing_profile.py frame.fits
    python wing_profile.py frame.fits --out my_wings.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # ~2.3548


# ===========================================================================
# I/O + x-position selection + peak detection (same as focus_sweep.py)
# ===========================================================================

def load_image(fits_path, ext=0):
    with fits.open(fits_path) as hdul:
        data = hdul[ext].data
        if data is None or np.ndim(data) != 2:
            for h in hdul:
                if h.data is not None and np.ndim(h.data) == 2:
                    return np.asarray(h.data, dtype=float)
            raise ValueError(f"No 2-D image in {fits_path}")
        return np.asarray(data, dtype=float)


def find_x_positions(data, n=3, threshold_frac=0.1):
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


def detect_trace_peaks(cut, expected_traces=7, min_distance=30):
    med = np.median(cut)
    mad = np.median(np.abs(cut - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(cut)
    prominence = max(3.0 * sigma, 1e-6)
    peaks, props = find_peaks(cut, distance=min_distance, prominence=prominence)
    if len(peaks) > expected_traces:
        order = np.argsort(props["prominences"])[::-1]
        peaks = peaks[order[:expected_traces]]
    return np.sort(peaks)


# ===========================================================================
# Gaussian-core + Lorentzian-wing fit
# ===========================================================================

def gauss_lorentz(y, amp_g, mu, sigma_g, amp_l, gamma, offset):
    """Gaussian core + Lorentzian wings + constant offset."""
    g = amp_g * np.exp(-0.5 * ((y - mu) / sigma_g) ** 2)
    l = amp_l / (1.0 + ((y - mu) / gamma) ** 2)
    return g + l + offset


def fit_gauss_lorentz(cut, y_guess, half_window=15):
    """
    Fit Gaussian + Lorentzian to `cut` centred near y_guess.

    The window is intentionally wide (default half_window=15) because wings
    need support — a narrow window gives the Lorentzian nothing to fit.

    Returns a dict with: yy, prof, popt, fit_ok, fwhm_gauss, wing_frac,
    width_ratio.
    """
    y_int = int(round(y_guess))
    y0 = max(0, y_int - half_window)
    y1 = min(len(cut), y_int + half_window + 1)
    yy = np.arange(y0, y1)
    prof = cut[y0:y1].astype(float)

    result = dict(
        y_guess=float(y_guess), yy=yy, prof=prof,
        popt=None, fit_ok=False,
        fwhm_gauss=np.nan, wing_frac=np.nan, width_ratio=np.nan,
    )

    if len(yy) < 8:
        return result
    offset0 = float(np.percentile(prof, 10))   # robust baseline
    amp_tot = float(prof.max() - offset0)
    if amp_tot <= 0:
        return result

    # Start with ~90% Gaussian, 10% Lorentzian; slightly wider Lorentzian.
    p0 = [0.9 * amp_tot,  y_guess, 2.0,
          0.1 * amp_tot,  3.0,     offset0]
    bounds_lo = [0.0,     yy[0],   0.3,
                 0.0,     0.5,    -np.inf]
    bounds_hi = [np.inf,  yy[-1],  15.0,
                 np.inf,  30.0,    np.inf]

    try:
        popt, _ = curve_fit(
            gauss_lorentz, yy, prof, p0=p0,
            bounds=(bounds_lo, bounds_hi), maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return result

    amp_g, mu, sigma_g, amp_l, gamma, offset = popt
    result["popt"] = popt

    # Sanity check on the Gaussian width
    if sigma_g < 0.3 or sigma_g > 0.5 * (yy[-1] - yy[0]):
        return result

    # Flux fractions (analytic integrals)
    flux_g = amp_g * sigma_g * np.sqrt(2.0 * np.pi)
    flux_l = amp_l * gamma * np.pi
    total = flux_g + flux_l

    result["fit_ok"] = True
    result["fwhm_gauss"] = SIGMA_TO_FWHM * sigma_g
    result["wing_frac"] = float(flux_l / total) if total > 0 else np.nan
    result["width_ratio"] = float(gamma / sigma_g) if sigma_g > 0 else np.nan
    return result


# ===========================================================================
# Per-frame processing
# ===========================================================================

def process_frame(fits_path, n_traces=7, n_positions=3, col_halfwidth=2,
                  fwhm_window=15, min_distance=30):
    data = load_image(fits_path)
    x_positions = find_x_positions(data, n=n_positions)

    fits_by_position = []
    for x in x_positions:
        x0 = max(0, x - col_halfwidth)
        x1 = min(data.shape[1], x + col_halfwidth + 1)
        cut = np.median(data[:, x0:x1], axis=1)
        peaks = detect_trace_peaks(cut, expected_traces=n_traces,
                                   min_distance=min_distance)
        row = []
        for y in peaks:
            entry = fit_gauss_lorentz(cut, y, half_window=fwhm_window)
            entry["x"] = int(x)
            row.append(entry)
        while len(row) < n_traces:
            row.append(dict(x=int(x), y_guess=np.nan, yy=None, prof=None,
                            popt=None, fit_ok=False, fwhm_gauss=np.nan,
                            wing_frac=np.nan, width_ratio=np.nan))
        fits_by_position.append(row)

    per_trace_fits = [
        [fits_by_position[j][i] for j in range(n_positions)]
        for i in range(n_traces)
    ]
    return dict(
        path=Path(fits_path),
        data=data,
        x_positions=x_positions,
        per_trace_fits=per_trace_fits,
    )


# ===========================================================================
# 7-panel log-scale plot
# ===========================================================================

def plot_wings(per_trace_fits, path, out_png, expected_traces=7):
    """7 panels, each overlaying the 3 profiles and Gaussian+Lorentzian fits
    on a log-y scale (wings visible). Dashed curve = the Gaussian component
    alone, so the Lorentzian excess at large |dy| is obvious."""
    n_traces = len(per_trace_fits)
    n_panels = max(n_traces, expected_traces)
    ncols = 4 if n_panels >= 4 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.9 * ncols, 3.2 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for i in range(n_panels):
        ax = axes[i]
        if i >= n_traces:
            ax.set_visible(False)
            continue
        row = per_trace_fits[i]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(row)))

        y_min_positive = np.inf   # track for sensible ylim
        for entry, color in zip(row, colors):
            if entry["yy"] is None or entry["prof"] is None or entry["popt"] is None:
                continue
            yy = entry["yy"]
            prof = entry["prof"]
            popt = entry["popt"]
            amp_g, mu, sigma_g, amp_l, gamma, offset = popt

            prof_sub = prof - offset
            pos = prof_sub > 0
            if np.any(pos):
                y_min_positive = min(y_min_positive, prof_sub[pos].min())

            wf = entry["wing_frac"]
            wr = entry["width_ratio"]
            wf_str = f"η={wf:.2f}" if np.isfinite(wf) else "η=nan"
            wr_str = f"γ/σ={wr:.1f}" if np.isfinite(wr) else ""
            label = f"x={entry['x']}  {wf_str}  {wr_str}"

            # Data points (only positive values on log scale)
            ax.plot(yy[pos] - mu, prof_sub[pos], "o",
                    color=color, ms=3.8, alpha=0.85, label=label)

            # Full fit (Gaussian + Lorentzian)
            yy_fine = np.linspace(yy[0], yy[-1], 500)
            full = gauss_lorentz(yy_fine, *popt) - offset
            ax.plot(yy_fine - mu, full, "-", color=color, lw=1.0, alpha=0.95)

            # Gaussian component only, dashed — so the wing excess pops out
            gauss_only = amp_g * np.exp(-0.5 * ((yy_fine - mu) / sigma_g) ** 2)
            ax.plot(yy_fine - mu, gauss_only, "--",
                    color=color, lw=0.8, alpha=0.45)

        ax.set_yscale("log")
        if np.isfinite(y_min_positive):
            ax.set_ylim(bottom=max(0.5 * y_min_positive, 1e-3))
        ax.set_title(f"Trace {i + 1}", fontsize=10)
        ax.set_xlabel(r"$y - \mu$ [pix]")
        ax.set_ylabel("counts - offset")
        ax.legend(fontsize=6.5, loc="lower center")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Gaussian + Lorentzian profiles: {path.name}\n"
        r"Solid = full fit,  dashed = Gaussian component only.  "
        r"$\eta$ = wing flux fraction (0 = pure Gaussian).  "
        r"$\gamma/\sigma$ = Lorentzian HWHM / Gaussian $\sigma$.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_png}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("fits_path", type=Path, help="Single FITS file")
    p.add_argument("--n-traces", type=int, default=7)
    p.add_argument("--n-positions", type=int, default=3)
    p.add_argument("--col-halfwidth", type=int, default=2)
    p.add_argument("--fwhm-window", type=int, default=15,
                   help="Half-window in y for the fit (wider than pure-Gaussian "
                        "case so the wings have support). Default 15.")
    p.add_argument("--min-distance", type=int, default=30)
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <stem>_wings.png next to input)")
    args = p.parse_args()

    if not args.fits_path.is_file():
        raise SystemExit(f"No such file: {args.fits_path}")

    print(f"=== {args.fits_path.name} ===")
    r = process_frame(
        args.fits_path,
        n_traces=args.n_traces,
        n_positions=args.n_positions,
        col_halfwidth=args.col_halfwidth,
        fwhm_window=args.fwhm_window,
        min_distance=args.min_distance,
    )

    print(f"  x-positions: {list(r['x_positions'])}")
    print(f"  {'Trace':<7} {'median η':>10} {'median γ/σ':>12} "
          f"{'median FWHM_core':>18}")
    all_eta = []
    for i, row in enumerate(r["per_trace_fits"]):
        etas = [e["wing_frac"] for e in row if np.isfinite(e["wing_frac"])]
        wrs = [e["width_ratio"] for e in row if np.isfinite(e["width_ratio"])]
        fws = [e["fwhm_gauss"] for e in row if np.isfinite(e["fwhm_gauss"])]
        e_s = f"{np.median(etas):.3f}" if etas else "nan"
        w_s = f"{np.median(wrs):.2f}" if wrs else "nan"
        f_s = f"{np.median(fws):.3f}" if fws else "nan"
        print(f"  {i + 1:<7} {e_s:>10} {w_s:>12} {f_s:>18}")
        all_eta.extend(etas)
    if all_eta:
        print(f"\n  Frame median η = {np.median(all_eta):.3f}  "
              f"(wing flux fraction — higher means more severe wings)")

    out = args.out or args.fits_path.with_name(
        args.fits_path.stem + "_wings.png")
    plot_wings(r["per_trace_fits"], args.fits_path, out,
               expected_traces=args.n_traces)


if __name__ == "__main__":
    main()
