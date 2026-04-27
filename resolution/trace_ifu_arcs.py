#!/usr/bin/env python3
"""
trace_ifu_arcs.py
=================

Traces IFU fibers in an arc-lamp image, extracts 1-D arc spectra along each
trace, performs a wavelength calibration against a user-supplied line atlas,
measures arc-line FWHMs and finally produces a 2-D contour plot of the
spectral resolution metric  (FWHM_lambda / lambda)  as a function of
wavelength (x-axis) and trace ID (y-axis).

Designed for the same kind of IFU data the companion ``trace_ifu_lines.py``
script handles for flat lamps, but adapted for the lower S/N continuum and
sparse line emission of arc-lamp frames.

USAGE
-----
    python trace_ifu_arcs.py ARCFITS --atlas ATLAS.txt [other options]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


# ============================================================================
# Atlas handling
# ============================================================================
def load_atlas(atlas_path):
    """Load a SALT-style line atlas: ``# wavelength  intensity  comment``.

    Lines beginning with ``#`` are skipped.  Lines whose intensity is exactly
    zero are also dropped (they're flagged as unusable in the atlas itself).
    """
    wl, intens, comment = [], [], []
    with open(atlas_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0]); inten = float(parts[1])
            except ValueError:
                continue
            if inten <= 0:
                continue                            # dummy / disabled
            wl.append(w); intens.append(inten)
            comment.append(parts[2] if len(parts) >= 3 else "")
    wl = np.array(wl); intens = np.array(intens)
    order = np.argsort(wl)
    return wl[order], intens[order], np.array(comment)[order]


def parse_atlas_filename(path):
    """Try to parse a SALT-style atlas filename like
    ``Ar_PG0900_GA13_6_CA27_2.txt``  →  ('Ar', 'PG0900', 13.6, 27.2).
    Returns whatever it can find; missing fields come back as None."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    elem, grating, ga, ca = None, None, None, None
    if parts:
        elem = parts[0]

    def _read_decimal(parts, idx, prefix):
        """Read a number that begins after `prefix` at position idx, possibly
        continuing into the next part as a fractional component (so 'GA13'
        followed by '6' becomes 13.6)."""
        cur = parts[idx][len(prefix):]
        if (idx + 1) < len(parts):
            nxt = parts[idx + 1]
            # next part must be a pure (positive) integer to count as a fraction
            if nxt.isdigit():
                cur = f"{cur}.{nxt}"
                consumed = 2
            else:
                consumed = 1
        else:
            consumed = 1
        try:
            return float(cur.replace(",", ".")), consumed
        except ValueError:
            return None, consumed

    i = 0
    while i < len(parts):
        p = parts[i].upper()
        if p.startswith("PG") and p[2:].isdigit():
            grating = p
            i += 1
        elif p.startswith("GA"):
            ga, used = _read_decimal(parts, i, "GA")
            i += used
        elif p.startswith("CA"):
            ca, used = _read_decimal(parts, i, "CA")
            i += used
        else:
            i += 1
    return elem, grating, ga, ca


def validate_atlas_against_header(atlas_path, header, tol_angle=0.1):
    """Compare the element / grating / grating-angle / camera-angle parsed
    from the atlas filename with the corresponding FITS header keywords.
    Prints a clear, prominent warning if anything looks inconsistent.
    Always continues – the user might have intentionally provided a
    different atlas (e.g. an Ar+Xe combined list)."""
    elem, grating, ga, ca = parse_atlas_filename(atlas_path)

    h_lamp    = str(header.get("LAMPID", "")).strip()
    h_grating = str(header.get("GRATING", "")).strip().upper()
    h_ga      = header.get("GR-ANGLE", header.get("GRTILT", None))
    h_ca      = header.get("CAMANG",   header.get("CAM-ANG", None))

    issues = []
    if elem and h_lamp:
        # exact case-insensitive match, OR atlas element appears in the lamp
        # name (e.g. atlas 'Ar' for lamp 'CuAr')
        if elem.upper() != h_lamp.upper() and elem.upper() not in h_lamp.upper():
            issues.append(f"atlas element '{elem}' vs header LAMPID '{h_lamp}'")
    if grating and h_grating and grating != h_grating:
        issues.append(f"atlas grating '{grating}' vs header GRATING '{h_grating}'")
    if ga is not None and h_ga is not None and abs(ga - float(h_ga)) > tol_angle:
        issues.append(f"atlas GR-ANGLE {ga} vs header GR-ANGLE {h_ga}")
    if ca is not None and h_ca is not None and abs(ca - float(h_ca)) > tol_angle:
        issues.append(f"atlas CAMANG {ca} vs header CAMANG {h_ca}")

    if issues:
        bar = "!" * 78
        print(bar)
        print("WARNING: the supplied atlas may not match this exposure!")
        for s in issues:
            print(f"   - {s}")
        print("Wavelength calibration & resolution numbers may be unreliable.")
        print("Pass an atlas matching the lamp/grating setup if you have one.")
        print(bar)
    else:
        msg = []
        if elem:    msg.append(f"element={elem}")
        if grating: msg.append(f"grating={grating}")
        if ga is not None: msg.append(f"GA={ga}")
        if ca is not None: msg.append(f"CA={ca}")
        if msg:
            print(f"      atlas filename consistent with header  ({', '.join(msg)})")


# ============================================================================
# Initial wavelength solution from RSS grating geometry
# ============================================================================
def initial_wavelength_solution(header, nx,
                                 grating_density=None,
                                 focal_camera=330.0,
                                 pixel_size=0.015):
    """Compute a *very* approximate linear pixel -> wavelength mapping using
    the grating equation and the FITS header.

    Returns (lam0, disp) with lam0 the wavelength at x=0 and disp = dλ/dx
    (Å/pixel).  Good enough to seed the line matcher."""
    a = header.get("GR-ANGLE", header.get("GRTILT", 13.625))
    cam = header.get("CAMANG", header.get("CAM-ANG", 2 * a))
    grating = str(header.get("GRATING", "PG0900")).upper()
    if grating_density is None:
        # parse number out of e.g. PG0900 / PG1300
        digits = "".join(ch for ch in grating if ch.isdigit())
        grating_density = float(digits) if digits else 900.0

    a_rad = np.deg2rad(a)
    b_rad = np.deg2rad(cam - a)
    d_AA = 1.0e7 / grating_density                     # groove spacing in Å
    lam_c = d_AA * (np.sin(a_rad) + np.sin(b_rad))     # 1st order
    disp = d_AA * np.cos(b_rad) * pixel_size / focal_camera   # Å / pixel
    lam0 = lam_c - disp * (nx / 2.0)
    return lam0, disp, lam_c


# ============================================================================
# Centroiding helper (same idea as in the flat-lamp version)
# ============================================================================
def centroid(profile, peak_index, window=3):
    y = np.arange(peak_index - window, peak_index + window + 1)
    y = y[(y >= 0) & (y < len(profile))]
    w = profile[y]
    w = np.maximum(w - np.nanmin(w), 0)
    if np.sum(w) == 0:
        return float(peak_index)
    return float(np.sum(y * w) / np.sum(w))


# ============================================================================
# Step 1 – find the X positions of the brightest arc lines
# ============================================================================
def find_arc_anchor_columns(data, prom_factor=1.0, min_separation=10):
    """Collapse the image along the spatial direction and find the columns
    that are dominated by arc lines.  These will be used as anchor points
    for the spatial tracing (high S/N spatial profiles)."""
    ny = data.shape[0]
    band = np.median(data[ny // 4: 3 * ny // 4, :], axis=0)
    band_sm = gaussian_filter1d(band, sigma=1.5)
    # mask the chip-gap troughs so they don't create spurious anchor peaks
    masked = band_sm.copy()
    gap_mask = band_sm < np.nanmedian(band_sm) * 0.3
    masked[gap_mask] = 0.0
    arc_cols, _ = find_peaks(masked,
                              distance=min_separation,
                              prominence=np.nanmedian(band_sm) * prom_factor)
    return arc_cols, band, band_sm


# ============================================================================
# Step 2 – trace the fibers (sequential, anchored on arc-line columns)
# ============================================================================
def trace_fibers(data, anchor_cols, expected_traces=244,
                  sum_half=10, max_jump=8, poly_order=3,
                  prominence_pct=5):
    """Sequential tracing.

    sum_half     – how many columns either side of an anchor are summed
                   (so total = 2*sum_half+1, default 21 cols ≈ user's "20-30")
    max_jump     – max Δy allowed between adjacent anchors per trace
    poly_order   – polynomial order for the per-trace fit
    """
    ny, nx = data.shape

    def column_profile(x):
        x0 = max(0, x - sum_half)
        x1 = min(nx, x + sum_half + 1)
        return np.sum(data[:, x0:x1], axis=1)

    # ----- Reference column = arc anchor closest to the centre with ~244 peaks
    centre = nx // 2
    anchors_sorted = anchor_cols[np.argsort(np.abs(anchor_cols - centre))]
    ref_x = None; ref_peaks = None
    for x in anchors_sorted[:10]:
        p = column_profile(x)
        ps = gaussian_filter1d(p, sigma=1.2)
        peaks, _ = find_peaks(ps, distance=4,
                              prominence=np.nanpercentile(ps, 50) *
                              prominence_pct / 100.0)
        if abs(len(peaks) - expected_traces) <= 2:
            ref_x, ref_peaks = x, np.sort(peaks); break
    if ref_peaks is None:                            # fall back to the best
        for x in anchors_sorted[:10]:
            p = column_profile(x)
            ps = gaussian_filter1d(p, sigma=1.2)
            peaks, _ = find_peaks(ps, distance=4,
                                  prominence=np.nanpercentile(ps, 50)*0.05)
            if ref_peaks is None or abs(len(peaks)-expected_traces) < \
                                     abs(len(ref_peaks)-expected_traces):
                ref_x, ref_peaks = x, np.sort(peaks)

    n_traces = len(ref_peaks)
    print(f"   reference column x={ref_x}: {n_traces} traces detected "
          f"(expected {expected_traces})")
    trace_pts = {i: {"x": [ref_x], "y": [centroid(column_profile(ref_x),
                                                   ref_peaks[i], 2)]}
                 for i in range(n_traces)}

    # ----- Walk outwards from ref_x in both directions
    for direction in (+1, -1):
        if direction == +1:
            xs = sorted(c for c in anchor_cols if c > ref_x)
        else:
            xs = sorted((c for c in anchor_cols if c < ref_x), reverse=True)
        cur = np.array([trace_pts[i]["y"][0] for i in range(n_traces)])
        for x in xs:
            p = column_profile(x)
            ps = gaussian_filter1d(p, sigma=1.2)
            peaks, _ = find_peaks(ps, distance=4,
                                  prominence=np.nanpercentile(ps, 50) *
                                  prominence_pct / 100.0)
            if len(peaks) == 0:
                continue
            for i in range(n_traces):
                d = np.abs(peaks - cur[i])
                j = int(np.argmin(d))
                if d[j] < max_jump:
                    sub = centroid(p, peaks[j], 2)
                    trace_pts[i]["x"].append(x)
                    trace_pts[i]["y"].append(sub)
                    cur[i] = sub

    # ----- Polynomial fits per trace
    coeffs, kept_pts = [], []
    for i in range(n_traces):
        xs = np.array(trace_pts[i]["x"]); ys = np.array(trace_pts[i]["y"])
        if len(xs) >= poly_order + 2:
            c = np.polyfit(xs, ys, poly_order)
            coeffs.append(c); kept_pts.append((xs, ys))
        else:
            coeffs.append(None); kept_pts.append((xs, ys))
    return coeffs, kept_pts, ref_x


# ============================================================================
# Step 3 – extract a 1-D arc spectrum along each trace (vectorised)
# ============================================================================
def extract_spectra(data, coeffs, half_width=3):
    """Sum a (2*half_width+1)-pixel-tall aperture centred on the trace at
    every column.  Vectorised for speed and to keep memory low."""
    ny, nx = data.shape
    n = len(coeffs)
    x = np.arange(nx, dtype=np.int32)
    spectra = np.full((n, nx), np.nan, dtype=np.float32)
    for i, c in enumerate(coeffs):
        if c is None:
            continue
        yc_int = np.rint(np.polyval(c, x)).astype(np.int32)
        spec = np.zeros(nx, dtype=np.float32)
        valid = np.ones(nx, dtype=bool)
        for h in range(-half_width, half_width + 1):
            idx = yc_int + h
            ok = (idx >= 0) & (idx < ny)
            valid &= ok
            spec += np.where(ok, data[np.clip(idx, 0, ny - 1), x], 0.0)\
                      .astype(np.float32)
        spec[~valid] = np.nan
        spectra[i] = spec
    return spectra


# ============================================================================
# Step 4 – find arc-line peaks in each extracted 1-D spectrum
# ============================================================================
def estimate_continuum(spec, window=51, smooth_sigma=15.0):
    """Estimate a slowly-varying pseudo-continuum.

    Method: a running minimum filter (catches the noise floor between arc
    lines) followed by a uniform smoother (reduces the spikes of the min
    filter) and a Gaussian smoother (further softens the result).  Robust
    against NaNs at chip gaps.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    s = np.array(spec, dtype=np.float64)
    valid = np.isfinite(s) & (s > 0)
    if not valid.any():
        return np.zeros_like(s)
    fill = np.nanpercentile(s[valid], 30)
    s_filled = np.where(valid, s, fill)
    cont = minimum_filter1d(s_filled, size=window, mode="nearest")
    cont = uniform_filter1d(cont,    size=window, mode="nearest")
    cont = gaussian_filter1d(cont,   sigma=smooth_sigma, mode="nearest")
    # mark continuum invalid where the data was invalid
    cont = np.where(valid, cont, np.nan)
    return cont


def detect_lines_in_spectrum(spec, min_separation=6, snr_thresh=5.0,
                              continuum_window=51):
    """Robust peak-finder for a single extracted arc spectrum.

    A continuum is estimated and subtracted before peak finding so that
    weak lines sitting on a bright pseudo-continuum are no longer missed.
    Returns ``(peak_pixels, continuum, noise)``.
    """
    s = np.array(spec, dtype=np.float64)
    bad = ~np.isfinite(s)
    s[bad] = 0.0

    cont = estimate_continuum(s, window=continuum_window)
    cont_filled = np.where(np.isfinite(cont), cont, 0.0)
    sub = s - cont_filled
    sub[bad] = 0.0

    # noise estimate from the *sub-continuum* spectrum, in line-free regions
    pos = sub[(sub > 0) & (sub < np.nanpercentile(sub, 80))]
    noise = float(np.nanstd(pos)) + 1e-6 if len(pos) > 50 else 1.0
    prom = max(snr_thresh * noise, 0.02 * float(np.nanmax(sub)) if np.isfinite(np.nanmax(sub)) else 0.0)
    pk, _ = find_peaks(sub, distance=min_separation, prominence=prom)
    pk = pk[~bad[pk]]
    return pk, cont, noise


# ============================================================================
# Step 5 – match detected pixel peaks to atlas wavelengths
# ============================================================================
def auto_refine_initial_solution(spectrum, atlas_wl, atlas_int, lam0, disp,
                                  n_top_peaks=15, n_top_atlas=15,
                                  max_disp_change=0.30,
                                  min_range_frac=0.6,
                                  match_tol_pix=3.0):
    """RANSAC-style pair matching.

    For every pair of bright detected peaks and every pair of bright atlas
    lines we form a candidate linear solution (λ = λ0 + disp · pix) and
    score it by how many *all* detected peaks land within
    ``match_tol_pix · disp`` Å of an atlas line.  Tolerance is scaled by
    dispersion so that compressed-dispersion solutions cannot accumulate
    accidental matches.  Solutions whose wavelength range is too small
    relative to the atlas are rejected outright."""
    nx = len(spectrum)
    s = np.where(np.isfinite(spectrum), spectrum, 0.0).astype(np.float64)
    if not np.any(s > 0):
        return lam0, disp, 0

    # ---- detect peaks and rank by *prominence* (continuum-independent)
    base_prom = np.nanpercentile(s[s > 0], 60)
    pks, props = find_peaks(s, distance=6, prominence=base_prom * 0.3)
    if len(pks) < 4:
        pks, props = find_peaks(s, distance=6,
                                 prominence=np.nanstd(s[s > 0]) * 3)
    if len(pks) < 4:
        return lam0, disp, 0
    proms = props["prominences"]
    order = np.argsort(proms)[::-1]
    bright_pks = pks[order[:min(n_top_peaks, len(pks))]]

    # brightest atlas lines
    a_order = np.argsort(atlas_int)[::-1]
    bright_atlas = atlas_wl[a_order[:min(n_top_atlas, len(atlas_wl))]]

    disp_lo = disp * (1 - max_disp_change)
    disp_hi = disp * (1 + max_disp_change)
    atlas_range = atlas_wl.max() - atlas_wl.min()

    best_score, best_n, best = -1, 0, (lam0, disp)
    bp_sorted = np.sort(bright_pks)
    ba_sorted = np.sort(bright_atlas)
    for i in range(len(bp_sorted)):
        for j in range(i + 1, len(bp_sorted)):
            p1, p2 = bp_sorted[i], bp_sorted[j]
            if p2 - p1 < nx * 0.20:           # need decent baseline
                continue
            for k in range(len(ba_sorted)):
                for l in range(k + 1, len(ba_sorted)):
                    w1, w2 = ba_sorted[k], ba_sorted[l]
                    d_try = (w2 - w1) / (p2 - p1)
                    if not (disp_lo <= d_try <= disp_hi):
                        continue
                    l0_try = w1 - d_try * p1
                    # solution must cover at least min_range_frac of atlas
                    cov_lo = max(l0_try, atlas_wl.min())
                    cov_hi = min(l0_try + d_try * (nx - 1), atlas_wl.max())
                    if (cov_hi - cov_lo) < min_range_frac * atlas_range:
                        continue
                    tol_AA = match_tol_pix * d_try        # scaled tolerance
                    # score every detected peak (not just bright ones)
                    wl_pks = l0_try + pks * d_try
                    n_ok, score = 0, 0.0
                    for w_p, prom in zip(wl_pks, proms):
                        d_a = np.abs(atlas_wl - w_p)
                        m = int(np.argmin(d_a))
                        if d_a[m] < tol_AA:
                            n_ok += 1
                            score += atlas_int[m] * np.log10(1.0 + prom)
                    if score > best_score:
                        best_score, best_n = score, n_ok
                        best = (l0_try, d_try)
    return best[0], best[1], best_n


def per_trace_shift(peaks_pix, atlas_wl, atlas_int, lam0, disp,
                     shift_range=25.0, shift_step=0.5, tol_AA=3.0,
                     min_improvement=2):
    """For one trace, find a small wavelength shift Δλ that maximises the
    *atlas-intensity-weighted* number of detected peaks within tol_AA of an
    atlas line, given the global linear (lam0, disp).  The shift is only
    applied if it adds at least ``min_improvement`` more matches than
    sticking with Δλ = 0 — this stops marginally-different shifts from
    wandering into bad local optima for already-well-aligned traces."""
    if len(peaks_pix) == 0:
        return 0.0, 0
    wl_pks_base = lam0 + np.array(peaks_pix) * disp
    shifts = np.arange(-shift_range, shift_range + shift_step, shift_step)

    def count_at(sh):
        wl_pks = wl_pks_base + sh
        n, score = 0, 0.0
        for w_p in wl_pks:
            d = np.abs(atlas_wl - w_p); j = int(np.argmin(d))
            if d[j] < tol_AA:
                n += 1
                score += atlas_int[j]
        return n, score

    n0, score0 = count_at(0.0)
    best_n, best_score, best_shift = n0, score0, 0.0
    for sh in shifts:
        n, score = count_at(sh)
        if score > best_score:
            best_n, best_score, best_shift = n, score, sh
    # only accept the shift if it materially improves the match count
    if (best_shift != 0.0) and (best_n - n0 < min_improvement):
        return 0.0, n0
    return float(best_shift), int(best_n)


def match_peaks_to_atlas(peaks_pix, atlas_wl, lam0, disp,
                          tol_AA=10.0):
    """First pass match using a *linear* initial solution."""
    lam_at_peaks = lam0 + np.array(peaks_pix) * disp
    matched_pix, matched_wl = [], []
    for px, lam_guess in zip(peaks_pix, lam_at_peaks):
        d = np.abs(atlas_wl - lam_guess)
        j = int(np.argmin(d))
        if d[j] <= tol_AA:
            matched_pix.append(px); matched_wl.append(atlas_wl[j])
    return np.array(matched_pix), np.array(matched_wl)


def fit_wl_solution(matched_pix, matched_wl, order=3, n_sigma=3.0):
    """Sigma-clipped polynomial fit pixel -> wavelength."""
    if len(matched_pix) < order + 2:
        return None
    pix, wl = np.asarray(matched_pix, float), np.asarray(matched_wl, float)
    keep = np.ones_like(pix, bool)
    for _ in range(3):
        c = np.polyfit(pix[keep], wl[keep], order)
        res = wl - np.polyval(c, pix)
        sig = np.nanstd(res[keep])
        if sig == 0: break
        keep = np.abs(res) < n_sigma * sig
        if keep.sum() < order + 2: break
    return np.polyfit(pix[keep], wl[keep], order)


# ============================================================================
# Step 6 – measure FWHM of each matched arc line via Gaussian fit
# ============================================================================
def _gauss(x, A, mu, sig, b):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b

def measure_fwhm(spec, peak_pix, half_window=6):
    """Fit a Gaussian + constant background to the line.  Returns
    (fwhm_pix, centroid_pix) or (np.nan, np.nan) on failure.

    NaN pixels inside the window (e.g. chip gaps) are masked out instead
    of disqualifying the whole line."""
    nx = len(spec)
    x0 = max(0, peak_pix - half_window)
    x1 = min(nx, peak_pix + half_window + 1)
    xs_all = np.arange(x0, x1, dtype=float)
    ys_all = spec[x0:x1]
    good = np.isfinite(ys_all)
    # need enough good pixels both sides of the peak
    n_good = int(good.sum())
    if n_good < 5:
        return np.nan, np.nan
    xs, ys = xs_all[good], ys_all[good]
    A0 = float(np.nanmax(ys) - np.nanmin(ys))
    b0 = float(np.nanmin(ys))
    if A0 <= 0:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(_gauss, xs, ys,
                            p0=[A0, float(peak_pix), 2.0, b0],
                            maxfev=3000)
    except Exception:
        return np.nan, np.nan
    A, mu, sig, b = popt
    # reject obviously unphysical fits, but allow widths up to ~ window size
    if (not np.isfinite(sig)) or sig <= 0 or abs(sig) > 1.5 * half_window:
        return np.nan, np.nan
    if not (x0 - 2 <= mu <= x1 + 2):
        return np.nan, np.nan
    if A <= 0:
        return np.nan, np.nan
    return 2.3548 * abs(sig), mu


# ============================================================================
# Plotting helpers
# ============================================================================
def plot_traces_overview(data, coeffs, kept_pts, anchor_cols, outpath,
                          half_width=3):
    fig, ax = plt.subplots(figsize=(14, 7))
    # ----- background image: stretched to expose both the inter-fiber
    #       background and the bright arc lines without saturating
    pos = data[np.isfinite(data) & (data > 0)]
    vmin = np.nanpercentile(pos, 60)
    vmax = np.nanpercentile(pos, 99.7)
    im = ax.imshow(data, origin="lower", cmap="magma", aspect="auto",
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)         # keep PDF small; overlays stay vector

    plot_x = np.arange(data.shape[1])
    n_traces = sum(c is not None for c in coeffs)
    for i, c in enumerate(coeffs):
        if c is None:
            continue
        xs, ys = kept_pts[i]
        ax.scatter(xs, ys, s=0.4, color="cyan", alpha=0.35, zorder=3)
        y_fit = np.polyval(c, plot_x)
        # central trace
        ax.plot(plot_x, y_fit, color="lime",
                lw=0.18, alpha=0.55, zorder=4)
        # extraction-aperture edges (where the 1-D spectrum is summed)
        ax.plot(plot_x, y_fit + half_width, color="white",
                lw=0.10, alpha=0.40, ls="--", zorder=4)
        ax.plot(plot_x, y_fit - half_width, color="white",
                lw=0.10, alpha=0.40, ls="--", zorder=4)
    for x in anchor_cols:
        ax.axvline(x, color="white", lw=0.25, ls=":", alpha=0.30)
    ax.set_title(
        f"IFU arc tracing: {n_traces} fits "
        f"(anchored on {len(anchor_cols)} arc-line columns)  "
        f"— green=trace centre, white dashed=extraction aperture (±{half_width} px)")
    ax.set_xlabel("Dispersion [pixel]"); ax.set_ylabel("Spatial [pixel]")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"   wrote {outpath}")


def plot_resolution_scatter(line_records, outpath, title=""):
    """Per-line scatter version of the resolution map (no interpolation).
    Color encodes R = lambda / FWHM_lambda."""
    xs, ys, cs = [], [], []
    for i, rec in enumerate(line_records):
        if not rec:
            continue
        for w, fw_AA, _, _ in rec:
            if fw_AA > 0:
                xs.append(w); ys.append(i); cs.append(w / fw_AA)
    if not xs:
        return
    xs = np.array(xs); ys = np.array(ys); cs = np.array(cs)
    # remove obvious outliers
    med = np.nanmedian(cs); mad = np.nanmedian(np.abs(cs - med)) * 1.4826
    if mad > 0:
        m = np.abs(cs - med) < 5 * mad
        xs, ys, cs = xs[m], ys[m], cs[m]
    fig, ax = plt.subplots(figsize=(11, 6))
    vmin, vmax = np.nanpercentile(cs, 5), np.nanpercentile(cs, 95)
    sc = ax.scatter(xs, ys, c=cs, s=8, cmap="viridis",
                     vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"Resolving power  $R = \lambda / \Delta\lambda$")
    ax.set_xlabel(r"Wavelength $\lambda$  [Å]")
    ax.set_ylabel("Trace ID")
    ax.set_title(title or "Per-line resolution measurements (scatter)")
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close(fig)
    print(f"   wrote {outpath}")


def plot_resolution_map(wl_grid, R_map, outpath, title=""):
    """Headline 2-D plot of resolving power R."""
    fig, ax = plt.subplots(figsize=(11, 6))
    n_traces, n_wl = R_map.shape
    finite = np.isfinite(R_map)
    if not finite.any():
        print("   no finite resolution values – skipping plot")
        return
    coverage = finite.sum(axis=0) / max(1, n_traces)
    has = np.where(coverage > 0.05)[0]
    if len(has) > 0:
        wl_lo, wl_hi = wl_grid[has[0]], wl_grid[has[-1]]
    else:
        wl_lo, wl_hi = wl_grid[0], wl_grid[-1]

    vmin = np.nanpercentile(R_map[finite], 5)
    vmax = np.nanpercentile(R_map[finite], 95)
    im = ax.imshow(R_map, origin="lower", aspect="auto",
                   cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[wl_grid[0], wl_grid[-1], 0, n_traces],
                   interpolation="nearest")
    masked = np.ma.array(R_map, mask=~finite)
    levels = np.linspace(vmin, vmax, 5)
    try:
        ax.contour(np.linspace(wl_grid[0], wl_grid[-1], n_wl),
                    np.arange(n_traces) + 0.5,
                    masked, levels=levels,
                    colors="white", alpha=0.35, linewidths=0.6)
    except Exception:
        pass
    ax.set_xlim(wl_lo, wl_hi)

    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"Resolving power  $R = \lambda / \Delta\lambda$")
    ax.set_xlabel(r"Wavelength $\lambda$  [Å]")
    ax.set_ylabel("Trace ID")
    ax.set_title(title or "Spectral resolution map")
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close(fig)
    print(f"   wrote {outpath}")


def plot_per_trace_diagnostics(spectrum, all_peaks_pix, matched_recs,
                                wl_solution, atlas_wl, atlas_int,
                                trace_id, outpath_or_pdf, snr_thresh=5.0):
    """Diagnostic plot for a single trace.

    - red lines        : atlas wavelengths in this fiber's range
                         (line height encodes atlas intensity)
    - grey dashed      : ALL detected peaks (whether or not they were used)
    - green solid     : peaks that were matched and used in the wavelength
                         solution & FWHM measurement
    - orange thin     : estimated continuum that was subtracted before
                         peak finding (just for illustration)
    """
    nx = len(spectrum)
    px = np.arange(nx)
    wl = np.polyval(wl_solution, px)
    fig, ax = plt.subplots(figsize=(13, 4.5))

    # ----- spectrum --------------------------------------------------------
    safe = np.where(np.isfinite(spectrum) & (spectrum > 0), spectrum, np.nan)
    ymin_data = np.nanmin(safe) if np.any(np.isfinite(safe)) else 1.0
    ymax_data = np.nanmax(safe) if np.any(np.isfinite(safe)) else 1e3
    ymin = max(1.0, ymin_data)
    ymax = max(ymin * 10.0, ymax_data * 1.5)
    ax.plot(wl, spectrum, color="black", lw=0.7, zorder=4)

    # ----- estimated continuum (what the peak-finder subtracts) -----------
    cont = estimate_continuum(spectrum)
    ax.plot(wl, cont, color="orange", lw=0.7, alpha=0.7, zorder=4.5,
             label="continuum")

    # ----- atlas lines (single batched vlines call) -----------------------
    in_range = (atlas_wl >= wl.min()) & (atlas_wl <= wl.max())
    if np.any(in_range):
        ws = atlas_wl[in_range]
        ints = atlas_int[in_range]
        heights = ymin * np.power(ymax / ymin, 0.10 + 0.55 * ints)
        ax.vlines(ws, ymin, heights, color="red", lw=0.8,
                   alpha=0.7, zorder=2)

    # ----- detected peaks (single batched call) ---------------------------
    if len(all_peaks_pix) > 0:
        wls_det = np.polyval(wl_solution, all_peaks_pix)
        ax.vlines(wls_det, ymin, ymax, color="0.55", lw=0.4,
                   alpha=0.55, linestyles="dashed", zorder=3)

    # ----- matched peaks --------------------------------------------------
    if matched_recs:
        wls_mat = np.array([np.polyval(wl_solution, r[2])
                             for r in matched_recs])
        ax.vlines(wls_mat, ymin, ymax, color="green", lw=0.8,
                   alpha=0.75, zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(wl.min(), wl.max())
    ax.set_xlabel(r"Wavelength $\lambda$  [Å]")
    ax.set_ylabel("Counts")
    n_match = len(matched_recs) if matched_recs else 0
    n_det   = len(all_peaks_pix) if all_peaks_pix is not None else 0
    R_med = np.nan
    if matched_recs:
        Rs = [r[0] / r[1] for r in matched_recs if r[1] > 0]
        if Rs:
            R_med = np.nanmedian(Rs)
    ax.set_title(f"Trace {trace_id} — detected={n_det}, matched={n_match}, "
                 f"median R={R_med:.0f}   "
                 f"(red=atlas | grey dashed=detected | green=matched | "
                 f"orange=continuum)")
    plt.tight_layout()
    if hasattr(outpath_or_pdf, "savefig"):
        outpath_or_pdf.savefig(fig, dpi=110)
    else:
        plt.savefig(outpath_or_pdf, dpi=110)
    plt.close(fig)
    plt.clf()


# ============================================================================
# Main pipeline
# ============================================================================
def run(fits_path, atlas_path, ext=1, expected_traces=244,
         sum_half=10, half_width=3, poly_order_trace=3,
         poly_order_wl=3, snr_thresh=5.0, line_match_tol=10.0,
         lam0=None, disp=None, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    print("[1/7] Loading FITS …")
    with fits.open(fits_path, memmap=False) as hdul:
        data = np.ascontiguousarray(hdul[ext].data, dtype=np.float32)
        header = hdul[0].header
        for k in ("INSTRUME", "GRATING", "GR-ANGLE", "CAMANG",
                  "LAMPID", "OBJECT"):
            if k in header:
                print(f"      {k} = {header[k]}")
    ny, nx = data.shape
    print(f"      shape = ({ny}, {nx})")

    print("[2/7] Loading atlas …")
    atlas_wl, atlas_int, _ = load_atlas(atlas_path)
    print(f"      {len(atlas_wl)} usable lines, "
          f"{atlas_wl.min():.1f}-{atlas_wl.max():.1f} Å")
    validate_atlas_against_header(atlas_path, header)

    if lam0 is None or disp is None:
        l0_g, d_g, lc_g = initial_wavelength_solution(header, nx)
        if lam0 is None: lam0 = l0_g
        if disp is None: disp = d_g
        print(f"      initial λ-solution from header: λ0={lam0:.1f} Å, "
              f"disp={disp:.4f} Å/pix, λc≈{lc_g:.1f} Å")
    else:
        print(f"      user-supplied λ-solution: λ0={lam0:.1f} Å, "
              f"disp={disp:.4f} Å/pix")

    print("[3/7] Locating arc-line anchor columns …")
    anchors, _, _ = find_arc_anchor_columns(data, prom_factor=1.0)
    print(f"      found {len(anchors)} anchors, "
          f"x range {anchors.min()}-{anchors.max()}")

    print("[4/7] Tracing fibers …")
    coeffs, kept_pts, ref_x = trace_fibers(
        data, anchors, expected_traces=expected_traces,
        sum_half=sum_half, poly_order=poly_order_trace)
    n_good = sum(c is not None for c in coeffs)
    print(f"      {n_good}/{len(coeffs)} traces successfully fitted")
    plot_traces_overview(data, coeffs, kept_pts, anchors,
                         os.path.join(outdir, "traces_overview.pdf"),
                         half_width=half_width)

    print("[5/7] Extracting 1-D arc spectra …")
    spectra = extract_spectra(data, coeffs, half_width=half_width)
    # the raw 2-D image isn't needed any more — release ~100 MB before fitting
    del data
    import gc; gc.collect()
    n_traces = len(coeffs)

    # ---- Auto-refine the linear initial solution against the atlas using a
    #      central trace's bright-line pattern.  This corrects the systematic
    #      offset between the geometric grating-equation prediction and the
    #      true detector calibration.
    ref_idx = None
    for cand in range(n_traces // 2, n_traces):
        if np.isfinite(spectra[cand]).any() and \
           np.nanmax(spectra[cand]) > 50 * np.nanmedian(spectra[cand][spectra[cand] > 0]):
            ref_idx = cand; break
    if ref_idx is None:
        ref_idx = n_traces // 2
    print(f"      auto-refining initial λ-solution on trace {ref_idx} …")
    lam0_r, disp_r, score = auto_refine_initial_solution(
        spectra[ref_idx], atlas_wl, atlas_int, lam0, disp)
    print(f"      refined: λ0={lam0_r:.2f} Å, disp={disp_r:.4f} Å/pix "
          f"(shift {lam0_r-lam0:+.1f} Å, dispersion×{disp_r/disp:.3f})")
    lam0, disp = lam0_r, disp_r

    print("[6/7] Finding lines, matching to atlas, "
          "fitting wavelength solution & measuring FWHMs …")
    wl_solutions  = [None] * n_traces
    line_records  = [None] * n_traces        # list of (lam, fwhm_AA, mu_pix, fwhm_pix) tuples
    all_peaks     = [None] * n_traces        # ALL detected peaks per trace
    n_matched_per = np.zeros(n_traces, int)

    for i in range(n_traces):
        if coeffs[i] is None or not np.isfinite(spectra[i]).any():
            continue
        peaks_px, _, _ = detect_lines_in_spectrum(
            spectra[i], min_separation=6, snr_thresh=snr_thresh)
        all_peaks[i] = peaks_px
        if len(peaks_px) < poly_order_wl + 2:
            continue
        # ---- per-trace wavelength shift (fixes slit-image curvature)
        sh, _ = per_trace_shift(peaks_px, atlas_wl, atlas_int,
                                 lam0, disp, shift_range=80.0)
        m_pix, m_wl = match_peaks_to_atlas(peaks_px, atlas_wl,
                                            lam0 + sh, disp,
                                            tol_AA=line_match_tol)
        if len(m_pix) < poly_order_wl + 2:
            continue
        wsol = fit_wl_solution(m_pix, m_wl, order=poly_order_wl)
        if wsol is None:
            continue
        # second-pass match using the polynomial solution (tighter tol)
        wl_at_peaks = np.polyval(wsol, peaks_px)
        m_pix2, m_wl2 = [], []
        for px, lam in zip(peaks_px, wl_at_peaks):
            d = np.abs(atlas_wl - lam); j = int(np.argmin(d))
            if d[j] <= 3.0:
                m_pix2.append(px); m_wl2.append(atlas_wl[j])
        if len(m_pix2) >= poly_order_wl + 2:
            wsol2 = fit_wl_solution(np.array(m_pix2), np.array(m_wl2),
                                     order=poly_order_wl)
            if wsol2 is not None:
                wsol = wsol2
            m_pix, m_wl = np.array(m_pix2), np.array(m_wl2)

        wl_solutions[i] = wsol
        n_matched_per[i] = len(m_pix)

        rec = []
        for px, w in zip(m_pix, m_wl):
            fw_pix, mu_pix = measure_fwhm(spectra[i], px, half_window=6)
            if not np.isfinite(fw_pix):
                continue
            local_disp = abs(np.polyval(np.polyder(wsol), mu_pix))
            fw_AA = fw_pix * local_disp
            rec.append((w, fw_AA, mu_pix, fw_pix))
        line_records[i] = rec

    n_with_sol = sum(s is not None for s in wl_solutions)
    print(f"      wavelength solutions found for {n_with_sol}/{n_traces} traces")
    if n_with_sol > 0:
        avg_match = np.mean([n for n in n_matched_per if n > 0])
        print(f"      average matched lines per trace: {avg_match:.1f}")

    # Multi-page PDF showing every Nth trace so the user can verify the
    # atlas alignment.
    diag_pdf_path = os.path.join(outdir, "trace_spectra_diagnostics.pdf")
    diag_step = max(1, n_traces // 13)        # ~13 pages
    from matplotlib.backends.backend_pdf import PdfPages
    diag_indices = list(range(0, n_traces, diag_step))
    if (n_traces - 1) not in diag_indices:
        diag_indices.append(n_traces - 1)
    print(f"      writing diagnostic PDF "
          f"(every {diag_step}-th trace, {len(diag_indices)} pages) …",
          flush=True)
    with PdfPages(diag_pdf_path) as pdf:
        for k, i in enumerate(diag_indices):
            if wl_solutions[i] is None:
                continue
            plot_per_trace_diagnostics(
                spectra[i], all_peaks[i] if all_peaks[i] is not None else [],
                line_records[i], wl_solutions[i],
                atlas_wl, atlas_int, i, pdf, snr_thresh=snr_thresh)
            print(f"         page {k+1}/{len(diag_indices)} (trace {i})",
                  flush=True)
    print(f"      wrote {diag_pdf_path}", flush=True)

    print("[7/7] Building 2-D resolution map …")
    # Build a 2-D map by smoothing the per-line scatter onto a regular grid
    # using a wavelength × trace-ID Gaussian kernel.  Every robust line
    # measurement contributes; we sigma-clip per-trace first.  The plotted
    # quantity is the resolving power R = λ / FWHM_λ.
    pts_w, pts_t, pts_v = [], [], []
    for i, rec in enumerate(line_records):
        if not rec:
            continue
        ws  = np.array([r[0] for r in rec])
        fws = np.array([r[1] for r in rec])
        Rvals = ws / fws
        med  = np.nanmedian(Rvals)
        mad  = np.nanmedian(np.abs(Rvals - med)) * 1.4826
        if mad > 0:
            keep = np.abs(Rvals - med) < 4 * mad
            ws, Rvals = ws[keep], Rvals[keep]
        pts_w.extend(ws.tolist()); pts_t.extend([i] * len(ws))
        pts_v.extend(Rvals.tolist())
    pts_w = np.array(pts_w); pts_t = np.array(pts_t); pts_v = np.array(pts_v)

    if len(pts_w) > 50:
        wl_grid = np.linspace(np.percentile(pts_w, 1),
                               np.percentile(pts_w, 99), 250)
    else:
        wl_grid = np.linspace(atlas_wl.min(), atlas_wl.max(), 250)
    tr_grid = np.arange(n_traces)

    sigma_w = max(20.0, (wl_grid[-1] - wl_grid[0]) / 60.0)        # Å
    sigma_t = max(2.0,  n_traces / 80.0)                            # traces
    R_map = np.full((n_traces, len(wl_grid)), np.nan)
    if len(pts_v) > 0:
        for it, t in enumerate(tr_grid):
            wt = np.exp(-0.5 * ((pts_t - t) / sigma_t) ** 2)
            m = wt > 0.05
            if m.sum() < 2:
                continue
            ww = pts_w[m]; vv = pts_v[m]; wt_m = wt[m]
            dw = (wl_grid[:, None] - ww[None, :]) / sigma_w
            kw = np.exp(-0.5 * dw * dw) * wt_m[None, :]
            num = (kw * vv[None, :]).sum(axis=1)
            den = kw.sum(axis=1)
            ok  = den > 1e-3
            row = np.full(len(wl_grid), np.nan)
            row[ok] = num[ok] / den[ok]
            R_map[it] = row

    plot_resolution_map(
        wl_grid, R_map,
        os.path.join(outdir, "resolution_map.png"),
        title=(f"Spectral resolution map — {os.path.basename(fits_path)}"
               f"   ({n_with_sol} of {n_traces} traces)"))
    plot_resolution_scatter(
        line_records,
        os.path.join(outdir, "resolution_scatter.png"),
        title=(f"Per-line resolution measurements — "
               f"{os.path.basename(fits_path)}"))

    # Also dump a CSV-ish summary
    summary = os.path.join(outdir, "resolution_summary.txt")
    with open(summary, "w") as f:
        f.write("# trace_id  n_matched  median_R  median_FWHM_AA\n")
        for i, rec in enumerate(line_records):
            if not rec: continue
            ws = np.array([r[0] for r in rec])
            fws = np.array([r[1] for r in rec])
            R = np.nanmedian(ws / fws)
            f.write(f"{i:4d}   {len(rec):4d}   {R:8.1f}   "
                    f"{np.nanmedian(fws):.4f}\n")
    print(f"      wrote {summary}")
    print("Done.")
    return dict(coeffs=coeffs, spectra=spectra,
                wl_solutions=wl_solutions, line_records=line_records,
                wl_grid=wl_grid, R_map=R_map, anchors=anchors)


# ============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file",                          help="arc lamp FITS file")
    p.add_argument("--atlas",  required=True,       help="line atlas text file")
    p.add_argument("--ext",    type=int, default=1, help="FITS image extension")
    p.add_argument("--ntraces",type=int, default=244,help="expected fiber count")
    p.add_argument("--sum_half", type=int, default=10,
                   help="half-width of column sum for tracing (default 10 → 21 cols)")
    p.add_argument("--half_width", type=int, default=4,
                   help="half-width of spatial extraction aperture (pixels);"
                        " total = 2*half_width+1 (default 4 → 9 px)")
    p.add_argument("--poly_trace", type=int, default=3, help="trace polynomial order")
    p.add_argument("--poly_wl",    type=int, default=3, help="wavelength solution order")
    p.add_argument("--snr",        type=float, default=20.0,
                   help="SNR threshold for arc line detection")
    p.add_argument("--match_tol",  type=float, default=200,
                   help="initial atlas matching tolerance (Å)")
    p.add_argument("--lam0", type=float, default=None,
                   help="override initial wavelength at x=0 (Å)")
    p.add_argument("--disp", type=float, default=None,
                   help="override initial dispersion (Å/pix)")
    p.add_argument("--outdir", default=".", help="output directory")
    args = p.parse_args()

    run(args.file, args.atlas, ext=args.ext,
        expected_traces=args.ntraces,
        sum_half=args.sum_half, half_width=args.half_width,
        poly_order_trace=args.poly_trace, poly_order_wl=args.poly_wl,
        snr_thresh=args.snr, line_match_tol=args.match_tol,
        lam0=args.lam0, disp=args.disp, outdir=args.outdir)


if __name__ == "__main__":
    main()
