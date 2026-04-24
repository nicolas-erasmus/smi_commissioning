#!/usr/bin/env python3
"""
Unified Dither Tracking & In-Place Astrometric Calibration.
Fixes: Dec sign error, Multi-extension Header access, and 1:1 Plot scaling.
"""

import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
import sep
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
from PIL import Image

# --- Settings ---
PLATE_SCALE = 0.1267  # arcsec/px
ast = AstrometryNet()

def load_config():
    key_file = Path("key.json")
    if not key_file.exists():
        raise FileNotFoundError("Missing 'key.json'")
    with open(key_file, "r") as f:
        config = json.load(f)
    ast.api_key = config["astro_api_key"]

def is_wcs_valid(header):
    ctype1 = header.get('CTYPE1', '').upper().strip()
    if not ctype1 or 'PIXEL' in ctype1:
        return False
    return any(t in ctype1 for t in ['RA-', 'DEC-', 'GLON', 'GLAT'])

def get_sources(data, limit=40, thresh=10.0):
    data = data.byteswap().newbyteorder() if data.dtype.byteorder != '=' else data
    data = data.astype(np.float32)
    try:
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        objects = sep.extract(data_sub, thresh, err=bkg.globalrms)
        if len(objects) > 0:
            x, y, _ = sep.winpos(data_sub, objects['x'], objects['y'], 1.0)
            idx = np.argsort(objects['flux'])[::-1]
            return x[idx][:limit], y[idx][:limit]
    except: pass
    return None, None

def solve_and_update(file_path):
    with fits.open(file_path) as hdul:
        if any(is_wcs_valid(h.header) for h in hdul if h.data is not None):
            return True
        idx = next((i for i, h in enumerate(hdul) if h.data is not None), None)
        if idx is None: return False
        img = hdul[idx].data.astype(float)
        ny, nx = img.shape

    x, y = get_sources(img)
    if x is None: return False

    print(f"Solving {file_path.name}...")
    try:
        wcs_header = ast.solve_from_source_list(x, y, nx, ny, solve_timeout=60)
        if wcs_header:
            with fits.open(file_path, mode='update') as hdul:
                target = next(h for h in hdul if h.data is not None)
                for k in ['WCSAXES', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                    if k in target.header: del target.header[k]
                target.header.update(wcs_header)
                hdul.flush()
            return True
    except: pass
    return False

def to_deg(val, is_ra=True):
    if val is None: return None
    try:
        if isinstance(val, str) and ':' in val:
            parts = val.strip().split(':')
            sign = -1.0 if '-' in parts[0] else 1.0
            hms = [abs(float(p)) for p in parts]
            deg = sign * (hms[0] + hms[1]/60.0 + hms[2]/3600.0)
            return deg * 15.0 if is_ra else deg
        v = float(val)
        return v * 15.0 if (is_ra and v < 24) else v
    except: return None

def save_cutout(image_data, pos, frame_name, star_idx, output_dir):
    output_dir.mkdir(exist_ok=True, parents=True)
    x_img, y_img = pos
    box = 15
    ix, iy = int(round(x_img)), int(round(y_img))
    y_min, y_max = max(0, iy-box), min(image_data.shape[0], iy+box+1)
    x_min, x_max = max(0, ix-box), min(image_data.shape[1], ix+box+1)
    cut = image_data[y_min:y_max, x_min:x_max]
    plt.figure(figsize=(4,4))
    plt.imshow(cut, cmap='gray', origin='lower', vmin=np.percentile(cut, 5), vmax=np.percentile(cut, 99))
    plt.plot(x_img-x_min, y_img-y_min, 'rx')
    plt.axis('off')
    plt.savefig(output_dir / f"{frame_name}_star{star_idx+1}.png", bbox_inches='tight')
    plt.close()

def run_analysis(folder):
    folder = Path(folder)
    cutout_dir = folder / "cutouts"
    files = sorted(list(folder.glob('*.fit*')))
    for f in files: solve_and_update(f)

    ref_img = None
    for f in files:
        with fits.open(f) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu and is_wcs_valid(hdu.header):
                ref_img = hdu.data.astype(float)
                break
    if ref_img is None: return

    sx, sy = get_sources(ref_img, limit=200, thresh=5.0)
    fig, ax = plt.subplots(); ax.imshow(ref_img, cmap='gray', origin='lower', vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 95))
    selected = []
    def onclick(event):
        if event.inaxes == ax:
            idx = np.argmin(np.sqrt((sx-event.xdata)**2 + (sy-event.ydata)**2))
            selected.append((sx[idx], sy[idx])); ax.plot(sx[idx], sy[idx], 'ro', mfc='none'); fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close(fig) if e.key=='enter' else None)
    plt.show()

    if not selected: return

    valid_stems, wcs_centers, tel_coords, pixel_positions = [], [], [], []
    last_p = selected

    for f in files:
        with fits.open(f) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu and is_wcs_valid(hdu.header):
                # Pull Header info from Primary HDU [0]
                pri_hdr = hdul[0].header
                tra = to_deg(pri_hdr.get('TELRA', pri_hdr.get('RA')), True)
                tdec = to_deg(pri_hdr.get('TELDEC', pri_hdr.get('DEC')), False)
                
                # Pull WCS from current Image HDU
                w = WCS(hdu.header); ny, nx = hdu.data.shape
                wra, wdec = w.all_pix2world(nx/2, ny/2, 0)
                
                cx, cy = get_sources(hdu.data.astype(float), limit=200, thresh=5.0)
                curr_p = []
                for rx, ry in last_p:
                    d = np.sqrt((cx-rx)**2 + (cy-ry)**2); idx = np.argmin(d)
                    curr_p.append((cx[idx], cy[idx]) if d[idx] < 20 else None)
                
                if all(p is not None for p in curr_p):
                    valid_stems.append(f.stem); wcs_centers.append((wra, wdec))
                    tel_coords.append((tra, tdec)); pixel_positions.append(curr_p)
                    for i, p in enumerate(curr_p): save_cutout(hdu.data, p, f.stem, i, cutout_dir)
                    last_p = curr_p

    # Final Plotting
    if len(valid_stems) > 1:
        for mode in ['TEL', 'WCS']:
            coords = tel_coords if mode == 'TEL' else wcs_centers
            if any(c[0] is None for c in coords): continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            r0, d0 = coords[0]
            
            for i in range(len(selected)):
                dx = np.array([p[i][0] - pixel_positions[0][i][0] for p in pixel_positions]) * PLATE_SCALE
                # SIGN FIX: Negating dy to account for orientation inversion
                dy = -np.array([p[i][1] - pixel_positions[0][i][1] for p in pixel_positions]) * PLATE_SCALE
                
                dr = np.array([(c[0]-r0)*3600*np.cos(np.radians(d0)) for c in coords])
                dd = np.array([(c[1]-d0)*3600 for c in coords])
                ax1.plot(dx, dr, 'o-', label=f"Star {i+1}"); ax2.plot(dy, dd, 'o-')

            for ax, title in zip([ax1, ax2], ["RA Analysis", "DEC Analysis"]):
                ax.set_title(f"{mode}: {title}")
                ax.set_xlabel("Measured Pixel Shift (arcsec)")
                ax.set_ylabel(f"{mode} Header Shift (arcsec)")
                
                # Dynamic axis scaling with 1:1 line
                all_data = np.concatenate([ax.get_xlim(), ax.get_ylim()])
                vmin, vmax = np.min(all_data), np.max(all_data)
                ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.3, label='1:1 Match')
                ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
                ax.set_aspect('equal')
                ax.grid(True)

            ax1.legend()
            plt.tight_layout()
            plt.savefig(folder / f"dither_analysis_{mode.lower()}.png")
            plt.show()

        for i in range(len(selected)):
            frames = [Image.open(cutout_dir / f"{s}_star{i+1}.png") for s in valid_stems]
            frames[0].save(folder / f"star_{i+1}_track.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)

if __name__ == "__main__":
    load_config()
    parser = argparse.ArgumentParser(); parser.add_argument("folder"); run_analysis(parser.parse_args().folder)