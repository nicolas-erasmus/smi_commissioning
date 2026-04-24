#!/usr/bin/env python3
"""
Unified Dither Tracking & In-Place Astrometric Calibration.
Updates headers, generates GIFs, and saves comparison plots (TEL and WCS).
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
        raise FileNotFoundError("Missing 'key.json'.")
    with open(key_file, "r") as f:
        config = json.load(f)
    ast.api_key = config["astro_api_key"]

def is_wcs_valid(header):
    ctype1 = header.get('CTYPE1', '').upper().strip()
    if not ctype1 or 'PIXEL' in ctype1:
        return False
    celestial_types = ['RA-', 'DEC-', 'GLON', 'GLAT', 'ELON', 'ELAT']
    return any(c_type in ctype1 for c_type in celestial_types)

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
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None or is_wcs_valid(hdu.header):
            return True
        img = hdu.data.astype(float)
        ny, nx = img.shape

    x, y = get_sources(img)
    if x is None: return False

    print(f"Solving {file_path.name}...")
    try:
        wcs_header = ast.solve_from_source_list(x, y, nx, ny, solve_timeout=60)
        if wcs_header:
            with fits.open(file_path, mode='update') as hdul:
                target_hdu = next(h for h in hdul if h.data is not None)
                for k in ['WCSAXES', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                    if k in target_hdu.header: del target_hdu.header[k]
                target_hdu.header.update(wcs_header)
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

def save_cutout(image_data, pos, frame_name, star_idx, output_dir, box_size=15):
    output_dir.mkdir(exist_ok=True, parents=True)
    x_img, y_img = pos
    ix, iy = int(round(x_img)), int(round(y_img))
    y_min, y_max = max(0, iy - box_size), min(image_data.shape[0], iy + box_size + 1)
    x_min, x_max = max(0, ix - box_size), min(image_data.shape[1], ix + box_size + 1)
    cutout = image_data[y_min:y_max, x_min:x_max]
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cutout, cmap='gray', origin='lower', vmin=np.percentile(cutout, 5), vmax=np.percentile(cutout, 99))
    ax.plot(x_img - x_min, y_img - y_min, 'rx')
    ax.axis('off')
    plt.savefig(output_dir / f"{frame_name}_star{star_idx+1}.png", bbox_inches='tight')
    plt.close(fig)

def run_analysis(folder):
    folder = Path(folder)
    cutout_dir = folder / "cutouts"
    files = sorted(list(folder.glob('*.fit*')))
    
    print("Step 1: Astrometric Solve (In-place)...")
    for f in files: solve_and_update(f)

    # Find first valid solved file for reference
    ref_img = None
    for f in files:
        with fits.open(f) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu and is_wcs_valid(hdu.header):
                ref_img = hdu.data.astype(float)
                break
    if ref_img is None:
        print("Error: No files have a valid WCS solution.")
        return

    sx, sy = get_sources(ref_img, limit=200, thresh=5.0)
    fig, ax = plt.subplots(); ax.imshow(ref_img, cmap='gray', origin='lower', vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 95))
    ax.set_title("Select stars, then press ENTER")
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

    print("Step 2: Tracking stars and collecting coordinates...")
    for f in files:
        with fits.open(f) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu and is_wcs_valid(hdu.header):
                w = WCS(hdu.header); ny, nx = hdu.data.shape
                wra, wdec = w.all_pix2world(nx/2, ny/2, 0)
                
                # Try multiple keys for telescope coords
                tra = to_deg(hdu.header.get('TELRA', hdu.header.get('RA')), True)
                tdec = to_deg(hdu.header.get('TELDEC', hdu.header.get('DEC')), False)
                
                cx, cy = get_sources(hdu.data.astype(float), limit=200, thresh=5.0)
                curr_p = []
                for rx, ry in last_p:
                    d = np.sqrt((cx-rx)**2 + (cy-ry)**2); idx = np.argmin(d)
                    curr_p.append((cx[idx], cy[idx]) if d[idx] < 20 else None)
                
                if all(p is not None for p in curr_p):
                    valid_stems.append(f.stem)
                    wcs_centers.append((wra, wdec))
                    tel_coords.append((tra, tdec))
                    pixel_positions.append(curr_p)
                    for i, p in enumerate(curr_p): 
                        save_cutout(hdu.data, p, f.stem, i, cutout_dir)
                    last_p = curr_p

    # Step 3: Plotting and GIF Creation
    if len(valid_stems) > 1:
        for mode in ['TEL', 'WCS']:
            # Safety: Check if we have valid coordinates for the entire list
            data_list = tel_coords if mode == 'TEL' else wcs_centers
            if any(c[0] is None or c[1] is None for c in data_list):
                print(f"Skipping {mode} plot: Header missing RA/Dec values.")
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            r0, d0 = data_list[0]
            
            for i in range(len(selected)):
                dx = np.array([p[i][0] - pixel_positions[0][i][0] for p in pixel_positions]) * PLATE_SCALE
                dy = np.array([p[i][1] - pixel_positions[0][i][1] for p in pixel_positions]) * PLATE_SCALE
                dr = np.array([(c[0]-r0)*3600*np.cos(np.radians(d0)) for c in data_list])
                dd = np.array([(c[1]-d0)*3600 for c in data_list])
                ax1.plot(dx, dr, 'o-', label=f"Star {i+1}"); ax2.plot(dy, dd, 'o-')

            ax1.set_xlabel("Measured dX (arcsec)"); ax1.set_ylabel(f"{mode} dRA (arcsec)")
            ax2.set_xlabel("Measured dY (arcsec)"); ax2.set_ylabel(f"{mode} dDec (arcsec)")
            fig.suptitle(f"Dither Analysis: {mode} Coordinates")
            ax1.legend(); ax1.grid(True); ax2.grid(True)
            
            save_path = folder / f"dither_analysis_{mode.lower()}.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")
            plt.show()

        print("Generating GIFs...")
        for i in range(len(selected)):
            frames = [Image.open(cutout_dir / f"{s}_star{i+1}.png") for s in valid_stems]
            gif_path = folder / f"star_{i+1}_track.gif"
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
            print(f"Saved GIF: {gif_path}")

if __name__ == "__main__":
    load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    run_analysis(parser.parse_args().folder)