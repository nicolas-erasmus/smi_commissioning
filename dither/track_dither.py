#!/usr/bin/env python3
"""
Unified Dither Tracking & In-Place Astrometric Calibration.
Updates original FITS headers, skips 'dummy' WCS, and plots results.
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

# --- Settings ---
PLATE_SCALE = 0.1267  # arcsec/px (Adjust for your instrument)
ast = AstrometryNet()

def load_config():
    """Loads the API key from key.json."""
    key_file = Path("key.json")
    if not key_file.exists():
        raise FileNotFoundError("Missing 'key.json'. Please create it with your 'astro_api_key'.")
    with open(key_file, "r") as f:
        config = json.load(f)
    ast.api_key = config["astro_api_key"]

def is_wcs_valid(header):
    """Checks if the WCS is real celestial or just a 'pixel' placeholder."""
    ctype1 = header.get('CTYPE1', '').upper().strip()
    if not ctype1 or 'PIXEL' in ctype1:
        return False
    # Look for standard celestial projection identifiers
    celestial_types = ['RA-', 'DEC-', 'GLON', 'GLAT', 'ELON', 'ELAT']
    return any(c_type in ctype1 for c_type in celestial_types)

def get_sources(data, limit=40, thresh=10.0):
    """Detects stars for the astrometric solve using SEP."""
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
    except:
        pass
    return None, None

def solve_and_update(file_path):
    """Solves the image and updates the header IN-PLACE."""
    with fits.open(file_path) as hdul:
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None: return False
        
        if is_wcs_valid(hdu.header):
            print(f"Skipping {file_path.name}: Valid celestial WCS already exists.")
            return True
            
        img = hdu.data.astype(float)
        ny, nx = img.shape

    x, y = get_sources(img)
    if x is None or len(x) < 5:
        print(f"Failed {file_path.name}: Not enough stars found.")
        return False

    print(f"Solving {file_path.name} (Astrometry.net)...")
    try:
        wcs_header = ast.solve_from_source_list(x, y, nx, ny, solve_timeout=60)
    except Exception as e:
        print(f"Solve Error on {file_path.name}: {e}")
        return False

    if wcs_header:
        with fits.open(file_path, mode='update') as hdul:
            target_hdu = next(h for h in hdul if h.data is not None)
            # Purge all possible old/placeholder WCS keys
            bad_keys = [
                'WCSAXES', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 
                'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CD1_1', 
                'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'
            ]
            for k in bad_keys:
                if k in target_hdu.header: del target_hdu.header[k]
            
            target_hdu.header.update(wcs_header)
            target_hdu.header['HISTORY'] = 'Astrometric solution updated by Astrometry.net'
            hdul.flush()
        print(f"Success: {file_path.name} updated.")
        return True
    
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
        return float(val) * 15.0 if (is_ra and float(val) < 24) else float(val)
    except: return None

def run_analysis(folder):
    folder = Path(folder)
    files = sorted(list(folder.glob('*.fit*')))
    
    # 1. Solve phase
    for f in files:
        solve_and_update(f)
        time.sleep(0.5)

    # 2. Collection phase
    valid_files, wcs_centers, tel_coords, pixel_positions = [], [], [], []

    ref_img, ref_header = None, None
    for f in files:
        with fits.open(f) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu and is_wcs_valid(hdu.header):
                ref_img = hdu.data.astype(float)
                ref_header = hdu.header
                break
    
    if ref_img is None:
        print("\nNo solved files found. Cannot perform tracking analysis.")
        return

    sx, sy = get_sources(ref_img, limit=200, thresh=5.0)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(ref_img, cmap='gray', origin='lower', vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 95))
    ax.set_title("Select Stars for Tracking, then press ENTER")
    
    selected = []
    def onclick(event):
        if event.inaxes != ax: return
        idx = np.argmin(np.sqrt((sx - event.xdata)**2 + (sy - event.ydata)**2))
        selected.append((sx[idx], sy[idx]))
        ax.plot(sx[idx], sy[idx], 'ro', mfc='none', markersize=10)
        fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close(fig) if e.key=='enter' else None)
    plt.show()

    if not selected: return

    last_p = selected
    for f in files:
        with fits.open(f) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu and is_wcs_valid(hdu.header):
                w = WCS(hdu.header)
                ny, nx = hdu.data.shape
                wra, wdec = w.all_pix2world(nx/2, ny/2, 0)
                tra = to_deg(hdu.header.get('TELRA', hdu.header.get('RA')), True)
                tdec = to_deg(hdu.header.get('TELDEC', hdu.header.get('DEC')), False)
                
                cx, cy = get_sources(hdu.data.astype(float), limit=200, thresh=5.0)
                current_frame_p = []
                for rx, ry in last_p:
                    d = np.sqrt((cx-rx)**2 + (cy-ry)**2)
                    idx = np.argmin(d)
                    current_frame_p.append((cx[idx], cy[idx]) if d[idx] < 20 else None)
                
                if all(p is not None for p in current_frame_p):
                    valid_files.append(f.name)
                    wcs_centers.append((wra, wdec))
                    tel_coords.append((tra, tdec))
                    pixel_positions.append(current_frame_p)
                    last_p = current_frame_p

    # 3. Plotting
    if len(valid_files) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        w0_ra, w0_dec = wcs_centers[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
        
        for i in range(len(selected)):
            dx = np.array([p[i][0] - pixel_positions[0][i][0] for p in pixel_positions]) * PLATE_SCALE
            dy = np.array([p[i][1] - pixel_positions[0][i][1] for p in pixel_positions]) * PLATE_SCALE
            dwra = np.array([(c[0]-w0_ra)*3600*np.cos(np.radians(w0_dec)) for c in wcs_centers])
            dwdec = np.array([(c[1]-w0_dec)*3600 for c in wcs_centers])
            
            ax1.plot(dx, dwra, 'o-', color=colors[i], label=f"Star {i+1}")
            ax2.plot(dy, dwdec, 'o-', color=colors[i])

        ax1.set_title("RA Verification (WCS)"); ax1.set_xlabel("Measured dX (arcsec)"); ax1.set_ylabel("WCS dRA (arcsec)")
        ax2.set_title("Dec Verification (WCS)"); ax2.set_xlabel("Measured dY (arcsec)"); ax2.set_ylabel("WCS dDec (arcsec)")
        ax1.legend(); ax1.grid(True); ax2.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    run_analysis(parser.parse_args().folder)