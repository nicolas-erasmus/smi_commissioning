#!/usr/bin/env python3
"""
Interactive star tracking + thumbnail cutouts + RA/Dec tracking
Uses 'sep' for high-precision centering and platescale for arcsec conversion.
Plots comparison between requested (telescope) and measured (image) offsets.
"""

import numpy as np
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 1. Platescale (arcsec/px)
PLATE_SCALE = 0.1267

def get_sep_objects(image_data):
    """
    Detects sources using the 'sep' library with sub-pixel refinement.
    """
    # Ensure byte order is correct for sep (native)
    data = image_data.byteswap().newbyteorder() if image_data.dtype.byteorder != '=' else image_data
    data = data.astype(np.float32)

    # Background subtraction
    try:
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
    except Exception:
        # Fallback if background estimation fails on small/noisy images
        data_sub = data - np.median(data)
        bkg_rms = np.std(data)
    else:
        bkg_rms = bkg.globalrms

    # Extraction (threshold 5*sigma)
    objects = sep.extract(data_sub, 5.0, err=bkg_rms)
    
    # Refine centroids using winpos for higher accuracy (sub-pixel)
    if len(objects) > 0:
        # sigma=1.0 is typical for stellar PSFs
        x, y, flags = sep.winpos(data_sub, objects['x'], objects['y'], 1.0)
        return x, y
    return None, None

def extract_telescope_coords(hdul):
    """
    Parses 'HH:MM:SS.SS' and 'DD:MM:SS.SS' string formats into decimal degrees.
    """
    ra_keys = ['TELRA', 'RA', 'OBJCTRA']
    dec_keys = ['TELDEC', 'DEC', 'OBJCTDEC']
    
    ra_raw, dec_raw = None, None
    for hdu in hdul:
        for k in ra_keys:
            if k in hdu.header:
                ra_raw = hdu.header[k]
                break
        for k in dec_keys:
            if k in hdu.header:
                dec_raw = hdu.header[k]
                break
        if ra_raw is not None and dec_raw is not None:
            break
            
    if ra_raw is None or dec_raw is None:
        return None, None

    # Parse RA
    try:
        if isinstance(ra_raw, str) and ':' in ra_raw:
            h, m, s = map(float, ra_raw.strip().split(':'))
            ra_deg = (h + m/60.0 + s/3600.0) * 15.0
        else:
            ra_deg = float(ra_raw)
            if ra_deg < 24: ra_deg *= 15.0
    except: ra_deg = None

    # Parse Dec
    try:
        if isinstance(dec_raw, str) and ':' in dec_raw:
            parts = dec_raw.strip().split(':')
            sign = -1.0 if '-' in parts[0] else 1.0
            d, m, s = map(abs, map(float, parts))
            dec_deg = sign * (d + m/60.0 + s/3600.0)
        else:
            dec_deg = float(dec_raw)
    except: dec_deg = None

    return ra_deg, dec_deg

def save_cutouts(image_data, positions, frame_name, output_dir, box_size=15):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, pos in enumerate(positions):
        if pos is None: continue
        x_img, y_img = pos
        ix, iy = int(round(x_img)), int(round(y_img))
        
        y_min, y_max = iy - box_size, iy + box_size + 1
        x_min, x_max = ix - box_size, ix + box_size + 1
        
        if y_min < 0 or x_min < 0 or y_max > image_data.shape[0] or x_max > image_data.shape[1]:
            continue
            
        cutout = image_data[y_min:y_max, x_min:x_max]
        mx, my = x_img - x_min, y_img - y_min
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cutout, cmap='gray', origin='lower', 
                  vmin=np.percentile(cutout, 5), vmax=np.percentile(cutout, 99))
        ax.plot(mx, my, 'rx', markersize=10)
        ax.set_title(f"Star {i+1} | {frame_name}")
        ax.axis('off')
        plt.savefig(output_dir / f"{frame_name}_star{i+1}.png", dpi=80, bbox_inches='tight')
        plt.close(fig)

def process(folder, search_radius=25):
    folder = Path(folder)
    files = sorted(folder.glob('*.fits')) + sorted(folder.glob('*.fit'))
    if not files:
        print("No FITS files found.")
        return

    with fits.open(files[0]) as hdul:
        img_hdu = next((h for h in hdul if h.data is not None), None)
        ref_img = img_hdu.data.astype(float)
        ra0, dec0 = extract_telescope_coords(hdul)

    # Detect sources in reference frame
    sx_list, sy_list = get_sep_objects(ref_img)
    if sx_list is None: 
        print("No sources detected with SEP.")
        return

    # Interactive selection (Snapping to SEP centroids)
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin, vmax = np.percentile(ref_img, [1, 99.5])
    ax.imshow(ref_img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title("Click stars to track. Press ENTER when done.")
    
    selected_coords = []
    def onclick(event):
        if event.inaxes != ax: return
        dist = np.sqrt((sx_list - event.xdata)**2 + (sy_list - event.ydata)**2)
        idx = np.argmin(dist)
        sx, sy = sx_list[idx], sy_list[idx]
        selected_coords.append((sx, sy))
        ax.plot(sx, sy, 'ro', mfc='none', markersize=10)
        ax.text(sx, sy + 5, f"{len(selected_coords)}", color='red', fontweight='bold')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close(fig) if e.key=='enter' else None)
    plt.show()

    if not selected_coords: return

    # Track through sequence
    all_positions = [selected_coords]
    ra_list, dec_list = [ra0], [dec0]
    save_cutouts(ref_img, selected_coords, files[0].stem, folder / "cutouts")

    print(f"Tracking {len(selected_coords)} star(s) across {len(files)} frames...")

    for f in files[1:]:
        with fits.open(f) as hdul:
            img_hdu = next((h for h in hdul if h.data is not None), None)
            if img_hdu is None: continue
            img = img_hdu.data.astype(float)
            ra, dec = extract_telescope_coords(hdul)
            
            cx_list, cy_list = get_sep_objects(img)
            
            if cx_list is not None:
                current_frame_stars = []
                for rx, ry in all_positions[-1]:
                    d = np.sqrt((cx_list - rx)**2 + (cy_list - ry)**2)
                    idx = np.argmin(d)
                    if d[idx] < search_radius:
                        current_frame_stars.append((cx_list[idx], cy_list[idx]))
                    else:
                        current_frame_stars.append(None)
                
                if all(p is not None for p in current_frame_stars):
                    all_positions.append(current_frame_stars)
                    ra_list.append(ra)
                    dec_list.append(dec)
                    save_cutouts(img, current_frame_stars, f.stem, folder / "cutouts")
                else:
                    print(f"Lost tracking in {f.name}")

    # 5. Final Plots
    if len(all_positions) > 1 and ra0 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate requested shifts (common for all stars in a frame)
        dra_req = np.array([(r - ra0) * 3600.0 * np.cos(np.radians(dec0)) for r in ra_list])
        ddec_req = np.array([(d - dec0) * 3600.0 for d in dec_list])

        # Loop through each selected star
        num_stars = len(selected_coords)
        colors = plt.cm.jet(np.linspace(0, 1, num_stars))

        for i in range(num_stars):
            # Calculate measured shifts for this specific star
            dx_arcsec = np.array([p[i][0] - all_positions[0][i][0] for p in all_positions]) * PLATE_SCALE
            dy_arcsec = np.array([p[i][1] - all_positions[0][i][1] for p in all_positions]) * PLATE_SCALE
            
            lbl = f"Star {i+1}"
            ax1.plot(dx_arcsec, dra_req, 'o-', color=colors[i], label=lbl, markersize=4, alpha=0.7)
            ax2.plot(dy_arcsec, ddec_req, 'o-', color=colors[i], label=lbl, markersize=4, alpha=0.7)

        # Plot Styling
        ax1.set_xlabel("Measured Position (arcseconds)")
        ax1.set_ylabel("Requested Position (arcseconds)")
        ax1.set_title("RA Offset: Measured ($\Delta X$) vs Requested ($\Delta RA$)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        ax2.set_xlabel("Measured Position (arcseconds)")
        ax2.set_ylabel("Requested Position (arcseconds)")
        ax2.set_title("Dec Offset: Measured ($\Delta Y$) vs Requested ($\Delta Dec$)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        
        # Save the final figure to disk
        plot_path = folder / "dither_tracking_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Final plot saved to {plot_path}")
        plt.show()
    else:
        print("Insufficient data for final plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing FITS files")
    parser.add_argument("--search-radius", type=float, default=25, help="Search radius (px)")
    args = parser.parse_args()
    process(args.folder, args.search_radius)