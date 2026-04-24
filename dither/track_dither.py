#!/usr/bin/env python3
"""
Interactive star tracking + thumbnail cutouts + RA/Dec tracking.
Features: 'sep' centering, platescale conversion, multi-star plots, and GIF creation.
"""

import numpy as np
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image

# Global Settings
PLATE_SCALE = 0.1267  # arcsec/px

def get_sep_objects(image_data):
    """Detects sources using 'sep' with winpos sub-pixel refinement."""
    # Ensure native byte order for sep
    data = image_data.byteswap().newbyteorder() if image_data.dtype.byteorder != '=' else image_data
    data = data.astype(np.float32)

    try:
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        bkg_rms = bkg.globalrms
    except:
        data_sub = data - np.median(data)
        bkg_rms = np.std(data)

    objects = sep.extract(data_sub, 5.0, err=bkg_rms)
    
    if len(objects) > 0:
        # winpos provides highly accurate centroids
        x, y, flags = sep.winpos(data_sub, objects['x'], objects['y'], 1.0)
        return x, y
    return None, None

def extract_telescope_coords(hdul):
    """Parses 'HH:MM:SS.SS' and 'DD:MM:SS.SS' into decimal degrees."""
    ra_keys = ['TELRA', 'RA', 'OBJCTRA']
    dec_keys = ['TELDEC', 'DEC', 'OBJCTDEC']
    
    ra_raw, dec_raw = None, None
    for hdu in hdul:
        for k in ra_keys:
            if k in hdu.header: ra_raw = hdu.header[k]; break
        for k in dec_keys:
            if k in hdu.header: dec_raw = hdu.header[k]; break
        if ra_raw is not None and dec_raw is not None: break
            
    if ra_raw is None or dec_raw is None: return None, None

    def to_deg(val, is_ra=True):
        if isinstance(val, str) and ':' in val:
            parts = val.strip().split(':')
            sign = -1.0 if '-' in parts[0] else 1.0
            hms = [abs(float(p)) for p in parts]
            deg = sign * (hms[0] + hms[1]/60.0 + hms[2]/3600.0)
            return deg * 15.0 if is_ra else deg
        return float(val) * 15.0 if (is_ra and float(val) < 24) else float(val)

    return to_deg(ra_raw, True), to_deg(dec_raw, False)

def save_cutouts(image_data, positions, frame_name, output_dir, box_size=15):
    """Saves sub-pixel accurate thumbnails."""
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

def create_tracking_gifs(output_dir, num_stars, frame_stems):
    """Combines PNG cutouts into GIFs."""
    output_dir = Path(output_dir)
    for i in range(num_stars):
        star_id = i + 1
        frames = []
        for stem in frame_stems:
            img_path = output_dir / f"{stem}_star{star_id}.png"
            if img_path.exists():
                frames.append(Image.open(img_path))
        
        if frames:
            gif_path = output_dir.parent / f"star_{star_id}_tracking.gif"
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                           duration=200, loop=0)
            print(f"GIF created: {gif_path}")

def process(folder, search_radius=25):
    folder = Path(folder)
    files = sorted(folder.glob('*.fits')) + sorted(folder.glob('*.fit'))
    if not files: return

    with fits.open(files[0]) as hdul:
        img_hdu = next((h for h in hdul if h.data is not None), None)
        ref_img = img_hdu.data.astype(float)
        ra0, dec0 = extract_telescope_coords(hdul)

    sx_list, sy_list = get_sep_objects(ref_img)
    if sx_list is None: return

    # Interactive Snapping
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(ref_img, cmap='gray', origin='lower', vmin=np.percentile(ref_img, 1), vmax=np.percentile(ref_img, 99))
    ax.set_title("Select stars, then press ENTER.")
    
    selected_coords = []
    def onclick(event):
        if event.inaxes != ax: return
        idx = np.argmin(np.sqrt((sx_list - event.xdata)**2 + (sy_list - event.ydata)**2))
        selected_coords.append((sx_list[idx], sy_list[idx]))
        ax.plot(sx_list[idx], sy_list[idx], 'ro', mfc='none', markersize=10)
        ax.text(sx_list[idx], sy_list[idx] + 5, f"{len(selected_coords)}", color='red', fontweight='bold')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', lambda e: plt.close(fig) if e.key=='enter' else None)
    plt.show()

    if not selected_coords: return

    all_positions = [selected_coords]
    ra_list, dec_list = [ra0], [dec0]
    frame_stems = [files[0].stem]
    save_cutouts(ref_img, selected_coords, files[0].stem, folder / "cutouts")

    for f in files[1:]:
        with fits.open(f) as hdul:
            img_hdu = next((h for h in hdul if h.data is not None), None)
            if img_hdu is None: continue
            img = img_hdu.data.astype(float)
            ra, dec = extract_telescope_coords(hdul)
            cx, cy = get_sep_objects(img)
            
            if cx is not None:
                current_stars = []
                for rx, ry in all_positions[-1]:
                    d = np.sqrt((cx - rx)**2 + (cy - ry)**2)
                    idx = np.argmin(d)
                    current_stars.append((cx[idx], cy[idx]) if d[idx] < search_radius else None)
                
                if all(p is not None for p in current_stars):
                    all_positions.append(current_stars)
                    ra_list.append(ra); dec_list.append(dec)
                    frame_stems.append(f.stem)
                    save_cutouts(img, current_stars, f.stem, folder / "cutouts")

    # Plotting and GIFs
    if len(all_positions) > 1 and ra0 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        dra_req = np.array([(r - ra0) * 3600.0 * np.cos(np.radians(dec0)) for r in ra_list])
        ddec_req = np.array([(d - dec0) * 3600.0 for d in dec_list])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_coords)))
        for i in range(len(selected_coords)):
            dx_arc = np.array([p[i][0] - all_positions[0][i][0] for p in all_positions]) * PLATE_SCALE
            dy_arc = np.array([p[i][1] - all_positions[0][i][1] for p in all_positions]) * PLATE_SCALE
            ax1.plot(dx_arc, dra_req, 'o-', color=colors[i], label=f"Star {i+1}")
            ax2.plot(dy_arc, ddec_req, 'o-', color=colors[i], label=f"Star {i+1}")

        ax1.set_xlabel("Measured Position (arcseconds)"); ax1.set_ylabel("Requested Position (arcseconds)")
        ax2.set_xlabel("Measured Position (arcseconds)"); ax2.set_ylabel("Requested Position (arcseconds)")
        ax1.legend(); ax2.legend(); plt.tight_layout()
        plt.savefig(folder / "dither_analysis.png", dpi=150)
        plt.show()

        create_tracking_gifs(folder / "cutouts", len(selected_coords), frame_stems)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    process(args.folder)