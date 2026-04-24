import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def get_centroid(profile, peak_index, window=3):
    """Calculates the center of mass around a peak for sub-pixel accuracy."""
    y = np.arange(peak_index - window, peak_index + window + 1)
    y = y[(y >= 0) & (y < len(profile))]
    weights = profile[y]
    if np.sum(weights) == 0:
        return peak_index
    return np.sum(y * weights) / np.sum(weights)

def analyze_ifu_traces(fits_path, extension=None, n_steps=15, poly_order=3):
    try:
        with fits.open(fits_path) as hdul:
            data = None
            if extension is not None:
                data = hdul[extension].data
            else:
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and isinstance(hdu.data, np.ndarray):
                        if hdu.data.ndim == 2:
                            data = hdu.data
                            extension = i
                            break
            if data is None:
                print("Error: No 2D image data found.")
                return
    except Exception as e:
        print(f"Error: {e}")
        return

    ny, nx = data.shape
    
    # 1. Get reference peaks from the center
    center_x = nx // 2
    profile_mid = np.median(data[:, center_x-2 : center_x+3], axis=1)
    profile_mid_sm = gaussian_filter1d(profile_mid, sigma=1.0)
    
    ref_peaks, _ = find_peaks(
        profile_mid_sm, 
        distance=10, 
        width=1.5, 
        prominence=np.nanpercentile(profile_mid_sm, 90) * 0.05
    )
    
    n_traces = len(ref_peaks)
    print(f"Detected {n_traces} reference traces.")

    # 2. Sample at specified X-locations
    x_locs = np.linspace(20, nx - 20, n_steps, dtype=int)
    trace_coords = {i: {"x": [], "y": []} for i in range(n_traces)}

    print(f"Tracing across {n_steps} X-locations...")
    for x in x_locs:
        col_prof = np.median(data[:, max(0, x-2) : min(nx, x+3)], axis=1)
        col_prof_sm = gaussian_filter1d(col_prof, sigma=1.0)
        
        peaks, _ = find_peaks(
            col_prof_sm, 
            distance=10, 
            width=1.5, 
            prominence=np.nanpercentile(col_prof_sm, 90) * 0.05
        )
        
        for i, ref_y in enumerate(ref_peaks):
            if len(peaks) > 0:
                diffs = np.abs(peaks - ref_y)
                closest_idx = np.argmin(diffs)
                
                # Match within 15 pixels to accommodate larger smile at edges
                if diffs[closest_idx] < 15:
                    sub_y = get_centroid(col_prof, peaks[closest_idx], window=2)
                    trace_coords[i]["x"].append(x)
                    trace_coords[i]["y"].append(sub_y)

    # 3. Fit and Visualize with Points
    plt.figure(figsize=(14, 10))
    plt.imshow(data, origin='lower', cmap='gray', aspect='auto', 
               vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
    
    plot_x = np.linspace(0, nx, nx)
    all_coeffs = []

    for i in range(n_traces):
        xs = np.array(trace_coords[i]["x"])
        ys = np.array(trace_coords[i]["y"])
        
        if len(xs) > 3: # Need at least 4 points for a cubic fit
            # Plot the raw detected points (Cyan dots)
            plt.scatter(xs, ys, color='cyan', s=5, alpha=0.8, zorder=3)
            
            # Fit and plot the polynomial (Red line)
            coeffs = np.polyfit(xs, ys, poly_order)
            all_coeffs.append(coeffs)
            
            fit_y = np.polyval(coeffs, plot_x)
            plt.plot(plot_x, fit_y, color='red', lw=0.6, alpha=0.6, zorder=4)

    plt.title(f"IFU Tracing: Polynomial fits (red) and detected points (cyan)\n{n_steps} steps, Order {poly_order}")
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.xlim(0, nx)
    plt.ylim(0, ny)
    plt.tight_layout()
    plt.show()
    
    return all_coeffs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFU Trace Fitting with Point Visualization.")
    parser.add_argument("file", help="Path to the FITS file")
    parser.add_argument("--ext", type=int, default=None, help="FITS extension index")
    parser.add_argument("--steps", type=int, default=15, help="Number of points along X")
    parser.add_argument("--order", type=int, default=3, help="Polynomial order")
    
    args = parser.parse_args()
    analyze_ifu_traces(args.file, args.ext, n_steps=args.steps, poly_order=args.order)