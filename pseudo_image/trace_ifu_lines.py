import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def get_centroid(profile, peak_index, window=3):
    """Calculates sub-pixel peak position using center of mass."""
    y = np.arange(peak_index - window, peak_index + window + 1)
    y = y[(y >= 0) & (y < len(profile))]
    weights = profile[y]
    # Background subtraction to sharpen the centroid
    weights = np.maximum(weights - np.nanmin(weights), 0)
    if np.sum(weights) == 0:
        return peak_index
    return np.sum(y * weights) / np.sum(weights)

def analyze_ifu_traces(fits_path, extension=1, n_steps=30, poly_order=3, expected_traces=244):
    try:
        with fits.open(fits_path) as hdul:
            if extension >= len(hdul):
                print(f"Error: Extension {extension} not found.")
                return
            data = hdul[extension].data.astype(float)
            print(f"Loaded data from extension {extension} (Shape: {data.shape})")
    except Exception as e:
        print(f"Error: {e}")
        return

    ny, nx = data.shape
    
    # 1. Establish reference peaks at the center
    center_x = nx // 2
    profile_mid = np.median(data[:, center_x-10 : center_x+10], axis=1)
    profile_mid_sm = gaussian_filter1d(profile_mid, sigma=1.2)
    
    ref_peaks, _ = find_peaks(
        profile_mid_sm, 
        distance=4, 
        prominence=np.nanpercentile(profile_mid_sm, 5) # Lowered for faint traces
    )
    
    ref_peaks = np.sort(ref_peaks)
    n_detected = len(ref_peaks)
    print(f"Detected {n_detected} reference traces at X={center_x}.")

    # 2. Sequential Tracing
    trace_data = {i: {"x": [], "y": []} for i in range(n_detected)}
    x_locs = np.linspace(50, nx - 50, n_steps, dtype=int)
    
    for direction in [1, -1]: 
        current_positions = np.copy(ref_peaks).astype(float)
        relevant_x = x_locs[x_locs > center_x] if direction == 1 else x_locs[x_locs < center_x][::-1]
        
        for x in relevant_x:
            col_prof = np.median(data[:, x-2 : x+3], axis=1)
            col_prof_sm = gaussian_filter1d(col_prof, sigma=1.0)
            peaks, _ = find_peaks(col_prof_sm, distance=4, prominence=np.nanpercentile(col_prof_sm, 5))
            
            if len(peaks) == 0: continue
                
            for i in range(n_detected):
                diffs = np.abs(peaks - current_positions[i])
                closest_idx = np.argmin(diffs)
                
                if diffs[closest_idx] < 8:
                    sub_y = get_centroid(col_prof, peaks[closest_idx], window=2)
                    trace_data[i]["x"].append(x)
                    trace_data[i]["y"].append(sub_y)
                    current_positions[i] = sub_y

    # 3. Fitting and Adjusted Visualization
    # Reduced plot size by 30% (original was roughly 12x7, now ~8.4x4.9)
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    
    # Apply Log Scale for the image display
    # We clip the data to avoid non-positive values for LogNorm
    vmin = np.nanpercentile(data[data > 0], 5)
    vmax = np.nanpercentile(data, 99)
    im = ax.imshow(data, origin='lower', cmap='magma', aspect='auto', 
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    
    all_coeffs = []
    plot_x = np.arange(nx)

    for i in range(n_detected):
        xs = np.array(trace_data[i]["x"])
        ys = np.array(trace_data[i]["y"])
        
        if len(xs) > poly_order + 2:
            # Reintroduce raw data points (Cyan)
            ax.scatter(xs, ys, color='cyan', s=1.5, alpha=0.4, zorder=3)
            
            coeffs = np.polyfit(xs, ys, poly_order)
            all_coeffs.append(coeffs)
            
            # Plot the Polynomial fit (Red)
            ax.plot(plot_x, np.polyval(coeffs, plot_x), color='red', lw=0.5, alpha=0.6, zorder=4)

    ax.set_title(f"IFU Tracing: {len(all_coeffs)} Fits | Log Scale")
    ax.set_xlabel("Dispersion (Pixels)")
    ax.set_ylabel("Spatial (Pixels)")
    plt.tight_layout()
    plt.show()

    return all_coeffs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--ext", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--order", type=int, default=3)
    args = parser.parse_args()
    analyze_ifu_traces(args.file, args.ext, n_steps=args.steps, poly_order=args.order)