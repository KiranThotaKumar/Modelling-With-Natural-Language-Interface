import numpy as np


def smooth_counts(intensities, window=5):
    kernel = np.ones(window) / window
    return np.convolve(intensities, kernel, mode="same")

def estimate_fwhm(energies, intensities, peak_index):
    from scipy.interpolate import interp1d

    peak_x = energies[peak_index]
    peak_y = intensities[peak_index]

    half_max = peak_y / 2.0

    # Find nearest points on left where intensity crosses half max
    left_mask = energies < peak_x
    right_mask = energies > peak_x

    if not np.any(left_mask) or not np.any(right_mask):
        return np.nan

    f_left = interp1d(intensities[left_mask], energies[left_mask], bounds_error=False)
    f_right = interp1d(intensities[right_mask], energies[right_mask], bounds_error=False)

    try:
        e_left = float(f_left(half_max))
        e_right = float(f_right(half_max))
        return abs(e_right - e_left)
    except:
        return np.nan

def cluster_peaks(peak_energies, peak_heights, cluster_width):
    # Ensure sorted by energy
    idx = np.argsort(peak_energies)
    peak_energies = peak_energies[idx]
    peak_heights = peak_heights[idx]

    clusters = []

    current_cluster = {
        "energies": [peak_energies[0]],
        "heights": [peak_heights[0]],
    }

    for E, H in zip(peak_energies[1:], peak_heights[1:]):
        E_ref = np.mean(current_cluster["energies"])

        if abs(E - E_ref) < cluster_width:
            current_cluster["energies"].append(E)
            current_cluster["heights"].append(H)
        else:
            clusters.append(current_cluster)
            current_cluster = {
                "energies": [E],
                "heights": [H],
            }

    clusters.append(current_cluster)
    return clusters


def reduce_clusters(clusters):
    line_energies = []
    line_amplitudes = []

    for c in clusters:
        line_energies.append(np.mean(c["energies"]))
        line_amplitudes.append(np.sum(c["heights"]))

    return np.array(line_energies), np.array(line_amplitudes)