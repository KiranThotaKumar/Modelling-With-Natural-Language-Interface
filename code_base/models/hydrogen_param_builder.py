from models.hydrogen_model import HydrogenData, HydrogenModelConfig
import logging
from scipy.signal import find_peaks
from signal_analysis.peak_finding import smooth_counts, estimate_fwhm, cluster_peaks, reduce_clusters
from physics.hydrogen import match_lines_to_hydrogen_transitions, hydrogen_transition_energies
import numpy as np

def build_hydrogen_params_from_measured(data: HydrogenData):

    energies = data.energies
    intensities = data.counts

    import matplotlib
    #matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    plt.plot(intensities)
    plt.grid(True)
    plt.show()
    plt.close()
    # Step 1: Smoothing (choose one)
    # A) Simple moving average
    intens_smooth = smooth_counts(intensities, window=7)

    import matplotlib.pyplot as plt1
    plt1.plot(intens_smooth)
    plt1.grid(True)
    plt1.show()
    plt1.close()
    # Step 2. Peak detection

    scale = np.max(intens_smooth) - np.median(intens_smooth)
    #prom = 0.055 * scale
    prom = 0.05 * scale
    peaks, props = find_peaks(intens_smooth, prominence=prom)
    raw_energies_peaks = energies[peaks]
    print("Raw peak energies:", raw_energies_peaks)
    idx = np.argsort(props["prominences"])[::-1]

    # keep top N physically plausible peaks
    Nmax = 3
    idx = idx[:Nmax]

    peaks = peaks[idx]
    peak_energies = energies[peaks]

    #peak_indices, props = find_peaks(intens_smooth, height=np.max(intens_smooth)*0.05)
    #peak_energies = energies[peak_indices]
    #peak_heights = props["peak_heights"]
    peak_heights = intens_smooth[peaks]

    # Sort by energy (optional)
    order = np.argsort(peak_energies)
    nlines = peak_energies.size
    peak_energies = peak_energies[order][:nlines]
    peak_heights = peak_heights[order][:nlines]
    print("Peak Energies", peak_energies)
    print("Peak Heights", peak_heights)
    for E, H in zip(peak_energies, peak_heights):
        print(f"Peak at {E:.3f} eV, height {H:.2f}")
    # Step 3. Estimate sigma_instr via FWHM
    fwhm_list = []
    for idx in peaks[:nlines]:
        fwhm = estimate_fwhm(energies, intens_smooth, idx)
        if not np.isnan(fwhm):
            fwhm_list.append(fwhm)

    if len(fwhm_list) == 0:
        sigma_instr = 0.2  # fallback
    else:
        sigma_instr = np.mean(fwhm_list) / 2.355  # convert FWHM to sigma

    # Step 4. Sort peaks by energy, cluster_peaks and reduce_clusters
    order = np.argsort(peak_energies)
    peak_energies = peak_energies[order]
    peak_heights = peak_heights[order]
    
    dE = np.median(np.diff(energies)) # energy grid spacing 
    m = 4 # m ≈ 3–5
    cluster_width = m * dE 

    clusters = cluster_peaks(peak_energies, peak_heights, cluster_width)
    line_energies, line_amplitudes = reduce_clusters(clusters)
    for i, c in enumerate(clusters):
        print(i, np.mean(c["energies"]), np.sum(c["heights"]))

    transitions, dE = match_lines_to_hydrogen_transitions(line_energies, n_max=7)

    print("Matched transitions:", transitions)
    print("Energy residuals (eV):", dE)

    # Step 5. Background estimate
    background = np.quantile(intens_smooth, 0.05)

    # Step 6. Relative amplitudes
    raw = np.maximum(peak_heights, 1e-12)
    rels = raw / np.sum(raw)
    log_rels = np.log(rels)

    # Step 7. Global scale
    scale = np.max(intens_smooth)

    # Parameter vector in same order as build_default_theta:
    # [sigma_instr, background, scale, log_rel_1, log_rel_2, ...]
    theta0 = np.array([sigma_instr, background, scale] + list(log_rels))

    model_config = HydrogenModelConfig(transitions, line_energies, line_amplitudes)
    
    return theta0, model_config