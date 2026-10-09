import numpy as np
import matplotlib.pyplot as plt

def plot_compare_spectra(
    measured,
    modeled,
    energies,
    labels=("Measured", "Modeled"),
):
    """
    Thin wrapper that builds plot_specs and calls plot_curves().
    """
    plt.figure(figsize=(6,4))
    x_axis = energies
    plt.plot(energies, measured, label=labels[0], lw=2)
    plt.plot(energies, modeled, label=labels[1], lw=2, linestyle='--')
    plt.title(f"Spectrum Comparison")
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    fig = plt.gcf() # Get the figure *after* setting elements

    plt.show()
    fig.savefig('spectrum_Comparison_plot.png')
    plt.close(fig)
