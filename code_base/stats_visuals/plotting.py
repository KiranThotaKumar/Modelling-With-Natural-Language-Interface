
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#def plot_curves(plot_specs, show=True, multiplicator_x = 0.25, multiplicator_y = 0.5, xlim_low = -1, xlim_high = 2):
def plot_curves(plot_specs, show=True):
    """
    Generic curve plotting.

    plot_specs: list of dicts with keys:
        x, y (required)
        label (optional)
        style (optional)
        title (optional, applied once)
        xlabel, ylabel (optional, applied once)
    """
    

    if not plot_specs:
        raise ValueError("plot_specs is empty")

    plt.figure()

    for spec in plot_specs:
        x = spec["x"]
        y = spec["y"]
        label = spec.get("label", None)
        style = spec.get("style", None)
        if style is None:
            plt.plot(x, y, label=label)
        else:
            plt.plot(x, y, style, label=label)

        
        from matplotlib.ticker import MultipleLocator
        # Place a major grid line every 2 units
        # 3. Fetch the current axes instance directly without subplots
        ax = plt.gca()

        # 4. Apply MultipleLocator to change grid spacing (e.g., tick every 1.0 unit)
        multiplicator_x = spec.get("multiplicator_x", None)
        if multiplicator_x is not None:
            ax.xaxis.set_major_locator(MultipleLocator(multiplicator_x))

        multiplicator_y = spec.get("multiplicator_y", None)
        if multiplicator_y is not None:
            ax.yaxis.set_major_locator(MultipleLocator(multiplicator_y))

        # 5. Enable and customize the grid lines
        plt.grid(True, which='major', alpha=0.25)
    # Apply shared metadata from first spec
    first = plot_specs[0]
    if "title" in first:
        plt.title(first["title"])
    if "xlabel" in first:
        plt.xlabel(first["xlabel"])
    if "ylabel" in first:
        plt.ylabel(first["ylabel"])

    if any(spec.get("label") for spec in plot_specs):
        plt.legend()
    
    #Apply crop the x-axis:
    xlim_low = None
    xlim_high = None
    if "xlim_low" in first:
        xlim_low = first["xlim_low"]
    if "xlim_high" in first:
        xlim_high = first["xlim_high"]

    # If both keys exist and are not None, set the limits
    if (xlim_low is not None) and (xlim_high is not None):
        plt.xlim(xlim_low, xlim_high)

    #print(matplotlib.get_backend())
    #print(plt.isinteractive())
    if show:
        #print("Showing plot...")
        plt.show()
        #plt.close('all')
        #print("Plot closed.")
        

def plot_overlay(energies, meas, spec_map, spec_mean, labels):


    # Plot overlay 
    plt.figure(figsize=(8,4))
    plt.plot(energies, meas, label=labels[0], lw=1)
    plt.plot(energies, spec_map, label=labels[1], lw=1, linestyle='--')
    plt.plot(energies, spec_mean, label=labels[2], lw=1, linestyle=':')
    plt.legend()
    plt.xlabel('Energy (eV)')
    plt.ylabel('Excited-State Probability')
    plt.title('Measured vs Generated (MAP & mean)')
    plt.savefig('compare_overlay.png')
    plt.show()
    plt.close('all')
    return 0

def plot_residuals(energies, res):

    # residual
    plt.figure(figsize=(8,3))
    plt.plot(energies, res)
    plt.axhline(0, color='k', lw=0.6)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Residual')
    plt.title('Residual: measured - spec_map')
    plt.savefig('residual_map.png')
    plt.show()
    plt.close('all')
    return 0

def plot_posterior_band(energies, median, lower, upper, y_obs=None, yerr=None):
    
    measured_counts = y_obs
    plt.figure(figsize=(8,3))
    plt.fill_between(energies, lower, upper, alpha=0.3, label='95% CI')
    plt.plot(energies, median, label='Posterior median', linewidth=1.0)
    
    # plot measured
    measured_errors = yerr
    plt.errorbar(energies, measured_counts, yerr=measured_errors, fmt='.', label='Measured', markersize=2)
    plt.legend(); plt.xlabel('Time'); plt.ylabel('Measured Signal'); 
    plt.title("Bayesian Parameter Inference: Posterior Predictive Distribution")
    plt.savefig('PosteriorPredictions.png')    
    plt.show()
    plt.close('all')

    return 0

#plot_single_channel_overlay is a duplication of  plot_curves. One of them need to be removed
def plot_single_channel_overlay(
    times,
    y_obs,
    y_pred,
    y_alt=None,
    ax=None,
    labels=("Observed", "Model", "Alt"),
):
    """
    Plot a single time-series overlay.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(times, y_obs, "o", label=labels[0])
    ax.plot(times, y_pred, "-", label=labels[1])

    if y_alt is not None:
        ax.plot(times, y_alt, "--", label=labels[2])

    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_multi_qubit_timeseries(times, y_obs, y_map, y_mean=None):
    y_obs = np.asarray(y_obs)
    y_map = np.asarray(y_map)

    if y_obs.ndim == 1:
        y_obs = y_obs[:, None]
        y_map = y_map[:, None]
        if y_mean is not None:
            y_mean = y_mean[:, None]

    T, n_qubits = y_obs.shape

    fig, axes = plt.subplots(
        n_qubits, 1, figsize=(7, 3 * n_qubits), sharex=True
    )

    if n_qubits == 1:
        axes = [axes]

    for q, ax in enumerate(axes):
        plot_single_channel_overlay(
            times,
            y_obs[:, q],
            y_map[:, q],
            y_alt=y_mean[:, q] if y_mean is not None else None,
            ax=ax,
            labels=("Observed", "MAP", "Mean"),
        )
        ax.set_title(f"Qubit {q}")

    plt.tight_layout()
    plt.show()
    plt.close('all')

def plot_multi_qubit_residuals(times, y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    if y_obs.ndim == 1:
        y_obs = y_obs[:, None]
        y_pred = y_pred[:, None]

    T, n_qubits = y_obs.shape

    fig, axes = plt.subplots(
        n_qubits, 1, figsize=(7, 2.5 * n_qubits), sharex=True
    )

    if n_qubits == 1:
        axes = [axes]

    for q, ax in enumerate(axes):
        res = y_obs[:, q] - y_pred[:, q]
        ax.plot(times, res, "o-")
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel("Residual")
        ax.set_title(f"Qubit {q}")

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    plt.close('all')

def compute_multi_qubit_stats(y_obs, y_pred):
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    if y_obs.ndim == 1:
        y_obs = y_obs[:, None]
        y_pred = y_pred[:, None]

    T, n_qubits = y_obs.shape

    stats = {
        "per_qubit": [],
        "flattened": {}
    }

    for q in range(n_qubits):
        rmse_q = np.sqrt(np.mean((y_obs[:, q] - y_pred[:, q]) ** 2))
        corr_q = np.corrcoef(y_obs[:, q], y_pred[:, q])[0, 1]

        stats["per_qubit"].append({
            "qubit": q,
            "rmse": rmse_q,
            "correlation": corr_q,
        })

    # Secondary, global diagnostic
    y_obs_flat = y_obs.ravel()
    y_pred_flat = y_pred.ravel()

    stats["flattened"]["rmse"] = np.sqrt(
        np.mean((y_obs_flat - y_pred_flat) ** 2)
    )
    stats["flattened"]["correlation"] = np.corrcoef(
        y_obs_flat, y_pred_flat
    )[0, 1]

    return stats


def plot_multi_qubit_posterior_bands(
    times,
    all_signals,   # shape (nsamp, T, n_qubits)
    y_obs=None,
    lower_pct=2.5,
    upper_pct=97.5,
):
    all_signals = np.asarray(all_signals)
    nsamp, T, n_qubits = all_signals.shape

    median = np.median(all_signals, axis=0)
    lower = np.percentile(all_signals, lower_pct, axis=0)
    upper = np.percentile(all_signals, upper_pct, axis=0)

    fig, axes = plt.subplots(
        n_qubits, 1, figsize=(7, 3 * n_qubits), sharex=True
    )

    if n_qubits == 1:
        axes = [axes]

    for q, ax in enumerate(axes):
        ax.fill_between(
            times, lower[:, q], upper[:, q],
            alpha=0.3, label="95% posterior band"
        )
        ax.plot(times, median[:, q], "-", label="Posterior median")

        if y_obs is not None:
            ax.plot(times, y_obs[:, q], "o", label="Observed")

        ax.set_title(f"Qubit {q}")
        ax.set_ylabel("Signal")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    plt.close('all')


def compareModelWithMeasured(y_model: np.ndarray, y_obs: np.ndarray, n_qubits = 2, times: np.ndarray =None):
    
    if times is None:
        times = np.linspace(0.0, 20.0, 1001)
     # Plot Comparison (Observed vs Predicted from theta0)

    fig, axs = plt.subplots(n_qubits + 1, 1, figsize=(10, 3*(n_qubits+1)), sharex=True)

    for q in range(n_qubits):
        axs[q].plot(times, y_obs[:, q], "o", label=f"Observed Qubit {q}", alpha=0.7, markersize=3)
        axs[q].plot(times, y_model[:, q], "-", label=f"Predicted from Model Qubit {q}", lw=2)
        axs[q].set_ylabel("Signal")
        axs[q].legend()
        axs[q].set_title(f"Qubit {q} Observed vs Predicted")

    # Combined (mean) overlay
    obs_mean = np.mean(y_obs, axis=1)
    pred_mean = np.mean(y_model, axis=1)
    axs[-1].plot(times, obs_mean, "k-", label="Observed mean", lw=2)
    axs[-1].plot(times, pred_mean, "r--", label="Predicted mean", lw=2)
    axs[-1].set_xlabel("Time")
    axs[-1].set_ylabel("Signal")
    axs[-1].legend()
    axs[-1].set_title("Combined (mean) Observed vs Predicted")

    plt.tight_layout()
    plt.suptitle("Observed vs Predicted", y=1.02, fontsize=16)
    plt.savefig('Signal_Comparison.png', dpi=150)
    plt.show()
    plt.close('all')

    # 5. Calculate and print RMSE and Correlation
    rmse_per_qubit = [np.sqrt(np.mean((y_obs[:, q] - y_model[:, q])**2)) for q in range(n_qubits)]
    corr_per_qubit = [np.corrcoef(y_obs[:, q], y_model[:, q])[0,1] for q in range(n_qubits)]

    print("\nComparison (Observed vs Predicted from theta0):")
    for q in range(n_qubits):
        print(f"Qubit {q}: RMSE = {rmse_per_qubit[q]:.4f}, Correlation = {corr_per_qubit[q]:.4f}")

    rmse_combined = np.sqrt(np.mean((obs_mean - pred_mean)**2))
    corr_combined = np.corrcoef(obs_mean, pred_mean)[0,1]
    print(f"Combined signal: RMSE = {rmse_combined:.4f}, Correlation = {corr_combined:.4f}")

    # 6. Plot Residuals
    fig_res, axs_res = plt.subplots(n_qubits + 1, 1, figsize=(10, 3*(n_qubits+1)), sharex=True)
    for q in range(n_qubits):
        residuals_q = y_obs[:, q] - y_model[:, q]
        axs_res[q].plot(times, residuals_q, label=f"Residuals Qubit {q}", alpha=0.7)
        axs_res[q].axhline(0, color='gray', linestyle='--')
        axs_res[q].set_ylabel("Residual")
        axs_res[q].legend()
        axs_res[q].set_title(f"Residuals (Observed - Predicted from theta0) Qubit {q}")

    axs_res[-1].plot(times, obs_mean - pred_mean, "k-", label="Combined Residuals", lw=2)
    axs_res[-1].axhline(0, color='gray', linestyle='--')
    axs_res[-1].set_xlabel("Time")
    axs_res[-1].set_ylabel("Residual")
    axs_res[-1].legend()
    axs_res[-1].set_title("Combined Residuals (Observed - Predicted from theta0)")

    plt.tight_layout()
    plt.suptitle("Residuals: Observed - Predicted from theta0", y=1.02, fontsize=16)
    plt.savefig('Signal_Differences.png', dpi=150)
    plt.show()
    plt.close('all')

    print("\n=== Signal Generation Verification Complete ===")