
# visualizer.py: Module 5 for GranTED - Plotting and Visualization

#######################
# Core Functionality: #
#######################
#
# Plot Generation: Creates Matplotlib plots for titration curves (volume vs. pH from mV),
# Gran functions (g1 with highlighted linear interval and fit line for both unoptimized and optimized), and
# screened k-values (multiple lines for comparison). Derivative subplot autoscale focuses on negative part.
#
# Customization: Supports output directories, filenames, and annotations (e.g., slope/intercept labels); uses seaborn style for professional look.
#
# Orchestration: Single entry visualize_all calls all plots, saving PNGs (300 DPI) to a specified dir.
#
# Output: High-res PNG files for reports/papers; extensible for Gnuplot 3D or batch summaries.

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter  # For derivative smoothing

def plot_titration_curve(df, params, output_dir='output', filename='titration_curve.png'):
    """
    Plot the raw titration curve (volume vs. potential or pH).
    Args:
        df (pd.DataFrame): Data with 'volume', 'potential'.
        params (dict): From preprocess.py (e.g., {'V': 25.0}).
        output_dir (str): Directory to save plot.
        filename (str): Output filename.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    potential = df['potential'].to_numpy()
    volume = df['volume'].to_numpy()
    pH = 7 - (potential / 59.16)  # Convert mV to pH for plotting

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volume, pH, 'o-', color='blue', label='Titration Curve (pH)')
    ax.set_xlabel('Volume Added (mL)')
    ax.set_ylabel('pH (from potential)')
    ax.set_title('Titration Curve')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()
    print(f"Titration curve saved to {output_dir}/{filename}")

def plot_gran_functions(results, params, output_dir='output', filename='gran_functions.png'):
    """
    Plot Gran functions in vertically stacked panels for raw and opt Zones (separate intervals if extended),
    each with linear interval and fit, plus derivative subplot.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    volume = params['volume']
    g1 = results['g1']  # Raw g1
    raw_start, raw_end = results['raw_zone']['start'], results['raw_zone']['end']
    opt_start, opt_end = results['opt_zone']['start'], results['opt_zone']['end']
    
    # Recompute optimized g1
    k5_opt = results['opt_zone']['k5']
    g1_opt = g1 * np.power(10, k5_opt)
    
    # Smooth for derivative (on raw g1)
    g1_smooth = savgol_filter(g1, window_length=min(7, len(g1)), polyorder=2)
    dg1 = np.gradient(g1_smooth, volume)
    
    # Raw/opt metrics
    raw_r2 = results['raw_zone']['r2']
    raw_veq = results['raw_zone']['veq']
    raw_fit = results['raw_zone']['fit']
    opt_r2 = results['opt_zone']['r2']
    opt_veq = results['opt_zone']['veq']
    opt_fit = results['opt_zone']['fit']
    
    plt.style.use('seaborn-v0_8')
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Top panel: Raw g1 + raw Zone
    axs[0].plot(volume, g1, 'o-', color='blue')
    axs[0].axvspan(volume[raw_start], volume[raw_end-1], alpha=0.3, color='red', label='Raw Zone')
    x_fit_full = np.linspace(0, volume[-1], 100)
    y_raw_fit = raw_fit[0] * x_fit_full + raw_fit[1]
    axs[0].plot(x_fit_full, y_raw_fit, '-', color='black')
    axs[0].set_ylabel('Raw g1 (k5=0)')
    axs[0].set_title(f'Raw Zone (R²={raw_r2:.3f}, V_eq={raw_veq:.3f} mL, {raw_end-raw_start} pts)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Middle panel: Opt g1 + opt Zone
    axs[1].plot(volume, g1_opt, 'o-', color='green')
    axs[1].axvspan(volume[opt_start], volume[opt_end-1], alpha=0.3, color='orange', label='Opt Zone')
    y_opt_fit = opt_fit[0] * x_fit_full + opt_fit[1]
    axs[1].plot(x_fit_full, y_opt_fit, '--', color='orange')
    axs[1].set_ylabel(f'Opt g1 (k5={k5_opt:.3f})')
    axs[1].set_title(f'Opt Zone (R²={opt_r2:.3f}, V_eq={opt_veq:.3f} mL, {opt_end-opt_start} pts)')
    axs[1].grid(True)
    axs[1].legend()
    
    # Bottom panel: Derivative (raw g1, raw Zone for consistency)
    axs[2].plot(volume, dg1, 'o-', color='orange', label='dg1/dV (smoothed, raw g1)')
    axs[2].axhline(y=0, color='gray', linestyle='--', label='Zero Slope')
    axs[2].axvspan(volume[raw_start], volume[raw_end-1], alpha=0.3, color='red')
    axs[2].set_xlabel('Volume Added (mL)')
    axs[2].set_ylabel('dg1/dV')
    axs[2].set_title('Derivative for Raw Zone Debugging (Negative Focus)')
    axs[2].grid(True)
    axs[2].legend()
    min_dg1 = np.min(dg1)
    axs[2].set_ylim([min_dg1, 0])
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()
    print(f"Gran plot (separate raw/opt Zones) saved to {output_dir}/{filename}")

def plot_screened_k(results, params, output_dir='output', filename='k_screening.png'):
    """
    Plot screened k-values for weakacid_g1 (if available).
    Args:
        results (dict): From gran_functions.py (e.g., {'weakacid_g1_screened': {k: array}}).
        params (dict): From preprocess.py.
        output_dir (str): Directory to save plot.
        filename (str): Output filename.
    """
    if 'weakacid_g1_screened' not in results:
        print("No screened k data available. Skipping plot.")
        return

    Path(output_dir).mkdir(exist_ok=True)
    
    volume = params['volume']
    screened = results['weakacid_g1_screened']

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(screened)))
    for i, (k_str, g1_k) in enumerate(screened.items()):
        k_val = float(k_str.split('=')[1])
        ax.plot(volume, g1_k, '-', color=colors[i], label=f'k5={k_val}')
    ax.set_xlabel('Volume Added (mL)')
    ax.set_ylabel('WeakAcid_G1 (screened k5)')
    ax.set_title('Gran Plot for Screened k5 Values')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()
    print(f"k screening plot saved to {output_dir}/{filename}")

def visualize_all(df, params, results, output_dir='output'):
    """
    Orchestrate all plots: titration curve, Gran functions (with comparison), screening.
    Args:
        df (pd.DataFrame): Raw data.
        params (dict): From preprocess.py.
        results (dict): From analyzer.py.
        output_dir (str): Directory to save plots.
    """
    plot_titration_curve(df, params, output_dir)
    plot_gran_functions(results, params, output_dir)
    plot_screened_k(results, params, output_dir)
    print(f"All visualizations saved to {output_dir}")

# Example usage (for testing)
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data.dat', names=['volume', 'potential'])
    params = {'V': 25.0}
    # Mock results (replace with analyzer call)
    mock_results = {
        'g1': np.random.rand(len(df)),
        'interval': (5, 15),
        'unoptimized': {'r2': 0.99, 'fit': (0.5, 2.0), 'veq': 4.0},
        'optimized': {'k5': 2.5, 'r2': 0.995, 'fit': (0.48, 1.92), 'veq': 4.0}
    }
    visualize_all(df, params, mock_results)