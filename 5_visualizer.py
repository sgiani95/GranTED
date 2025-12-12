# visualizer.py: Module 5 for GranTED - Plotting and Visualization

#######################
# Core Functionality: #
#######################
#
# Plot Generation: Creates Matplotlib plots for titration curves (volume vs. pH from mV),
# Gran functions (g1 with highlighted linear interval and fit line), and
# screened k-values (multiple lines for comparison).
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
from analyzer import prediction_ci  # Import CI helper

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
    Plot Gran functions with linear interval, fit line, and CI bands, plus derivative subplot.
    Args:
        results (dict): From analyzer.py (e.g., {'optimized': {'fit': (slope, intercept), 'fit_ci': ...}}).
        params (dict): From preprocess.py.
        output_dir (str): Directory to save plot.
        filename (str): Output filename.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    volume = params['volume']
    g1 = results['g1']  # weakacid_g1 array
    start_idx, end_idx = results['interval']
    slope, intercept = results['optimized']['fit']
    slope_ci, intercept_ci = results['optimized']['fit_ci']  # New: CI from analyzer

    # Compute smoothed derivative for debugging subplot
    g1_smooth = savgol_filter(g1, window_length=min(7, len(g1)), polyorder=2)
    dg1 = np.gradient(g1_smooth, volume)

    # Compute prediction CI band for interval
    interval_volume = volume[start_idx:end_idx]
    interval_g1 = g1[start_idx:end_idx]
    x_fit = np.linspace(interval_volume[0], interval_volume[-1], 100)
    y_upper, y_lower = prediction_ci(interval_volume, interval_g1, x_fit, slope, intercept)

    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Main plot: g1 with interval, fit line, and CI band
    ax1.plot(volume, g1, 'o-', color='green', label='WeakAcid_G1', alpha=0.7)
    ax1.plot(volume[start_idx:end_idx], g1[start_idx:end_idx], 'o-', color='red', label='Linear Interval')
    # Fit line
    y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, '--', color='black', label=f'Fit: slope={slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}], int={intercept:.3f}')
    # CI band (shaded)
    ax1.fill_between(x_fit, y_lower, y_upper, color='lightblue', alpha=0.3, label='95% Prediction CI')
    ax1.set_ylabel('WeakAcid_G1')
    ax1.set_title('Gran Plot with Optimized Interval, Fit, and CI')
    ax1.grid(True)
    ax1.legend()

    # Subplot: Derivative dg1/dV for debugging
    ax2.plot(volume, dg1, 'o-', color='orange', label='dg1/dV (smoothed)')
    ax2.axhline(y=0, color='gray', linestyle='--', label='Zero Slope')
    ax2.axvline(x=volume[start_idx], color='red', linestyle='--', alpha=0.7, label='Interval Start')
    ax2.axvline(x=volume[end_idx], color='red', linestyle='--', alpha=0.7, label='Interval End')
    ax2.set_xlabel('Volume Added (mL)')
    ax2.set_ylabel('dg1/dV')
    ax2.set_title('Derivative for Interval Debugging')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()
    print(f"Gran plot with CI saved to {output_dir}/{filename}")

def plot_screened_k(results, params, output_dir='output', filename='k_screening.png'):
    """
    Plot screened k-values for weakacid_g1 (if available).
    Args:
        results (dict): From gran_functions.py (e.g., {'weakacid_g1_screened