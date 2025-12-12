
# analyzer.py: Module 4 for GranTED - Interval Identification and Parameter Optimization

#######################
# Core Functionality: #
#######################
#
# Interval Identification: Automatically detects the "Zone 2" linear region in the Gran function (g1) using derivative analysis + rolling R² maximization,
# targeting the steep negative-slope part for reliable fitting.
#
# Parameter Optimization: Placeholder for tuning k5 (and potentially k1/k6) to maximize R² in the identified interval
# (e.g., via Brent's method or grid search; currently fixed for testing).
#
# Analysis Orchestration: Integrates Gran computation, interval finding, and optimization, outputting results (interval indices, fit params, R²)
# for visualizer.py and reporter.py.
#
# Output: Dict with {'interval': (start, end), 'optimized': {'best_k5': float, 'max_r2': float, 'fit': (slope, intercept)}, 'g1': array}, ready for plotting/reporting.

import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
from gran_functions import compute_gran_functions
from scipy.optimize import minimize_scalar

def identify_linear_interval(g1, volume, min_points=5, window_size=5):
    """
    Identify Zone 2 interval using derivative + rolling R².
    Args:
        g1 (np.array): Gran function values.
        volume (np.array): Volume values.
        min_points (int): Minimum interval length.
        window_size (int): Smoothing window for derivative.
    Returns:
        tuple: (start_idx, end_idx, max_r2).
    """
    # Step 1: Smooth g1 and compute derivative
    g1_smooth = savgol_filter(g1, window_length=window_size, polyorder=2)
    dg1 = np.gradient(g1_smooth, volume)

    # Step 2: Find candidate interval (derivative negative to minimum)
    negative_start = np.where(dg1 < 0)[0]
    if len(negative_start) == 0:
        print("Warning: No negative derivative found. Using full range.")
        return 0, len(g1) - 1, 0.0

    start_idx = negative_start[0]
    min_deriv_idx = start_idx + np.argmin(dg1[start_idx:])

    # Step 3: Rolling R² within candidate window
    best_r2 = 0.0
    best_start, best_end = start_idx, min_deriv_idx
    for test_start in range(start_idx, min_deriv_idx - min_points + 1):
        for test_end in range(test_start + min_points, min(min_deriv_idx + 10, len(g1))):
            if test_end - test_start < min_points:
                continue
            slope, intercept, r_value, _, _ = linregress(volume[test_start:test_end], g1_smooth[test_start:test_end])
            r2 = r_value**2
            if r2 > best_r2:
                best_r2 = r2
                best_start, best_end = test_start, test_end

    print(f"Identified interval: indices {best_start}-{best_end}, R²={best_r2:.3f}")
    return best_start, best_end, best_r2

def optimize_k_params(df, params, interval_indices, method='bounded', k_bounds=(-10, 10)):
    """
    Optimize k5 (or k1) using bounded method to maximize R² in interval.
    Args:
        df (pd.DataFrame): Data with 'volume', 'potential'.
        params (dict): Preprocess params.
        interval_indices (tuple): (start, end) from identify_linear_interval.
        method (str): 'bounded' for 1D with bounds.
        k_bounds (tuple): Bounds for k5 (default -10 to 10).
    Returns:
        dict: {'best_k5': float, 'max_r2': float, 'fit': (slope, intercept)}.
    """
    volume = df['volume'].iloc[interval_indices[0]:interval_indices[1]].values
    potential = df['potential'].iloc[interval_indices[0]:interval_indices[1]].values
    pH = 7 - (potential / 59.16)

    def negative_r2(k5):
        y = volume * np.power(10, k5 - pH)  # k6=1 fixed
        slope, intercept, r_value, _, _ = linregress(volume, y)
        return -r_value**2

    result = minimize_scalar(negative_r2, bounds=k_bounds, method=method)
    best_k5 = result.x
    max_r2 = -result.fun

    # Recompute fit with best k5
    y_opt = volume * np.power(10, best_k5 - pH)
    slope, intercept, _, _, _ = linregress(volume, y_opt)
    fit = (slope, intercept)

    print(f"Optimized k5={best_k5:.3f}, Max R²={max_r2:.3f}")
    return {'best_k5': best_k5, 'max_r2': max_r2, 'fit': fit}

def analyze_gran(df, params):
    """
    Main analysis: Compute Gran, identify interval, optimize k5.
    Args:
        df (pd.DataFrame): Preprocessed data.
        params (dict): From preprocess.py.
    Returns:
        dict: Results with interval, optimized k, fit.
    """
    # Compute Gran functions (weakacid_g1 focus)
    gran_results = compute_gran_functions(df, params)
    g1 = gran_results['weakacid_g1']
    volume = params['volume']

    # Identify interval
    start_idx, end_idx, interval_r2 = identify_linear_interval(g1, volume)

    # Optimize k5 in interval
    optimized = optimize_k_params(df, params, (start_idx, end_idx))

    results = {
        'interval': (start_idx, end_idx),
        'interval_r2': interval_r2,
        'optimized': optimized,
        'g1': g1,  # For plotting
    }

    print("Gran analysis complete.")
    return results

# Example usage (for testing)
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data.dat', names=['volume', 'potential'])
    params = {'V': 25.0}
    results = analyze_gran(df, params)
    print("Results:", results)