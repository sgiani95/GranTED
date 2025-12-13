# analyzer.py: Module 4 for GranTED - Interval Identification and Parameter Optimization

#######################
# Core Functionality: #
#######################
#
# Interval Identification: Automatically detects "the Zone" linear region in the Gran function (g1) using derivative analysis + rolling R² maximization,
# targeting the steep negative-slope part for reliable fitting. Includes optional SavGol smoothing based on raw derivative noise check and segmented plateau detection for robustness.
#
# Parameter Optimization: Tunes k5 to maximize R² in the identified interval, with optional extension to include more points while maintaining linearity. Computes and stores both unoptimized (k5=0) and optimized results for comparison.
#
# Analysis Orchestration: Integrates Gran computation, interval finding, and optimization, outputting results (interval indices, fit params, R²) for both cases
# for visualizer.py and reporter.py.
#
# Output: Dict with {'interval': (start, end), 'unoptimized': {...}, 'optimized': {...}, 'g1': array}, ready for plotting/reporting.

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import savgol_filter
from gran_functions import compute_gran_functions
from scipy.optimize import minimize_scalar

def _compute_r2(g1_smooth, volume, start, end):
    """Helper: Compute R² and fit for a segment."""
    if end - start < 2:
        return 0.0, 0.0, 0.0
    slope, intercept, r_value, _, _ = linregress(volume[start:end], g1_smooth[start:end])
    return r_value**2, slope, intercept

def identify_linear_interval(g1, volume, min_points=5, window_size=5, noise_threshold=0.001, 
                             var_threshold_rel=0.1, min_r2=0.95):
    """
    Identify "the Zone" using derivative-based segmentation and ranking.
    Handles Cases 1-3 via plateau detection in negative dg1.
    Args:
        g1 (np.array): Gran function values.
        volume (np.array): Volume values.
        min_points (int): Minimum plateau length.
        window_size (int): Smoothing window (if noisy).
        noise_threshold (float): Std dev for applying smoothing.
        var_threshold_rel (float): Relative variance threshold for plateaus.
        min_r2 (float): Minimum R² for high-quality candidates.
    Returns:
        tuple: (start_idx, end_idx, max_r2).
    """
    # Step 1: Compute dg1 (with optional smoothing)
    dg1_raw = np.gradient(g1, volume)
    std_dg1 = np.std(dg1_raw)
    if std_dg1 > noise_threshold:
        print(f"High noise (std_dg1={std_dg1:.2e} > {noise_threshold}). Smoothing.")
        g1_smooth = savgol_filter(g1, window_length=window_size, polyorder=2)
        dg1 = np.gradient(g1_smooth, volume)
    else:
        print(f"Low noise (std_dg1={std_dg1:.2e} <= {noise_threshold}). No smoothing.")
        g1_smooth = g1
        dg1 = dg1_raw
    
    # Focus on negative dg1 region
    neg_mask = dg1 < 0
    if not np.any(neg_mask):
        print("Warning: No negative derivative. Using full range.")
        return 0, len(g1) - 1, 0.0
    
    dg1_neg = dg1[neg_mask]
    vol_neg_idx = np.where(neg_mask)[0]
    
    # Step 2: Segment into candidate plateaus via sliding-window variance
    candidates = []
    win_sizes = range(max(3, min_points//2), min(15, len(dg1_neg)//2) + 1)
    for win_len in win_sizes:
        for i in range(len(dg1_neg) - win_len + 1):
            seg_dg1 = dg1_neg[i:i+win_len]
            mean_dg = np.mean(seg_dg1)
            var_dg = np.var(seg_dg1)
            if mean_dg < 0 and var_dg < var_threshold_rel * abs(mean_dg):
                orig_start = vol_neg_idx[i]
                orig_end = vol_neg_idx[i + win_len - 1] + 1
                candidates.append({
                    'orig_start': orig_start,
                    'orig_end': orig_end,
                    'length': win_len,
                    'mean_dg': mean_dg,
                    'var_dg': var_dg
                })
    
    if not candidates:
        print("No low-var negative segments found. Falling back to full negative region.")
        full_start = vol_neg_idx[0]
        full_end = vol_neg_idx[-1] + 1
        _, _, r2_full = _compute_r2(g1_smooth, volume, full_start, full_end)
        return full_start, full_end, r2_full
    
    # Step 3: Rank candidates by R², length, |mean_dg|
    scored = []
    for cand in candidates:
        r2, slope, intercept = _compute_r2(g1_smooth, volume, cand['orig_start'], cand['orig_end'])
        if r2 >= min_r2:
            score = r2 * 100 + (cand['length'] / len(g1)) * 10 + abs(cand['mean_dg']) * 0.1
            scored.append((cand, score, r2))
    
    if not scored:
        # Fallback: Score all by composite even if R² < min_r2
        fallback_scored = []
        for cand in candidates:
            r2, _, _ = _compute_r2(g1_smooth, volume, cand['orig_start'], cand['orig_end'])
            score = r2 * 100 + (cand['length'] / len(g1)) * 10 + abs(cand['mean_dg']) * 0.1
            fallback_scored.append((cand, score, r2))
        scored = fallback_scored
    
    best_cand, best_score, best_r2 = max(scored, key=lambda x: x[1])
    start_idx, end_idx = best_cand['orig_start'], best_cand['orig_end']
    
    # Case labeling (heuristic)
    case = "Case 3"  # Default gradual
    if start_idx > 0:
        pre_mean = np.mean(dg1[:start_idx])
        pre_std = np.std(dg1[:start_idx])
        if pre_mean > 0:
            case = "Case 1"
        elif pre_std < 0.01 * abs(pre_mean):
            case = "Case 2"
    
    print(f"Identified 'the Zone': indices {start_idx}-{end_idx}, R²={best_r2:.3f} (Case: {case}, Score: {best_score:.2f})")
    return start_idx, end_idx, best_r2

def _compute_fit(df, start, end, k5=0.0):
    """
    Helper: Compute fit, R², and V_eq for a zone with given k5.
    """
    volume = df['volume'].iloc[start:end].values
    potential = df['potential'].iloc[start:end].values
    pH = 7 - (potential / 59.16)
    y = volume * np.power(10, k5 - pH)  # k6=1 fixed
    slope, intercept, r_value, _, _ = linregress(volume, y)
    r2 = r_value**2
    veq = -intercept / slope if slope != 0 else np.nan
    return {'r2': r2, 'fit': (slope, intercept), 'veq': veq}

def _optimize_single_zone(df, start, end, k_bounds):
    """
    Helper: Optimize k5 for a fixed zone (k6=1).
    """
    volume = df['volume'].iloc[start:end].values
    potential = df['potential'].iloc[start:end].values
    pH = 7 - (potential / 59.16)
    
    def negative_r2(k5):
        y = volume * np.power(10, k5 - pH)  # k6=1 fixed
        slope, intercept, r_value, _, _ = linregress(volume, y)
        return -r_value**2
    
    result = minimize_scalar(negative_r2, bounds=k_bounds, method='bounded')
    best_k5 = result.x
    max_r2 = -result.fun
    y_opt = volume * np.power(10, best_k5 - pH)
    slope, intercept, _, _, _ = linregress(volume, y_opt)
    return {'best_k5': best_k5, 'best_r2': max_r2, 'fit': (slope, intercept)}

def optimize_k_params(df, params, initial_interval, max_expand=5, r2_threshold=0.99, 
                      method='bounded', k_bounds=(-10, 10), use_extension=True):
    """
    Optimize k5 starting from initial interval, optionally extending for better linearization.
    (No unoptimized return—handled in analyze_gran; focuses on opt extension.)
    """
    if not use_extension:
        print("No extension: Optimizing on initial interval only.")
        zone_results = _optimize_single_zone(df, initial_interval[0], initial_interval[1], k_bounds)
        best_k5 = zone_results['best_k5']
        final_start, final_end = initial_interval
        best_num_points = final_end - final_start
        opt = _compute_fit(df, final_start, final_end, k5=best_k5)
        opt['k5'] = best_k5  # Add k5 to opt dict
        print(f"Optimized (k5={best_k5:.3f}): R²={opt['r2']:.4f}, V_eq={opt['veq']:.3f} mL over {best_num_points} points")
        return {
            'optimized': opt,
            'final_start': final_start,
            'final_end': final_end
        }
    
    # Extension mode (starting from initial)
    current_start, current_end = initial_interval
    final_start, final_end = current_start, current_end
    best_num_points = current_end - current_start
    best_r2 = 0.0
    best_k5 = 0.0
    best_fit = (0.0, 0.0)
    best_veq = np.nan
    
    for _ in range(max_expand + 1):
        zone_results = _optimize_single_zone(df, current_start, current_end, k_bounds)
        curr_k5 = zone_results['best_k5']
        curr_r2 = zone_results['best_r2']
        curr_fit = zone_results['fit']
        curr_veq = -curr_fit[1] / curr_fit[0] if curr_fit[0] != 0 else np.nan
        
        if (curr_r2 > best_r2 or 
            (abs(curr_r2 - best_r2) < 0.001 and (current_end - current_start) > best_num_points)):
            best_r2 = curr_r2
            best_k5 = curr_k5
            best_fit = curr_fit
            best_veq = curr_veq
            final_start, final_end = current_start, current_end
            best_num_points = current_end - current_start
        
        if curr_r2 < r2_threshold:
            break
        
        expanded = False
        new_start = max(0, current_start - 1)
        if new_start < current_start:
            left_results = _optimize_single_zone(df, new_start, current_end, k_bounds)
            if left_results['best_r2'] > curr_r2 * 0.99:
                current_start = new_start
                expanded = True
        
        # Try right expansion
        new_end = min(len(df), current_end + 1)
        if new_end > current_end:
            # Compute candidate zone g1 and dg1 for guardrail check
            candidate_start = current_start
            candidate_end = new_end
            candidate_volume = df['volume'].iloc[candidate_start:candidate_end].values
            candidate_pH = 7 - (df['potential'].iloc[candidate_start:candidate_end] / 59.16)
            candidate_g1 = candidate_volume * np.power(10, curr_k5 - candidate_pH)  # Use current k5 for consistency
            candidate_g1_smooth = savgol_filter(candidate_g1, window_length=min(5, len(candidate_g1)), polyorder=2)
            candidate_dg1 = np.gradient(candidate_g1_smooth, candidate_volume)
    
            # New segment (last min(5, added points) for artifact check)
            new_seg_len = min(5, new_end - current_end)
            new_seg_start = new_end - new_seg_len
            new_seg_g1 = candidate_g1[new_seg_start:]
            new_seg_dg1 = candidate_dg1[new_seg_start:]
    
            # Guardrails
            g1_near_zero = np.mean(new_seg_g1) < 1e-7
            dg1_near_zero = np.abs(np.mean(new_seg_dg1)) < 5e-6
            steep_dg1_change = np.mean(np.abs(np.diff(new_seg_dg1))) > 1e-6
    
            if not (g1_near_zero or dg1_near_zero or steep_dg1_change):
                right_results = _optimize_single_zone(df, candidate_start, candidate_end, k_bounds)
                if right_results['best_r2'] > curr_r2 * 0.99:
                    current_end = new_end
                    expanded = True
                    print(f"Extended right to {current_end} (R²={right_results['best_r2']:.4f}) - passed guardrails")
                else:
                    print("Right extension rejected: R² too low")
            else:
                print(f"Right extension rejected: g1_near_zero={g1_near_zero}, dg1_near_zero={dg1_near_zero}, steep_dg1={steep_dg1_change}")
        
        if not expanded:
            break
    
    opt = _compute_fit(df, final_start, final_end, k5=best_k5)
    opt['k5'] = best_k5  # Add k5 to opt dict
    print(f"Optimized Zone (extended): k5={best_k5:.3f}, R²={opt['r2']:.4f}, V_eq={opt['veq']:.3f} mL over {best_num_points} points")
    
    return {
        'optimized': opt,
        'final_start': final_start,
        'final_end': final_end
    }

def analyze_gran(df, params, use_segmented=True, use_extension=True):
    """
    Main analysis: Compute Gran, identify raw interval, fit raw, then optimize/extend for opt Zone.
    Returns results with separate raw/opt Zones for comparison.
    """
    # Compute Gran functions (weakacid_g1 focus, with k5=0 default)
    gran_results = compute_gran_functions(df, params)
    g1 = gran_results['weakacid_g1']
    params['volume'] = df['volume'].values  # Ensure volume in params for visualizer

    # Identify initial interval (on raw g1)
    if use_segmented:
        start_idx, end_idx, _ = identify_linear_interval(g1, params['volume'])
    else:
        start_idx, end_idx, _ = _identify_linear_original(g1, params['volume'])

    initial_interval = (start_idx, end_idx)
    
    # Raw Zone: Fit on initial interval (no opt/extension)
    raw_zone = _compute_fit(df, start_idx, end_idx, k5=0.0)
    raw_zone['start'] = start_idx
    raw_zone['end'] = end_idx
    raw_zone['num_points'] = end_idx - start_idx
    
    print(f"Raw Zone (initial): indices {start_idx}-{end_idx}, R²={raw_zone['r2']:.4f}, V_eq={raw_zone['veq']:.3f} mL over {raw_zone['num_points']} points")

    # Opt k5 on raw Zone (no extension yet)
    k_bounds = (-10, 10)  # Default bounds
    opt_k5 = _optimize_single_zone(df, start_idx, end_idx, k_bounds)['best_k5']
    
    # Recompute g1_opt and re-detect Zone on it
    g1_opt = g1 * np.power(10, opt_k5)
    if use_segmented:
        opt_start, opt_end, opt_interval_r2 = identify_linear_interval(g1_opt, params['volume'])
    else:
        opt_start, opt_end, opt_interval_r2 = _identify_linear_original(g1_opt, params['volume'])
    
    # Opt fit on re-detected Zone
    opt_zone = _compute_fit(df, opt_start, opt_end, k5=opt_k5)
    opt_zone['k5'] = opt_k5
    opt_zone['start'] = opt_start
    opt_zone['end'] = opt_end
    opt_zone['num_points'] = opt_end - opt_start

    # Fallback if opt Zone smaller than raw (prevent shrinkage)
    if opt_zone['num_points'] < raw_zone['num_points']:
        opt_start, opt_end = raw_zone['start'], raw_zone['end']
        opt_zone = _compute_fit(df, opt_start, opt_end, k5=opt_k5)
        opt_zone['k5'] = opt_k5  # Ensure k5 in fallback
        opt_zone['start'] = opt_start
        opt_zone['end'] = opt_end
        opt_zone['num_points'] = opt_end - opt_start
        print(f"Opt fallback to raw Zone (to avoid shrinkage): R²={opt_zone['r2']:.4f}, points={opt_zone['num_points']}")

    print(f"Opt Zone (final): k5={opt_k5:.3f}, R²={opt_zone['r2']:.4f}, V_eq={opt_zone['veq']:.3f} mL over {opt_zone['num_points']} points")

    results = {
        'raw_zone': raw_zone,  # Initial raw
        'opt_zone': opt_zone,  # Re-detected or fallback opt
        'g1': g1,  # Raw g1 for plotting
        'g1_opt': g1_opt,  # Opt g1 for plotting
        'interval_r2': raw_zone['r2'],  # Legacy
    }

    print("Gran analysis complete with separate raw/opt Zones.")
    return results

def optimize_k_params(df, params, initial_interval, max_expand=5, r2_threshold=0.99, 
                      method='bounded', k_bounds=(-10, 10), use_extension=True):
    """
    Optimize k5 starting from initial interval, optionally extending for better linearization.
    Returns optimized results with 'k5' key included.
    """
    if not use_extension:
        print("No extension: Optimizing on initial interval only.")
        zone_results = _optimize_single_zone(df, initial_interval[0], initial_interval[1], k_bounds)
        best_k5 = zone_results['best_k5']
        final_start, final_end = initial_interval
        best_num_points = final_end - final_start
        opt = _compute_fit(df, final_start, final_end, k5=best_k5)
        opt['k5'] = best_k5  # Ensure k5 is in opt dict
        opt['num_points'] = best_num_points  # Add num_points
        print(f"Optimized (k5={best_k5:.3f}): R²={opt['r2']:.4f}, V_eq={opt['veq']:.3f} mL over {best_num_points} points")
        return {
            'optimized': opt,
            'final_start': final_start,
            'final_end': final_end
        }
    
    # In extension loop, for right expansion:
    new_end = min(len(df), current_end + 1)
    if new_end > current_end:
        # Candidate g1/dg1 for guardrail (use curr_k5)
        candidate_start = current_start
        candidate_end = new_end
        candidate_volume = df['volume'].iloc[candidate_start:candidate_end].values
        candidate_pH = 7 - (df['potential'].iloc[candidate_start:candidate_end] / 59.16)
        candidate_g1 = candidate_volume * np.power(10, curr_k5 - candidate_pH)
        candidate_g1_smooth = savgol_filter(candidate_g1, window_length=min(5, len(candidate_g1)), polyorder=2)
        candidate_dg1 = np.gradient(candidate_g1_smooth, candidate_volume)
        
        # New segment check (last 5 points)
        new_seg_len = min(5, new_end - current_end)
        new_seg_start = new_end - new_seg_len
        new_seg_g1 = candidate_g1[new_seg_start:]
        new_seg_dg1 = candidate_dg1[new_seg_start:]
        
        g1_near_zero = np.mean(new_seg_g1) < g1_zero_thresh
        dg1_near_zero = np.abs(np.mean(new_seg_dg1)) < dg1_zero_thresh
        steep_dg1_change = np.mean(np.abs(np.diff(new_seg_dg1))) > dg1_steep_thresh
        
        if not (g1_near_zero or dg1_near_zero or steep_dg1_change):
            right_results = _optimize_single_zone(df, candidate_start, candidate_end, k_bounds)
            if right_results['best_r2'] >= curr_r2 * r2_threshold:  # Stricter: must not drop
                current_end = new_end
                expanded = True
                print(f"Extended right to {current_end} (R²={right_results['best_r2']:.4f})")
        else:
            print("Right extension rejected by guardrails")

def _identify_linear_original(g1, volume, min_points=5, window_size=5):
    """
    Original non-segmented interval identification (fallback).
    """
    dg1_raw = np.gradient(g1, volume)
    std_dg1 = np.std(dg1_raw)
    if std_dg1 > 0.001:
        g1_smooth = savgol_filter(g1, window_length=window_size, polyorder=2)
        dg1 = np.gradient(g1_smooth, volume)
    else:
        g1_smooth = g1
        dg1 = dg1_raw
    negative_start = np.where(dg1 < 0)[0]
    if len(negative_start) == 0:
        return 0, len(g1) - 1, 0.0
    start_idx = negative_start[0]
    min_deriv_idx = start_idx + np.argmin(dg1[start_idx:])
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
    print(f"Original method: indices {best_start}-{best_end}, R²={best_r2:.3f}")
    return best_start, best_end, best_r2

# Example usage (for testing)
if __name__ == "__main__":
    df = pd.read_csv('data.dat', names=['volume', 'potential'], sep='\s+')
    params = {'V': 25.0}
    results = analyze_gran(df, params)
    print("Results:", results)