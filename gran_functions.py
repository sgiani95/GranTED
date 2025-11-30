# gran_functions.py: Module 3 for GranTED - Gran Function Computation

import numpy as np
import pandas as pd

def compute_gran_functions(df, params):
    """
    Compute Gran functions for strong and weak acid titrations.
    Args:
        df (pd.DataFrame): Data with 'volume' and 'potential' columns.
        params (dict): From preprocess.py (e.g., {'V': 25.0, 'C_B': 0.1}).
    Returns:
        dict: {'strongacid_g1': array, 'weakacid_g1': array}.
    """
    # Ensure numeric types
    volume = pd.to_numeric(df['volume'], errors='coerce').to_numpy()
    potential = pd.to_numeric(df['potential'], errors='coerce').to_numpy()
    pH = 7 - (potential / 59.16)  # Convert mV to pH

    V = float(params.get('V', 25.0))
    k1 = float(params.get('k1', 0.0))
    k5 = float(params.get('k5', 0.0))
    k2 = float(params.get('k2', 1.0))
    k6 = float(params.get('k6', 1.0))

    # Default Gran functions (k=0)
    strongacid_g1 = ((volume + V) * np.power(10, k1 - pH)) / k2
    weakacid_g1 = (volume * np.power(10, k5 - pH)) / k6

    results = {
        'strongacid_g1': strongacid_g1,
        'weakacid_g1': weakacid_g1,
    }

    print(f"Computed Gran functions with V={V}, k1={k1}, k5={k5}")
    return results

# Example usage (for testing)
if __name__ == "__main__":
    df = pd.read_csv('data.dat', sep='\t', names=['volume', 'potential'])
    params = {'V': 25.0, 'k1': 0.0, 'k5': 0.0, 'k2': 1.0, 'k6': 1.0}
    gran_results = compute_gran_functions(df, params)
    print("Gran results keys:", list(gran_results.keys()))