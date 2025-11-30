# second_prototype.py: Script for PyGranTitEQP
# Loads data.dat (volume in mL, pH) and plots titration curve and Schwarz functions for V=25 and varying k values

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """Load titration data from a text file."""
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, names=['volume', 'pH'])
        print("Data loaded successfully:")
        print(df.head())
        print(f"Volume range: {df['volume'].min()} to {df['volume'].max()} mL")
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Ensure the file is in the same directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def validate_data(df):
    """Validate titration data for monotonic volumes and reasonable pH."""
    if df is None:
        print("No data to validate.")
        return False
    if not df['volume'].is_monotonic_increasing:
        print("Warning: Volumes are not strictly increasing. Check data integrity.")
        return False
    if (df['pH'] < 0).any() or (df['pH'] > 14).any():
        print("Warning: pH values outside typical range (0 to 14).")
    print(f"Data points: {len(df)}")
    return True

def plot_titration_and_schwarz(df, output_file='titration_and_schwarz.png'):
    """Plot titration curve and Schwarz functions for V=25 and varying k values in a 5x1 grid."""
    if df is None or not validate_data(df):
        print("Cannot plot: Invalid or no data.")
        return
    plt.style.use('seaborn-v0_8')

    volume = df['volume'].to_numpy()
    pH = df['pH'].to_numpy()

    # Fixed V and k values
    V = 25
    k_values = [0.99, 0.9, 1, 1.1, 1.11]
    k2 = k22 = k6 = k66 = 1.0  # Fixed scaling constants

    # Define Schwarz functions
    def strongacid_g1(volume, pH, k1):
        return ((volume + V) * np.power(10, k1 - pH)) / k2

    def strongbase_g1(volume, pH, k11):
        return ((volume + V) * np.power(10, k11 + pH)) / k22

    def weakacid_g1(volume, pH, k5):
        return (volume * np.power(10, k5 - pH)) / k6  # No V dependence

    def weakbase_g1(volume, pH, k55):
        return (volume * np.power(10, k55 + pH)) / k66  # No V dependence

    schwarz_types = [strongacid_g1, strongbase_g1, weakacid_g1, weakbase_g1]
    schwarz_labels = ['Schwarz_StrongAcid_G1', 'Schwarz_StrongBase_G1', 'Schwarz_WeakAcid_G1', 'Schwarz_WeakBase_G1']
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    fig, axes = plt.subplots(5, 1, figsize=(8, 25), sharex=True)

    # Row 1: Titration curve
    axes[0].plot(volume, pH, marker='o', linestyle='-', color='blue', label='Titration Data')
    axes[0].set_ylabel('pH')
    axes[0].set_title('Titration Curve')
    ### axes[0].set_xlim(0, 10)
    if pH.size > 0:
        y_min, y_max = np.min(pH), np.max(pH)
        y_margin = (y_max - y_min) * 0.1
        axes[0].set_ylim(y_min - y_margin, y_max + y_margin)
    axes[0].grid(True)
    axes[0].legend()

    # Rows 2-5: Schwarz functions with varying k
    for row in range(1, 5):
        schwarz_func = schwarz_types[row-1]
        label = schwarz_labels[row-1]
        axes[row].set_ylabel(label)
        axes[row].set_title(f'{label} for V={V}, varying k')
        for i, k in enumerate(k_values):
            y = schwarz_func(volume, pH, k)
            axes[row].plot(volume, y, label=f'k={k}', color=colors[i])
            if y.size > 0:
                y_min, y_max = np.min(y), np.max(y)
                if y_max != y_min:  # Avoid zero range
                    y_margin = (y_max - y_min) * 0.1
                    axes[row].set_ylim(y_min - y_margin, y_max + y_margin)
        axes[row].grid(True)
        if row == 1:
            axes[row].legend(ncol=3, bbox_to_anchor=(0.75, 1.0), loc='upper center', fontsize='small')

    axes[-1].set_xlabel('Volume Added (mL)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plots saved as '{output_file}'")
    plt.show()

def main():
    """Main function to run the prototype."""
    file_path = 'data.dat'
    df = load_data(file_path)
    if df is not None:
        plot_titration_and_schwarz(df)

if __name__ == '__main__':
    main()