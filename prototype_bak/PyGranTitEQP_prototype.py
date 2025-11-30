# prototype.py: Quick and dirty PyGranTitEQP prototype
# Loads data.dat (volume in mL, pH) and plots titration curve, all G1's and G2's, and their first and second derivatives in a 7x4 grid

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """Load titration data from a text file."""
    try:
        # Assume space-separated, no headers
        # Adjust delimiter (e.g., delimiter=',') or header (e.g., header=0) if needed
        df = pd.read_csv(file_path, delim_whitespace=True, names=['volume', 'pH'])
        print("Data loaded successfully:")
        print(df.head())
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
    # Check monotonic increasing volumes
    if not df['volume'].is_monotonic_increasing:
        print("Warning: Volumes are not strictly increasing. Check data integrity.")
        return False
    # Check reasonable pH values (0 to 14)
    if (df['pH'] < 0).any() or (df['pH'] > 14).any():
        print("Warning: pH values outside typical range (0 to 14).")
    print(f"Data points: {len(df)}")
    return True

def plot_titration_and_gran(df, output_file='titration_and_gran.png'):
    """Plot titration curve, all G1's and G2's, and their first and second derivatives in a 7x4 grid."""
    if df is None or not validate_data(df):
        print("Cannot plot: Invalid or no data.")
        return
    # Set professional plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(7, 4, figsize=(16, 28), sharex=True)

    # Convert to NumPy arrays for plotting
    volume = df['volume'].to_numpy()
    pH = df['pH'].to_numpy()
    V = 25.0  # Initial volume to be titrated (mL)

    # Define Gran functions
    strongacid_g1 = (volume + V) * np.power(10, -pH)  # StrongAcid_G1 = (v + V) * 10^(-pH)
    strongacid_g2 = (volume + V) * np.power(10, pH)   # StrongAcid_G2 = (v + V) * 10^(pH)
    strongbase_g1 = (volume + V) * np.power(10, pH)   # StrongBase_G1 = (v + V) * 10^(pH)
    strongbase_g2 = (volume + V) * np.power(10, -pH)  # StrongBase_G2 = (v + V) * 10^(-pH)
    weakacid_g1   =  volume * np.power(10, -pH)       # WeakAcid_G1   =  v * 10^(-pH)
    weakacid_g2   = (volume + V) * np.power(10, pH)   # WeakAcid_G2   = (v + V) * 10^(pH)
    weakbase_g1   =  volume * np.power(10, pH)        # WeakBase_G1   =  v * 10^(pH)
    weakbase_g2   = (volume + V) * np.power(10, -pH)  # WeakBase_G2   = (v + V) * 10^(-pH)
    
    # Define Schwarz functions
    #schwarz_strongacid_g1 = (volume + V) *  np.power(10, -pH)  # Schwarz_StrongAcid_G1 = (v + V) * 10^(-pH)
    #schwarz_strongacid_g2 = (volume + V) *  np.power(10, pH - pKw)  # Schwarz_StrongAcid_G2 = (v + V) * 10^(pH - pKw)
    #schwarz_strongbase_g1 = (volume + V) *  np.power(10, pH - pKw)  # Schwarz_StrongBase_G1 = (v + V) * 10^(pH - pKw)
    #schwarz_strongbase_g2 = (volume + V) *  np.power(10, -pH)  # Schwarz_StrongBase_G2 = (v + V) * 10^(-pH)
    #schwarz_weakacid_g1   = (volume + V) * (np.power(10, -pH) / (np.power(10, -pH) + Ka)) * np.power(10, -pH)  # Schwarz_WeakAcid_G1 = (v + V) * ([H^+]/([H^+] + Ka_c)) * [H^+]
    #schwarz_weakacid_g2   = (volume + V) *  np.power(10, pH - pKw)  # Schwarz_WeakAcid_G2 = (v + V) * 10^(pH - pKw)
    #schwarz_weakbase_g1   = (volume + V) * (np.power(10, 14 - pH) / (np.power(10, 14 - pH) + Kb)) * np.power(10, 14 - pH)  # Schwarz_WeakBase_G1 = (v + V) * ([OH^-]/([OH^-] + Kb_c)) * [OH^-]
    #schwarz_weakbase_g2   = (volume + V) *  np.power(10, -pH)  # Schwarz_WeakBase_G2 = (v + V) * 10^(-pH)

    # Row 1: Titration curve in all 4 columns
    for col in range(4):
        axes[0, col].plot(volume, pH, marker='o', linestyle='-', color='blue', label='Titration Data')
        axes[0, col].set_ylabel('pH')
        axes[0, col].set_title('Titration Curve')
        axes[0, col].grid(True)
        axes[0, col].legend()

    # Row 2: All G1's (StrongAcid_G1, StrongBase_G1, WeakAcid_G1, WeakBase_G1)
    g1_list = [strongacid_g1, strongbase_g1, weakacid_g1, weakbase_g1]
    g1_labels = ['StrongAcid_G1 = (v + V) * 10^(-pH)', 'StrongBase_G1 = (v + V) * 10^(pH)', 'WeakAcid_G1 = v * 10^(-pH)', 'WeakBase_G1 = v * 10^(pH)']
    g1_colors = ['green', 'olive', 'orange', 'purple']
    g1_styles = ['-', '-', '--', ':']
    g1_markers = ['o', 'o', 's', '^']
    for col in range(4):
        axes[1, col].plot(volume, g1_list[col], marker=g1_markers[col], linestyle=g1_styles[col], color=g1_colors[col], label=g1_labels[col])
        axes[1, col].set_ylabel('Gran G1')
        axes[1, col].set_title(g1_labels[col].split(' = ')[0] + ' Plot')
        axes[1, col].grid(True)
        axes[1, col].legend()
        # Optional: Uncomment for log-scaling if values are too large/small
        # axes[1, col].set_yscale('log')

    # Row 3: First derivatives of all G1's
    for col in range(4):
        dg1 = np.gradient(g1_list[col], volume)  # dG1/dv
        axes[2, col].plot(volume, dg1, marker=g1_markers[col], linestyle=g1_styles[col], color=g1_colors[col], label='d(' + g1_labels[col].split(' = ')[0] + ')/dv')
        axes[2, col].set_ylabel('dG1/dv')
        axes[2, col].set_title('First Derivative of ' + g1_labels[col].split(' = ')[0])
        axes[2, col].grid(True)
        axes[2, col].legend()

    # Row 4: Second derivatives of all G1's
    for col in range(4):
        dg1 = np.gradient(g1_list[col], volume)  # dG1/dv
        d2g1 = np.gradient(dg1, volume)  # d²G1/dv²
        axes[3, col].plot(volume, d2g1, marker=g1_markers[col], linestyle=g1_styles[col], color=g1_colors[col], label='d²(' + g1_labels[col].split(' = ')[0] + ')/dv²')
        axes[3, col].set_ylabel('d²G1/dv²')
        axes[3, col].set_title('Second Derivative of ' + g1_labels[col].split(' = ')[0])
        axes[3, col].grid(True)
        axes[3, col].legend()

    # Row 5: All G2's (StrongAcid_G2, StrongBase_G2, WeakAcid_G2, WeakBase_G2)
    g2_list = [strongacid_g2, strongbase_g2, weakacid_g2, weakbase_g2]
    g2_labels = ['StrongAcid_G2 = (v + V) * 10^(pH)', 'StrongBase_G2 = (v + V) * 10^(-pH)', 'WeakAcid_G2 = (v + V) * 10^(pH)', 'WeakBase_G2 = (v + V) * 10^(-pH)']
    g2_colors = ['red', 'brown', 'cyan', 'magenta']
    g2_styles = ['-', '-', '--', ':']
    g2_markers = ['o', 'o', 's', '^']
    for col in range(4):
        axes[4, col].plot(volume, g2_list[col], marker=g2_markers[col], linestyle=g2_styles[col], color=g2_colors[col], label=g2_labels[col])
        axes[4, col].set_ylabel('Gran G2')
        axes[4, col].set_title(g2_labels[col].split(' = ')[0] + ' Plot')
        axes[4, col].grid(True)
        axes[4, col].legend()
        # Optional: Uncomment for log-scaling if values are too large/small
        # axes[4, col].set_yscale('log')

    # Row 6: First derivatives of all G2's
    for col in range(4):
        dg2 = np.gradient(g2_list[col], volume)  # dG2/dv
        axes[5, col].plot(volume, dg2, marker=g2_markers[col], linestyle=g2_styles[col], color=g2_colors[col], label='d(' + g2_labels[col].split(' = ')[0] + ')/dv')
        axes[5, col].set_ylabel('dG2/dv')
        axes[5, col].set_title('First Derivative of ' + g2_labels[col].split(' = ')[0])
        axes[5, col].grid(True)
        axes[5, col].legend()

    # Row 7: Second derivatives of all G2's
    for col in range(4):
        dg2 = np.gradient(g2_list[col], volume)  # dG2/dv
        d2g2 = np.gradient(dg2, volume)  # d²G2/dv²
        axes[6, col].plot(volume, d2g2, marker=g2_markers[col], linestyle=g2_styles[col], color=g2_colors[col], label='d²(' + g2_labels[col].split(' = ')[0] + ')/dv²')
        axes[6, col].set_xlabel('Volume Added (mL)')
        axes[6, col].set_ylabel('d²G2/dv²')
        axes[6, col].set_title('Second Derivative of ' + g2_labels[col].split(' = ')[0])
        axes[6, col].grid(True)
        axes[6, col].legend()

    plt.tight_layout()
    # Save combined plot
    plt.savefig(output_file, dpi=300)
    print(f"Plots saved as '{output_file}'")
    plt.show()

def main():
    """Main function to run the prototype."""
    file_path = 'data.dat'
    df = load_data(file_path)
    if df is not None:
        plot_titration_and_gran(df)

if __name__ == '__main__':
    main()