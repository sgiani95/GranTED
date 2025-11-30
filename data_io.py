# data_io.py: Module 1 for GranTED - Data Input and Validation

import pandas as pd
import numpy as np
from pathlib import Path
import os

class DataLoader:
    def __init__(self, default_format='dat'):
        self.default_format = default_format  # e.g., 'dat', 'csv', 'xlsx'
        self.supported_formats = ['dat', 'txt', 'csv', 'xlsx']

    def load_single_file(self, file_path, columns=['volume', 'potential']):
        """
        Load a single titration file.
        Args:
            file_path (str): Path to the file.
            columns (list): Expected columns (default: ['volume', 'potential']).
        Returns:
            pd.DataFrame: Loaded data, or None if invalid.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File '{file_path}' not found.")
            return None

        try:
            if path.suffix in ['.dat', '.txt', '.csv']:
                df = pd.read_csv(file_path, delim_whitespace=True, names=columns, header=None)
            elif path.suffix == '.xlsx':
                df = pd.read_excel(file_path, names=columns, header=None)
            else:
                print(f"Unsupported format: {path.suffix}. Use {self.supported_formats}.")
                return None

            # Basic shape check
            if df.shape[1] < 2:
                print("Error: File must have at least 2 columns (volume, potential).")
                return None

            print(f"Loaded {len(df)} points from '{file_path}'.")
            print(df.head())
            return df
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
            return None

    def load_batch_files(self, directory_path):
        """
        Load multiple files from a directory.
        Args:
            directory_path (str): Directory containing .dat/.csv files.
        Returns:
            dict: {filename: DataFrame} for valid files.
        """
        path = Path(directory_path)
        if not path.exists():
            print(f"Error: Directory '{directory_path}' not found.")
            return {}

        files = {}
        for file in path.glob('*.dat'):
            df = self.load_single_file(file)
            if df is not None:
                files[str(file.name)] = df

        print(f"Loaded {len(files)} files from '{directory_path}'.")
        return files

    def validate_data(self, df):
        """
        Validate DataFrame: monotonic volume, reasonable potential.
        Args:
            df (pd.DataFrame): Data with 'volume' and 'potential' columns.
        Returns:
            bool: True if valid.
        """
        if df is None or len(df) < 3:
            print("Error: DataFrame too small or None.")
            return False

        if not df['volume'].is_monotonic_increasing:
            print("Warning: Volumes not strictly increasing.")
            return False

        # Check reasonable potential values (-420 to 420 mV, approx pH 0-14, see Nernst equation pH = 7 - (E / 59.16))
        if (df['potential'] < -420).any() or (df['potential'] > 420).any():
            print("Warning: Potential values outside typical range (-420 to 420 mV).")

        print(f"Validated: {len(df)} points, volume range {df['volume'].min():.2f}-{df['volume'].max():.2f} mL")
        return True

# Example usage (for testing)
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_single_file('data.dat')
    if df is not None:
        loader.validate_data(df)

    # Batch example
    ### batch = loader.load_batch_files('./titrations/')
    ### print("Batch files:", list(batch.keys()))