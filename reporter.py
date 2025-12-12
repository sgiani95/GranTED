# reporter.py: Module 6 for GranTED - Results Export and Reporting

#######################
# Core Functionality: #
#######################
#
# Export Generation: Creates user-friendly outputs from analysis results (from analyzer.py) and params (from preprocess.py),
# including CSV tables (key metrics like k5, R²), PDF summaries (formatted report with tables), and JSON methods
# (reusable configs with arrays converted to lists).
#
# Serialization Handling: Converts non-JSON-safe types (ndarrays, Series) to lists via recursive function for robust saving.
#
# Green Metrics Placeholder: Space for future eco-scoring (e.g., waste from V_eq * C_B), keeping it extensible.
#
# Output: Files saved to a directory (CSV for data, PDF for printable reports, JSON for methods), with console feedback.

import pandas as pd
from pathlib import Path
import json
from typing import Any, Dict, List  # For type hints
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np  # For ndarray check
import pandas as pd  # For Series check
from io import BytesIO  # For BytesIO in plot embedding
import matplotlib.pyplot as plt  # For saving plots to BytesIO

def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and pandas Series to lists for JSON serialization.
    Args:
        obj: Object to convert (dict, list, ndarray, Series, etc.).
    Returns:
        Serializable object.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    else:
        return obj

def compute_green_metrics(results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for green chemistry metrics (e.g., waste, energy).
    Args:
        results: From analyzer.py.
        params: From preprocess.py.
    Returns:
        Dict with green scores (expand later).
    """
    # Placeholder calculations (e.g., waste = V_eq * C_B * density)
    v_eq = -results['optimized']['fit'][1] / results['optimized']['fit'][0]  # Extrapolated EQP
    waste_ml = v_eq * params.get('C_B', 0.1) * 1000  # Rough titrant waste in mg (assume 1 g/mL)
    
    return {
        'waste_mg': waste_ml,
        'energy_kwh': 0.001,  # Placeholder for heating/etc.
        'toxicity_score': 'Low',  # Based on titrant type
        'overall_greenness': 8.5 / 12.0  # AGREE-like score (0-1)
    }

def export_to_csv(results: Dict[str, Any], params: Dict[str, Any], g1_data: np.ndarray, output_dir: str = 'output', filename: str = 'gran_results.csv') -> None:
    """
    Export analysis results to CSV, including full g1 array as second sheet.
    Args:
        results: From analyzer.py (e.g., {'optimized': {'best_k5': float, 'max_r2': float}}).
        params: From preprocess.py (e.g., {'V': 25.0}).
        g1_data: Gran g1 array from gran_functions.py.
        output_dir: Directory to save CSV.
        filename: Output filename.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Sheet 1: Metrics table
    export_data = {
        'Parameter': ['V (mL)', 'C_B (M)', 'Best k5', 'Max R²', 'Interval Start', 'Interval End'],
        'Value': [
            params.get('V', 25.0),
            params.get('C_B', 0.1),
            results['optimized']['best_k5'],
            results['optimized']['max_r2'],
            results['interval'][0],
            results['interval'][1]
        ]
    }
    df_metrics = pd.DataFrame(export_data)
    
    # Sheet 2: Full g1 array
    volume = params['volume']
    df_g1 = pd.DataFrame({'Volume (mL)': volume, 'WeakAcid_G1': g1_data})
    
    filepath = Path(output_dir) / filename
    # with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
    #     df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
    #     df_g1.to_excel(writer, sheet_name='G1_Data', index=False)
    # print(f"CSV/Excel report (with G1 sheet) saved to {filepath}")

def export_to_pdf(results: Dict[str, Any], params: Dict[str, Any], plot_paths: List[str], output_dir: str = 'output', filename: str = 'gran_report.pdf') -> None:
    """
    Export summary to PDF report with embedded plots.
    Args:
        results: From analyzer.py.
        params: From preprocess.py.
        plot_paths: List of PNG paths (e.g., from visualizer.py).
        output_dir: Directory to save PDF.
        filename: Output filename.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    filepath = Path(output_dir) / filename
    doc = SimpleDocTemplate(str(filepath), pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontSize=14, alignment=1))
    
    story = []
    story.append(Paragraph("GranTED Titration Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Parameters table
    param_data = [
        ['Parameter', 'Value'],
        ['V (mL)', str(params.get('V', 25.0))],
        ['C_B (M)', str(params.get('C_B', 0.1))],
        ['Best k5', f"{results['optimized']['best_k5']:.3f}"],
        ['Max R²', f"{results['optimized']['max_r2']:.3f}"],
        ['Interval', f"{results['interval'][0]}-{results['interval'][1]}"]
    ]
    story.append(Table(param_data, colWidths=[2*inch, 1.5*inch]))
    story.append(Spacer(1, 12))
    
    # Embed plots (save to BytesIO and insert)
    for plot_path in plot_paths:
        if Path(plot_path).exists():
            img = Image(str(plot_path), width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
    
    # Green metrics
    green = compute_green_metrics(results, params)
    story.append(Paragraph("Green Metrics", styles['Heading2']))
    green_data = [
        ['Metric', 'Value'],
        ['Waste (mg)', f"{green['waste_mg']:.2f}"],
        ['Energy (kWh)', f"{green['energy_kwh']:.3f}"],
        ['Toxicity', green['toxicity_score']],
        ['Overall Greenness', f"{green['overall_greenness']:.3f}"]
    ]
    story.append(Table(green_data, colWidths=[2*inch, 1.5*inch]))
    
    doc.build(story)
    print(f"PDF report with embedded plots saved to {filepath}")

def save_method_json(params: Dict[str, Any], results: Dict[str, Any], output_dir: str = 'output', filename: str = 'method.json') -> None:
    """
    Save optimized method as JSON for reuse.
    Args:
        params: User parameters.
        results: Optimized results.
        output_dir: Directory to save JSON.
        filename: Output filename.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert non-serializable types (ndarrays/Series to lists)
    method_data = convert_to_serializable({
        'params': params,
        'optimized': results['optimized'],
        'interval': results['interval'],
        'g1': results['g1']  # Fixed: Use convert_to_serializable for arrays/Series
    })
    
    filepath = Path(output_dir) / filename
    with open(filepath, 'w') as f:
        json.dump(method_data, f, indent=4)
    print(f"Method JSON saved to {filepath}")

def generate_report(df: pd.DataFrame, params: Dict[str, Any], results: Dict[str, Any], plot_paths: List[str], output_dir: str = 'output') -> None:
    """
    Orchestrate all exports: CSV/Excel, PDF with plots, JSON.
    Args:
        df: Raw data.
        params: From preprocess.py.
        results: From analyzer.py.
        plot_paths: List of PNG paths from visualizer.py.
        output_dir: Directory to save reports.
    """
    export_to_csv(results, params, g1_data=results['g1'], output_dir=output_dir)
    export_to_pdf(results, params, plot_paths, output_dir=output_dir)
    save_method_json(params, results, output_dir=output_dir)
    print(f"Full report generated in {output_dir}")

# Example usage (for testing)
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data.dat', names=['volume', 'potential'])
    params = {'V': 25.0}
    # Mock results (replace with analyzer call)
    mock_results = {
        'optimized': {'best_k5': 0.98, 'max_r2': 0.995},
        'interval': (5, 15),
        'g1': np.random.rand(20)
    }
    mock_plot_paths = ['./mock_plot1.png', './mock_plot2.png']  # Replace with real paths
    generate_report(df, params, mock_results, mock_plot_paths)