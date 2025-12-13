# reporter.py: Module 6 for GranTED - Results Export and Reporting

import pandas as pd
from pathlib import Path
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np  # For ndarray check
import pandas as pd  # For Series check

def convert_to_serializable(obj):
    """
    Recursively convert numpy arrays, scalars, tuples, and pandas Series to lists/primitive types for JSON serialization.
    Args:
        obj: Object to convert (dict, list, tuple, ndarray, Series, np.int64, np.float64, etc.).
    Returns:
        Serializable object.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
            
def export_to_csv(results, params, output_dir='output', filename='gran_results.csv'):
    """
    Export analysis results to CSV, including raw and opt Zone comparisons.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare export data with comparison (raw = unopt baseline, opt = extended)
    export_data = {
        'Parameter': [
            'V (mL)', 'C_B (M)', 
            'Raw Zone k5', 'Raw Zone R²', 'Raw Zone V_eq (mL)', 'Raw Zone Num Points',
            'Opt Zone k5', 'Opt Zone R²', 'Opt Zone V_eq (mL)', 'Opt Zone Num Points',
            'Raw Interval Start', 'Raw Interval End', 'Opt Interval Start', 'Opt Interval End'
        ],
        'Value': [
            params.get('V', 25.0),
            params.get('C_B', 0.1),
            0.0,  # Fixed for raw
            results['raw_zone']['r2'],
            results['raw_zone']['veq'],
            results['raw_zone']['num_points'],
            results['opt_zone']['k5'],
            results['opt_zone']['r2'],
            results['opt_zone']['veq'],
            results['opt_zone']['num_points'],
            results['raw_zone']['start'],
            results['raw_zone']['end'],
            results['opt_zone']['start'],
            results['opt_zone']['end']
        ]
    }
    df_export = pd.DataFrame(export_data)
    
    filepath = Path(output_dir) / filename
    df_export.to_csv(filepath, index=False)
    print(f"CSV report saved to {filepath}")

def export_to_pdf(results, params, output_dir='output', filename='gran_report.pdf'):
    """
    Export summary to PDF report, including raw/opt comparisons.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    filepath = Path(output_dir) / filename
    doc = SimpleDocTemplate(str(filepath), pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontSize=14, alignment=1))
    
    story = []
    story.append(Paragraph("GranTED Titration Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Parameters table with raw/opt
    param_data = [
        ['Parameter', 'Value'],
        ['V (mL)', str(params.get('V', 25.0))],
        ['C_B (M)', str(params.get('C_B', 0.1))],
        ['Raw Zone R²', f"{results['raw_zone']['r2']:.3f}"],
        ['Raw Zone V_eq', f"{results['raw_zone']['veq']:.3f} mL"],
        ['Opt Zone k5', f"{results['opt_zone']['k5']:.3f}"],
        ['Opt Zone R²', f"{results['opt_zone']['r2']:.3f}"],
        ['Opt Zone V_eq', f"{results['opt_zone']['veq']:.3f} mL"],
        ['Raw Interval', f"{results['raw_zone']['start']}-{results['raw_zone']['end']}"],
        ['Opt Interval', f"{results['opt_zone']['start']}-{results['opt_zone']['end']}"]
    ]
    story.append(Table(param_data, colWidths=[2*inch, 1.5*inch]))
    story.append(Spacer(1, 12))
    
    # Green metrics (placeholder)
    story.append(Paragraph("Green Metrics (Placeholder)", styles['Heading2']))
    story.append(Paragraph("Waste: Low | Energy: Minimal | Toxicity: Safe", styles['Normal']))
    
    doc.build(story)
    print(f"PDF report saved to {filepath}")

def save_method_json(params, results, output_dir='output', filename='method.json'):
    """
    Save raw/opt method as JSON for reuse.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert non-serializable types
    method_data = convert_to_serializable({
        'params': params,
        'raw_zone': results['raw_zone'],
        'opt_zone': results['opt_zone'],
        'g1': results['g1']  # Raw g1
    })
    
    filepath = Path(output_dir) / filename
    with open(filepath, 'w') as f:
        json.dump(method_data, f, indent=4)
    print(f"Method JSON saved to {filepath}")

def generate_report(df, params, results, output_dir='output'):
    """
    Orchestrate all exports: CSV, PDF, JSON.
    Args:
        df (pd.DataFrame): Raw data.
        params (dict): From preprocess.py.
        results (dict): From analyzer.py.
        output_dir (str): Directory to save reports.
    """
    export_to_csv(results, params, output_dir)
    export_to_pdf(results, params, output_dir)
    save_method_json(params, results, output_dir)
    print(f"Full report generated in {output_dir}")

# Example usage (for testing)
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data.dat', names=['volume', 'potential'])
    params = {'V': 25.0}
    # Mock results (replace with analyzer call)
    mock_results = {
        'optimized': {'k5': 0.98, 'r2': 0.995, 'fit': (0.5, 2.0), 'veq': 4.0, 'num_points': 15},
        'unoptimized': {'r2': 0.99, 'fit': (0.48, 1.92), 'veq': 4.0, 'num_points': 15},
        'interval': (5, 15),
        'g1': np.random.rand(20)
    }
    generate_report(df, params, mock_results)