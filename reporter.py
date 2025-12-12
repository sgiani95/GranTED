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

def export_to_csv(results, params, output_dir='output', filename='gran_results.csv'):
    """
    Export analysis results to CSV.
    Args:
        results (dict): From analyzer.py (e.g., {'optimized': {'best_k5': float, 'max_r2': float}}).
        params (dict): From preprocess.py (e.g., {'V': 25.0}).
        output_dir (str): Directory to save CSV.
        filename (str): Output filename.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare export data
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
    df_export = pd.DataFrame(export_data)
    
    filepath = Path(output_dir) / filename
    df_export.to_csv(filepath, index=False)
    print(f"CSV report saved to {filepath}")

def export_to_pdf(results, params, output_dir='output', filename='gran_report.pdf'):
    """
    Export summary to PDF report.
    Args:
        results (dict): From analyzer.py.
        params (dict): From preprocess.py.
        output_dir (str): Directory to save PDF.
        filename (str): Output filename.
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
    
    # Green metrics (placeholder)
    story.append(Paragraph("Green Metrics (Placeholder)", styles['Heading2']))
    story.append(Paragraph("Waste: Low | Energy: Minimal | Toxicity: Safe", styles['Normal']))
    
    doc.build(story)
    print(f"PDF report saved to {filepath}")

def save_method_json(params, results, output_dir='output', filename='method.json'):
    """
    Save optimized method as JSON for reuse.
    Args:
        params (dict): User parameters.
        results (dict): Optimized results.
        output_dir (str): Directory to save JSON.
        filename (str): Output filename.
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
        'optimized': {'best_k5': 0.98, 'max_r2': 0.995},
        'interval': (5, 15),
        'g1': np.random.rand(20)
    }
    generate_report(df, params, mock_results)