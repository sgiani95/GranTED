# PyGranTitEQP
Gran and Schwartz methods for titration processing, analysis, visualization and output.

Fusion: Py (Python) + Gran + Tit (Titration) + EQP (Equivalence Point).

![Logo](/logo.png)

---

**Meaning:** Sounds like “pigment,” evoking the idea of “coloring” titration data with clear endpoints, like a vivid lab result. The fusion of Gran and Titration highlights the core methods and context, while EQP emphasizes the endpoint goal.

**Why It Fits:** Suggests a vibrant, precise tool for titration analysis, with a nod to chemistry’s visual flair (e.g., indicator color changes).

---

## Flowchart

![Flowchart](/flowchart.png)

---

# Gran and Schwartz Titration Curve Processing and Evaluation: Project Requirements

## 1. Project Overview
- **Goal**: Develop a Python module for processing titration data to generate Gran plots and apply the Schwartz formula, identify linear regions, calculate endpoints (e.g., equivalence points), and evaluate metrics like accuracy or uncertainty.
- **Scope**: Core implementation of Gran and Schwartz methods with optional extensions for automation, error handling, or instrument integration.
- **Assumptions**: Input data from potentiometric titrations (e.g., pH vs. volume). Users have basic Python knowledge.

## 2. Functional Requirements

### 2.1 Data Input
- Accept titration data in CSV, Excel (.xlsx), or plain text formats.
- Required columns: Titrant volume (mL) and measured response (e.g., pH or potential in mV).
- Optional metadata: Sample concentration, titrant concentration, temperature, ionic strength.
- Handle variable data sizes (10–1000 points).

### 2.2 Data Preprocessing
- Clean data: Remove outliers, handle missing values, smooth noise (e.g., moving averages or Savitzky-Golay filters).
- Normalize volumes (e.g., correct for initial volume).
- Validate input: Ensure monotonic increasing volumes, reasonable pH ranges (0–14).

### 2.3 Gran Plot and Schwartz Formula Processing
- Compute **Gran functions**:
  - For strong acid-strong base: F1 = V * 10^(pH - 14) or similar variants.
  - Support first and second Gran plots for polyprotic systems.
  - Generalize for other titrations (e.g., weak acids/bases) with customizable equations.
- Compute **Schwartz formula**:
  - Implement the Schwartz linearization method (e.g., for weak acid/base titrations, using equations like V * (10^(pH - pK) - 1) or equivalents based on titration type).
  - Support user-defined parameters (e.g., pKa estimates, stability constants) for flexible application.
- Allow users to select Gran, Schwartz, or both methods for analysis via GUI or CLI flags.
- Automatically identify linear regions for both methods (e.g., via linear regression on segments).
- Detect endpoints by finding intersections of linear fits (e.g., using least squares) for Gran and Schwartz outputs.

### 2.4 Evaluation and Analysis
- Calculate key parameters: Equivalence volume, analyte concentration, pKa (if applicable), using both Gran and Schwartz methods.
- Compare results from Gran and Schwartz for consistency (e.g., endpoint differences, statistical metrics).
- Compute statistics: R² of linear fits, standard deviation of endpoint, confidence intervals for both methods.
- Compare multiple datasets (e.g., replicates) for reproducibility.

### 2.5 Visualization
- Generate plots: Raw titration curve, Gran plot(s), Schwartz plot(s), with marked endpoints and linear fits.
- Use Matplotlib or Seaborn for customizable outputs (save as PNG/PDF).
- Interactive options: Enable zooming or tooltips in Jupyter environments, with options to toggle Gran/Schwartz results.

### 2.6 Output
- Export results: Summary report in text/CSV (endpoint volume, errors, Gran vs. Schwartz comparisons).
- Optional: Generate PDF report with plots and tables, including both Gran and Schwartz results.

### 2.7 Advanced Features (Optional)
- Batch processing for multiple files, supporting both Gran and Schwartz methods.
- Simulation mode: Generate synthetic titration data for testing Gran and Schwartz calculations.
- Integration with external libraries (e.g., SciPy for optimization, Pandas for data handling).

## 3. Non-Functional Requirements
- **Performance**: Process datasets up to 10,000 points in under 5 seconds on standard hardware.
- **Usability**:
  - Provide **two interfaces** to support different workflows:
    - **Graphical User Interface (GUI)**: For method development and parameter tuning (e.g., Gran function constants, Schwartz pKa estimates, smoothing parameters). Built with Tkinter or similar, allowing interactive input (file selection, parameter adjustment), visualization of Gran and Schwartz results, and manual validation of endpoints.
    - **Command-Line Interface (CLI)**: For routine batch analysis. Use argparse for input file paths, configuration files, or parameters (e.g., titrant concentration, method selection: Gran/Schwartz/both, output directory). Support automation for processing multiple datasets.
  - Provide clear error messages and help documentation for both interfaces.
- **Reliability**: Handle edge cases (e.g., non-ideal data, multiple endpoints, Schwartz-specific constraints). Include unit tests for Gran and Schwartz functions.
- **Dependencies**: Minimize external libraries; use NumPy, SciPy, Pandas, Matplotlib. No internet-required packages.
- **Portability**: Python 3.8+ compatible; cross-platform (Windows, macOS, Linux).
- **Documentation**: Include README with usage examples (GUI and CLI, including Schwartz options), code comments, and references to Gran and Schwartz method theory.
- **Security/Privacy**: All processing local; no data transmission.
