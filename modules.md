| # | Module Name | Role |
|---|-------------|------|
| 1 | data_io.py | Load/validate data from files (CSV/dat/Excel), handle batch inputs, extract metadata (e.g., V_0, C_B). |
| 2 | preprocess.py | Smoothing, titration type detection (acid/base, strong/weak via pH trend/Gran linearity), sample-driven param estimation. |
| 3 | gran_functions.py | Compute Gran G1/G2 variants (strong/weak acid/base), including Schwartz adaptations as screened functions (k-values). |
| 4 | analyzer.py | Endpoint detection (linear regions, regression), batch comparison, pKa estimation, green metrics. |
| 5 | visualizer.py | Generate plots (Matplotlib/Gnuplot: titration, Gran/Schwartz, 3D for screening), annotations. |
| 6 | reporter.py | Export results (CSV/PDF: endpoints, pKa, R², green scores), method sharing (JSON). |
| 7 | gui_cli.py | Tkinter GUI for batch param selection, argparse CLI for headless/batch runs. |
| 8 | learning.py | Parameter optimization (grid search/ML for k-values, R²-based scoring), auto-tuning for sample-driven mode. |
| 9 | main.py | Entry point: Orchestrate workflow (load → preprocess → analyze → visualize → report), CLI/GUI dispatch. |
| 10 | unit_testing.py | Pytest suite for modules (e.g., test Gran functions with mock data.dat, edge cases like sparse data). |