import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("Titration Setup Mockup")
    root.geometry("600x260")
    root.resizable(False, False)

    # Configure grid: 4 equal-width columns
    for i in range(4):
        root.grid_columnconfigure(i, weight=1)

    # === Row 0: Header 1 ===
    ttk.Label(root, text="Titration Type", font=("Arial", 10, "bold"))\
        .grid(row=0, column=0, columnspan=4, pady=(10,5), sticky="w")

    # === Row 1: Titration Type Options ===
    titration_var = tk.StringVar(value="acidbase")

    ttk.Radiobutton(root, text="Acid / Base", variable=titration_var, value="acidbase")\
        .grid(row=1, column=0, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Precipitation", variable=titration_var, value="precipitation")\
        .grid(row=1, column=1, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Redox", variable=titration_var, value="redox")\
        .grid(row=1, column=2, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Complexometry", variable=titration_var, value="complexometry")\
        .grid(row=1, column=3, pady=5, sticky="w")

    # === Row 2: Separator ===
    ttk.Separator(root, orient="horizontal")\
        .grid(row=2, column=0, columnspan=4, sticky="ew", pady=8)

    # === Row 3: Header 2 ===
    ttk.Label(root, text="Acid (Base) Strength: Analyte (Sample)", font=("Arial", 10, "bold"))\
        .grid(row=3, column=0, columnspan=4, pady=(0,5), sticky="w")

    # === Row 4: Strength Options ===
    strength_var = tk.StringVar(value="strong_acid")

    ttk.Radiobutton(root, text="Strong Acid", variable=strength_var, value="strong_acid")\
        .grid(row=4, column=0, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Weak Acid", variable=strength_var, value="weak_acid")\
        .grid(row=4, column=1, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Strong Base", variable=strength_var, value="strong_base")\
        .grid(row=4, column=2, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Weak Base", variable=strength_var, value="weak_base")\
        .grid(row=4, column=3, pady=5, sticky="w")

    # === Row 5: OK / Cancel Buttons ===
    ttk.Button(root, text="OK")\
        .grid(row=5, column=1, pady=15, sticky="ew")
    ttk.Button(root, text="Cancel")\
        .grid(row=5, column=2, pady=15, sticky="ew")

    root.mainloop()

if __name__ == "__main__":
    main()
