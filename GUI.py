import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("Titration Setup Mockup")
    root.geometry("400x300")
    root.resizable(False, False)

    # === Field 1: Titration Type ===
    frame1 = ttk.LabelFrame(root, text="Titration Type")
    frame1.pack(fill="x", padx=10, pady=10)

    titration_var = tk.StringVar()

    ttk.Radiobutton(frame1, text="Acid / Base", variable=titration_var, value="acidbase").pack(anchor="w", padx=10, pady=2)
    ttk.Radiobutton(frame1, text="Precipitation", variable=titration_var, value="precipitation").pack(anchor="w", padx=10, pady=2)
    ttk.Radiobutton(frame1, text="Redox", variable=titration_var, value="redox").pack(anchor="w", padx=10, pady=2)
    ttk.Radiobutton(frame1, text="Complexometry", variable=titration_var, value="complexometry").pack(anchor="w", padx=10, pady=2)

    # === Field 2: Acid/Base Strength ===
    frame2 = ttk.LabelFrame(root, text="Acid (Base) Strength: Analyte (Sample)")
    frame2.pack(fill="x", padx=10, pady=10)

    strength_var = tk.StringVar()

    ttk.Radiobutton(frame2, text="Strong Acid", variable=strength_var, value="strong_acid").pack(anchor="w", padx=10, pady=2)
    ttk.Radiobutton(frame2, text="Weak Acid", variable=strength_var, value="weak_acid").pack(anchor="w", padx=10, pady=2)
    ttk.Radiobutton(frame2, text="Strong Base", variable=strength_var, value="strong_base").pack(anchor="w", padx=10, pady=2)
    ttk.Radiobutton(frame2, text="Weak Base", variable=strength_var, value="weak_base").pack(anchor="w", padx=10, pady=2)

    # === Buttons ===
    button_frame = ttk.Frame(root)
    button_frame.pack(side="bottom", pady=15)

    ttk.Button(button_frame, text="OK").pack(side="left", padx=10)
    ttk.Button(button_frame, text="Cancel").pack(side="right", padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()
