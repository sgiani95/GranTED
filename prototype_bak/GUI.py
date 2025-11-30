import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os

class TitrationMethodGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyGranTitEQP - Method Parameter Selection")
        self.root.geometry("500x450")

        # Variables for selections
        self.titration_type = tk.StringVar(value="don't know")
        self.titration_strength = tk.StringVar(value="don't know")
        self.method_file_path = tk.StringVar(value="")
        self.data_file_path = tk.StringVar(value="")
        self.analyte_volume = tk.StringVar(value="")
        self.titrant_concentration = tk.StringVar(value="")

        self.method_enabled = tk.BooleanVar(value=False)
        self.method_entry_state = tk.DISABLED

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Titration Type Block - Vertical grid-like radio buttons
        ttk.Label(main_frame, text="1. Titration Type:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        type_frame = ttk.Frame(main_frame)
        type_frame.grid(row=1, column=0, sticky=tk.W, pady=(0,10))
        type_frame.columnconfigure((0,1,2), weight=1)
        ttk.Radiobutton(type_frame, text="Acid", variable=self.titration_type, value="acid").grid(row=0, column=0, sticky=tk.W, padx=10)
        ttk.Radiobutton(type_frame, text="Basic", variable=self.titration_type, value="basic").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Radiobutton(type_frame, text="Don't know", variable=self.titration_type, value="don't know").grid(row=0, column=2, sticky=tk.W, padx=10)

        # Titration Strength Block - Vertical grid-like radio buttons
        ttk.Label(main_frame, text="2. Titration Strength:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(10,5))
        strength_frame = ttk.Frame(main_frame)
        strength_frame.grid(row=3, column=0, sticky=tk.W, pady=(0,10))
        strength_frame.columnconfigure((0,1,2), weight=1)
        ttk.Radiobutton(strength_frame, text="Strong", variable=self.titration_strength, value="strong").grid(row=0, column=0, sticky=tk.W, padx=10)
        ttk.Radiobutton(strength_frame, text="Weak", variable=self.titration_strength, value="weak").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Radiobutton(strength_frame, text="Don't know", variable=self.titration_strength, value="don't know").grid(row=0, column=2, sticky=tk.W, padx=10)

        # Titration Parameters Block - Vertical
        ttk.Label(main_frame, text="3. Titration Parameters:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(10,5))
        param_frame = ttk.Frame(main_frame)
        param_frame.grid(row=5, column=0, sticky=tk.W, pady=(0,10))
        ttk.Label(param_frame, text="Analyte Volume (mL):").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        ttk.Entry(param_frame, textvariable=self.analyte_volume, width=20).grid(row=0, column=1, sticky=tk.W, padx=(0,10))
        ttk.Label(param_frame, text="Titrant Concentration (M):").grid(row=1, column=0, sticky=tk.W, pady=(5,0), padx=(0,5))
        ttk.Entry(param_frame, textvariable=self.titrant_concentration, width=20).grid(row=1, column=1, sticky=tk.W, pady=(5,0), padx=(0,10))

        # File Selection Block
        ttk.Label(main_frame, text="4. File Locations:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky=tk.W, pady=(10,5))

        # Titration Data File Path first
        ttk.Label(main_frame, text="Titration Data File Path:").grid(row=7, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.data_file_path, width=50).grid(row=7, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_data_file).grid(row=7, column=2, padx=5)

        # Checkbox before Method File Save Path
        ttk.Checkbutton(main_frame, text="Use custom Method File Path", variable=self.method_enabled, command=self.toggle_method_entry).grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(10,0))

        # Method File Save Path second, disabled by default
        ttk.Label(main_frame, text="Method File Save Path:").grid(row=9, column=0, sticky=tk.W, pady=(0,0))
        method_entry = ttk.Entry(main_frame, textvariable=self.method_file_path, width=50, state=self.method_entry_state)
        method_entry.grid(row=9, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_method_file).grid(row=9, column=2, padx=5)

        # Buttons Block
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=20)
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def toggle_method_entry(self):
        if self.method_enabled.get():
            self.method_entry_state = tk.NORMAL
        else:
            self.method_entry_state = tk.DISABLED
        # For mockup, print state; in full app, reconfigure the entry
        print(f"Method entry state: {self.method_entry_state}")

    def browse_data_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.dat *.txt *.csv"), ("All files", "*.*")])
        if filename:
            self.data_file_path.set(filename)
            # Default Method File to same directory as Data File if not enabled
            if not self.method_enabled.get():
                data_dir = os.path.dirname(filename)
                default_method = os.path.join(data_dir, "method.json")
                self.method_file_path.set(default_method)

    def browse_method_file(self):
        filename = filedialog.asksaveasfilename(initialdir=os.path.dirname(self.data_file_path.get()) if self.data_file_path.get() else None,
                                               defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if filename:
            self.method_file_path.set(filename)

    def ok_clicked(self):
        # Mockup action: Print selections
        print("Titration Type:", self.titration_type.get())
        print("Titration Strength:", self.titration_strength.get())
        print("Analyte Volume:", self.analyte_volume.get())
        print("Titrant Concentration:", self.titrant_concentration.get())
        print("Method File Path:", self.method_file_path.get())
        print("Data File Path:", self.data_file_path.get())
        messagebox.showinfo("OK", "Parameters saved! (Mockup)")
        self.root.quit()

    def cancel_clicked(self):
        messagebox.showinfo("Cancel", "Operation cancelled. (Mockup)")
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = TitrationMethodGUI(root)
    root.mainloop()