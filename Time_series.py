import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import font

class AnomalyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Detection in Time Series Data")

        # Title Label
        self.title_label = tk.Label(root, text="Anomaly Detection in Time Series Data", font=("Times New Roman", 28, "bold"))
        self.title_label.pack(pady=10)  # Padding to give some space around the title

        self.button_frame = tk.Frame(root, bg='lightgray', padx=10, pady=10)
        self.button_frame.pack(pady=20)

        # Define a font style
        btn_font = font.Font(family='Helvetica', size=10, weight='bold')

        # Create buttons with styling
        self.load_button = tk.Button(self.button_frame, text="Load Data", command=self.load_data)
        self.load_button.configure(bg='blue', fg='white', font=btn_font)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.detect_button = tk.Button(self.button_frame, text="Detect Anomalies", command=self.detect_anomalies, state=tk.DISABLED)
        self.detect_button.configure(bg='green', fg='white', font=btn_font)
        self.detect_button.grid(row=0, column=1, padx=10, pady=10)

        self.visualize_button = tk.Button(self.button_frame, text="Visualize Data", command=self.visualize_data, state=tk.DISABLED)
        self.visualize_button.configure(bg='purple', fg='white', font=btn_font)
        self.visualize_button.grid(row=0, column=2, padx=10, pady=10)

    def load_data(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            try:
                self.data = pd.read_csv(self.filepath)
                if 'Date' in self.data.columns:
                    try:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
                    except Exception as e:
                        raise ValueError("Date parsing failed: " + str(e))
                    self.data.set_index('Date', inplace=True)
                    messagebox.showinfo("Info", "Data Loaded Successfully")
                    self.detect_button.config(state=tk.NORMAL)
                else:
                    raise ValueError("Missing 'Date' column in the CSV file.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
    def detect_anomalies(self):
        try:
            self.anomaly_results = {}
            for column in self.data.columns:
                # Calculate the z-score of the column
                self.data[f'z_score_{column}'] = stats.zscore(self.data[column])
                threshold = 3
                self.data[f'anomaly_{column}'] = self.data[f'z_score_{column}'].apply(
                    lambda x: 'Anomaly' if np.abs(x) > threshold else 'Normal'
                )
                self.anomaly_results[column] = self.data[self.data[f'anomaly_{column}'] == 'Anomaly']
            messagebox.showinfo("Info", "Anomaly Detection Completed")
            self.visualize_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def visualize_data(self):
        try:
            columns_to_plot = [column for column in self.data.columns if not column.startswith('z_score') and not column.startswith('anomaly')]
            num_columns = len(columns_to_plot)
            plt.figure(figsize=(15, 10))
            for i, column in enumerate(columns_to_plot, 1):  # Exclude 'z_score' and 'anomaly' columns
                plt.subplot((num_columns // 2) + 1, 2, i)
                plt.plot(self.data.index, self.data[column], label='Value')
                anomalies = self.anomaly_results.get(column)
                if anomalies is not None:
                    plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomaly')
                plt.title(f'Column {column}')
                plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    root.mainloop()
