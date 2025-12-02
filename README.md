Excel Data Inspection and Visualization Tool

This project provides a command-line tool for loading Excel datasets, performing basic data preparation, generating descriptive summaries, and producing exploratory data visualizations.
It is designed for quickly inspecting structured datasets containing property repair, housing, or similar tabular data.

📦 Dependencies

Install the required libraries:

pip install pandas numpy matplotlib openpyxl xlrd

🚀 How to Run the Script:

Below are examples of how to use the available command-line arguments.

1. Inspect an Excel file (no plots):
python housing_repair_descriptive_analysis.py -f data.xlsx

2. Show plots:
python housing_repair_descriptive_analysis.py -f data.xlsx --plot

3. Export processed data to CSV:
python housing_repair_descriptive_analysis.py -f data.xlsx --to-csv cleaned/output.csv

4. Load a specific sheet from the Excel file:
python housing_repair_descriptive_analysis.py -f data.xlsx --sheet "RepairData" --plot

5. Process a multi-sheet Excel file. Automatically prints summaries and plots for every sheet:

python housing_repair_descriptive_analysis.py -f synthetic_property_data.xlsx --plot --to-csv output.c
