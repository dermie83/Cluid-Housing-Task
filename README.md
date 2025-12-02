Project: Excel Data Inspection and Visualization Tool

This project provides a command-line tool for loading Excel datasets, performing basic data preparation, 
generating descriptive summaries, and creating exploratory data visualizations. 
It is designed for quickly inspecting structured datasets containing property 
repair information or similar tabular data.

Libraries/dependencies: pip install pandas numpy matplotlib openpyxl xlrd



How to execute the different args in the descriptive file

1. Inspect an Excel file with no plots - 
python housing_repair_descriptive_analysis.py -f data.xlsx

2. Show plots - 
python housing_repair_descriptive_analysis.py -f data.xlsx --plot

3. Export processed sheet as CSV - 
python housing_repair_descriptive_analysis.pyy -f data.xlsx --to-csv cleaned/output.csv

4. Load a specific sheet - 
python housing_repair_descriptive_analysis.pyy -f data.xlsx --sheet "RepairData" --plot

5. Handle multi-sheet Excel file - 

Automatically prints and plots each sheet:

python housing_repair_descriptive_analysis.py -f synthetic_property_data.xlsx --plot --to-csv output.csv
