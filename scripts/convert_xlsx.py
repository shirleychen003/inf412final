import pandas as pd
import os

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from scripts)
project_root = os.path.dirname(script_dir)

# Construct absolute paths
excel_file = os.path.join(project_root, "raw_data", "neighbourhood-profiles-2021-158-model.xlsx")
output_file = os.path.join(project_root, "raw_data", "neighbourhood-profiles-2021-158-model.csv")

# Read the Excel file
df = pd.read_excel(excel_file)

# Export to CSV
df.to_csv(output_file, index=False)

print(f"Successfully converted {excel_file} to {output_file}")
