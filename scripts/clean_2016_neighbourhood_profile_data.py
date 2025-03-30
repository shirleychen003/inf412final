import pandas as pd
import numpy as np
import os

# Ensure cleaned_data directory exists
os.makedirs('../cleaned_data', exist_ok=True)

# Load the data from CSV file
print("Loading raw 2016 data...")
raw_data = pd.read_csv('../raw_data/neighbourhood-profiles-2016-140-model.csv')

print(f"Raw data shape: {raw_data.shape}")

# Extract the column names after the first few columns which are neighborhood names
neighbourhood_names = raw_data.columns[5:].tolist()  # Skip metadata columns

# Create a new DataFrame for cleaned data
print("Creating cleaned data frame...")
cleaned_data = pd.DataFrame({'neighbourhood_name': neighbourhood_names})

# Remove rows with invalid neighborhood names (like header rows that got included)
# Filter out any non-neighborhood names from the columns
invalid_names = ['Category', 'Topic', 'Data Source', 'Characteristic', 'City of Toronto']
cleaned_data = cleaned_data[~cleaned_data['neighbourhood_name'].isin(invalid_names)]

# Reset index after filtering
cleaned_data = cleaned_data.reset_index(drop=True)

# Extract neighbourhood IDs and match properly
print("Extracting neighbourhood IDs...")
neighbourhood_id_row = raw_data[raw_data['Characteristic'] == 'Neighbourhood Number']
tsns_row = raw_data[raw_data['Characteristic'] == 'TSNS2020 Designation']

if not neighbourhood_id_row.empty and not tsns_row.empty:
    # Create a mapping dataframe with both neighbourhood names and IDs
    mapping_df = pd.DataFrame({
        'neighbourhood_name': raw_data.columns[5:],
        'neighbourhood_id': neighbourhood_id_row.iloc[0, 5:].values,
        'tsns_designation': tsns_row.iloc[0, 5:].values
    })
    
    # Filter out invalid names
    mapping_df = mapping_df[~mapping_df['neighbourhood_name'].isin(invalid_names)]
    
    # Reset index
    mapping_df = mapping_df.reset_index(drop=True)
    
    # Add data to cleaned_data
    cleaned_data['neighbourhood_id'] = mapping_df['neighbourhood_id']
    cleaned_data['tsns_designation'] = mapping_df['tsns_designation']
else:
    print("Warning: Could not find necessary rows for mapping neighbourhood IDs and TSNS designation")

# Extract demographic data and low income metrics and align them properly with neighborhoods
print("Extracting demographic and income metrics...")
population_row = raw_data[raw_data['Characteristic'] == 'Population, 2016']
low_income_row = raw_data[raw_data['Characteristic'] == 'Prevalence of low income based on the Low-income measure, after tax (LIM-AT) (%)']

if not population_row.empty and not low_income_row.empty:
    # Create function to clean and convert population values (handles commas in numbers)
    def clean_population(val):
        if isinstance(val, str):
            # Remove commas and convert to numeric
            return pd.to_numeric(val.replace(',', ''), errors='coerce')
        return pd.to_numeric(val, errors='coerce')
    
    # Create a dataframe with metrics for each neighborhood
    metrics_df = pd.DataFrame({
        'neighbourhood_name': raw_data.columns[5:],
        'total_population': population_row.iloc[0, 5:].apply(clean_population),
        'low_income_percent': pd.to_numeric(low_income_row.iloc[0, 5:].values, errors='coerce')
    })
    
    # Filter out invalid names
    metrics_df = metrics_df[~metrics_df['neighbourhood_name'].isin(invalid_names)]
    
    # Reset index
    metrics_df = metrics_df.reset_index(drop=True)
    
    # Add data to cleaned_data
    cleaned_data['total_population'] = metrics_df['total_population']
    cleaned_data['low_income_percent'] = metrics_df['low_income_percent']
else:
    print("Warning: Could not find necessary rows for demographic and income metrics")

# Create binary indicator for NIA (Neighbourhood Improvement Area)
if 'tsns_designation' in cleaned_data.columns:
    # First fill NaN values with empty string, then check for NIA
    cleaned_data['tsns_designation'] = cleaned_data['tsns_designation'].fillna('')
    cleaned_data['is_improvement_area'] = cleaned_data['tsns_designation'].str.contains('NIA', case=False).astype(int)

# Fill missing values with 0 or appropriate value
cleaned_data = cleaned_data.fillna(0)

# Print sample of the data to verify
print("\nSample of processed data:")
print(cleaned_data.head())

# Print summary statistics
print("\nSummary statistics for key metrics:")
numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
if len(numeric_columns) > 0:
    print(cleaned_data[numeric_columns].describe().T[['mean', 'min', 'max']])
else:
    print("No numeric columns found in cleaned data.")

# Export cleaned data to CSV
print("\nSaving cleaned data...")
cleaned_data.to_csv('../cleaned_data/2016/cleaned_neighbourhood_income_data_2016.csv', index=False)

print(f"Done! Cleaned data saved for {len(cleaned_data)} neighbourhoods.")
