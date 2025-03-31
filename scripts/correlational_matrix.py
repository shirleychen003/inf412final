import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directories if they don't exist
os.makedirs('output/correlational_matrix', exist_ok=True)

# Function to clean neighborhood names for consistent matching
def clean_name(name):
    name = str(name).strip()
    name = name.replace("St. ", "St.")
    name = name.replace("St ", "St.")
    name = name.replace("-East ", "-East")
    name = name.replace("O`Connor", "O'Connor")
    name = name.replace(" - ", "-")
    return name

# Load 2016 data
nbh_data_2016 = pd.read_csv("cleaned_data/2016/cleaned_neighbourhood_income_data_2016.csv")
crime_data_2016 = pd.read_csv("cleaned_data/2016/cleaned_crime_data_2016.csv")

# Load 2021 data
nbh_data_2021 = pd.read_csv("cleaned_data/2021/cleaned_neighbourhood_income_data_2021.csv")
crime_data_2021 = pd.read_csv("cleaned_data/2021/cleaned_crime_data_2021.csv")

# Clean column names and neighborhood names
nbh_data_2016.columns = nbh_data_2016.columns.str.strip()
crime_data_2016.columns = crime_data_2016.columns.str.strip()
nbh_data_2021.columns = nbh_data_2021.columns.str.strip()
crime_data_2021.columns = crime_data_2021.columns.str.strip()

# Apply name cleaning to all datasets
nbh_data_2016['neighbourhood_name'] = nbh_data_2016['neighbourhood_name'].apply(clean_name)
crime_data_2016['AREA_NAME'] = crime_data_2016['AREA_NAME'].apply(clean_name)
nbh_data_2021['neighbourhood_name'] = nbh_data_2021['neighbourhood_name'].apply(clean_name)
crime_data_2021['AREA_NAME'] = crime_data_2021['AREA_NAME'].apply(clean_name)

# Merge datasets for each year
merged_2016 = pd.merge(
    nbh_data_2016, 
    crime_data_2016,
    left_on='neighbourhood_name',
    right_on='AREA_NAME',
    how='inner'
)

merged_2021 = pd.merge(
    nbh_data_2021, 
    crime_data_2021,
    left_on='neighbourhood_name',
    right_on='AREA_NAME',
    how='inner'
)

print(f"Number of matched neighborhoods (2016): {len(merged_2016)}")
print(f"Number of matched neighborhoods (2021): {len(merged_2021)}")

# Define crime types to analyze
crime_types = [
    'ASSAULT',
    'AUTOTHEFT',
    'BIKETHEFT',
    'BREAKENTER',
    'ROBBERY',
    'THEFTFROMMV',
    'THEFTOVER'
]

# Define socioeconomic variables for 2016
socio_vars_2016 = [
    'total_population',
    'low_income_percent',
]

# Define socioeconomic variables for 2021
socio_vars_2021 = [
    'total_population',
    'low_income_percent',
]

# Add additional 2021 socioeconomic variables if available
additional_vars_2021 = [
    'median_income',
    'average_income',
    'median_after_tax_income',
    'average_after_tax_income',
    'government_support_percent'
]

for var in additional_vars_2021:
    if var in merged_2021.columns:
        socio_vars_2021.append(var)

# Define columns for correlation analysis
crime_cols_2016 = [f"{crime}_2016" for crime in crime_types]
crime_cols_2021 = [f"{crime}_2021" for crime in crime_types]

# Add total crime count columns (they don't have year suffix in original data)
if 'TOTAL_CRIME_COUNT' in merged_2016.columns:
    crime_cols_2016.append('TOTAL_CRIME_COUNT')
if 'TOTAL_CRIME_COUNT' in merged_2021.columns:
    crime_cols_2021.append('TOTAL_CRIME_COUNT')

# Ensure numeric data types
for col in crime_cols_2016:
    merged_2016[col] = pd.to_numeric(merged_2016[col], errors='coerce')
for col in socio_vars_2016:
    merged_2016[col] = pd.to_numeric(merged_2016[col], errors='coerce')
    
for col in crime_cols_2021:
    merged_2021[col] = pd.to_numeric(merged_2021[col], errors='coerce')
for col in socio_vars_2021:
    merged_2021[col] = pd.to_numeric(merged_2021[col], errors='coerce')

# 1. Create correlation matrix between crime types for 2016
corr_2016 = merged_2016[crime_cols_2016].corr()

# Create and save 2016 crime correlation matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr_2016, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('Correlation Matrix - Crime Types (2016)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_2016.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create correlation matrix between crime types for 2021
corr_2021 = merged_2021[crime_cols_2021].corr()

# Create and save 2021 crime correlation matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr_2021, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('Correlation Matrix - Crime Types (2021)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_2021.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Calculate correlation changes between crime types
# We'll need to standardize the column names for comparison
corr_2016_std = merged_2016[[col for col in crime_cols_2016 if col != 'TOTAL_CRIME_COUNT']].corr()
corr_2021_std = merged_2021[[col for col in crime_cols_2021 if col != 'TOTAL_CRIME_COUNT']].corr()

# Standardize column names for comparison
corr_2016_clean = corr_2016_std.copy()
corr_2021_clean = corr_2021_std.copy()

# Update the index and column names to remove the year suffix
corr_2016_clean.index = [col.replace('_2016', '') for col in corr_2016_clean.index]
corr_2016_clean.columns = [col.replace('_2016', '') for col in corr_2016_clean.columns]
corr_2021_clean.index = [col.replace('_2021', '') for col in corr_2021_clean.index]
corr_2021_clean.columns = [col.replace('_2021', '') for col in corr_2021_clean.columns]

# Calculate correlation differences
corr_diff = corr_2021_clean - corr_2016_clean

# Create and save correlation change heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_diff, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('Change in Correlation Between Crime Types (2021 - 2016)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_change.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Create correlation matrix between socioeconomic variables and crime for 2016
socio_crime_corr_2016 = pd.DataFrame(index=socio_vars_2016, columns=crime_cols_2016)

for socio_var in socio_vars_2016:
    for crime_col in crime_cols_2016:
        socio_crime_corr_2016.loc[socio_var, crime_col] = merged_2016[socio_var].corr(merged_2016[crime_col])

# Ensure numeric data
socio_crime_corr_2016 = socio_crime_corr_2016.astype(float)

# Create and save 2016 socioeconomic-crime correlation heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(socio_crime_corr_2016, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=False)
plt.title('Correlations: Socioeconomic Factors vs Crime Types (2016)', fontsize=16)
plt.xlabel('Crime Types', fontsize=12)
plt.ylabel('Socioeconomic Variables', fontsize=12)
plt.tight_layout()
plt.savefig('output/correlational_matrix/socio_crime_correlation_2016.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Create correlation matrix between socioeconomic variables and crime for 2021
socio_crime_corr_2021 = pd.DataFrame(index=socio_vars_2021, columns=crime_cols_2021)

for socio_var in socio_vars_2021:
    for crime_col in crime_cols_2021:
        socio_crime_corr_2021.loc[socio_var, crime_col] = merged_2021[socio_var].corr(merged_2021[crime_col])

# Ensure numeric data
socio_crime_corr_2021 = socio_crime_corr_2021.astype(float)

# Create and save 2021 socioeconomic-crime correlation heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(socio_crime_corr_2021, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=False)
plt.title('Correlations: Socioeconomic Factors vs Crime Types (2021)', fontsize=16)
plt.xlabel('Crime Types', fontsize=12)
plt.ylabel('Socioeconomic Variables', fontsize=12)
plt.tight_layout()
plt.savefig('output/correlational_matrix/socio_crime_correlation_2021.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Calculate changes in socioeconomic-crime correlations between 2016 and 2021
# First create comparable dataframes with matching indices and columns
common_socio_vars = list(set(socio_vars_2016) & set(socio_vars_2021))

# Create standardized dataframes for comparison
socio_crime_cols_2016 = [col for col in crime_cols_2016 if col != 'TOTAL_CRIME_COUNT']
socio_crime_cols_2021 = [col for col in crime_cols_2021 if col != 'TOTAL_CRIME_COUNT']

socio_crime_corr_2016_std = pd.DataFrame(index=socio_vars_2016, columns=socio_crime_cols_2016)
socio_crime_corr_2021_std = pd.DataFrame(index=socio_vars_2021, columns=socio_crime_cols_2021)

for socio_var in socio_vars_2016:
    for crime_col in socio_crime_cols_2016:
        socio_crime_corr_2016_std.loc[socio_var, crime_col] = merged_2016[socio_var].corr(merged_2016[crime_col])

for socio_var in socio_vars_2021:
    for crime_col in socio_crime_cols_2021:
        socio_crime_corr_2021_std.loc[socio_var, crime_col] = merged_2021[socio_var].corr(merged_2021[crime_col])

# Clean column names for comparison
socio_crime_corr_2016_clean = socio_crime_corr_2016_std.copy()
socio_crime_corr_2021_clean = socio_crime_corr_2021_std.copy()

socio_crime_corr_2016_clean.columns = [col.replace('_2016', '') for col in socio_crime_corr_2016_clean.columns]
socio_crime_corr_2021_clean.columns = [col.replace('_2021', '') for col in socio_crime_corr_2021_clean.columns]

# Filter for common variables and crime types
socio_crime_corr_2016_clean = socio_crime_corr_2016_clean.loc[common_socio_vars]
socio_crime_corr_2021_clean = socio_crime_corr_2021_clean.loc[common_socio_vars]

# Calculate changes
socio_crime_corr_change = socio_crime_corr_2021_clean - socio_crime_corr_2016_clean

# Ensure numeric data
socio_crime_corr_change = socio_crime_corr_change.astype(float)

# Create and save socioeconomic-crime correlation change heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(socio_crime_corr_change, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=False)
plt.title('Changes in Correlation: Socioeconomic Factors vs Crime (2021 - 2016)', fontsize=16)
plt.xlabel('Crime Types', fontsize=12)
plt.ylabel('Socioeconomic Variables', fontsize=12)
plt.tight_layout()
plt.savefig('output/correlational_matrix/socio_crime_correlation_change.png', dpi=300, bbox_inches='tight')
plt.close()

# Save correlation matrices to CSV
corr_2016.to_csv('output/correlational_matrix/correlation_matrix_2016.csv')
corr_2021.to_csv('output/correlational_matrix/correlation_matrix_2021.csv')
corr_diff.to_csv('output/correlational_matrix/correlation_change_matrix.csv')
socio_crime_corr_2016.to_csv('output/correlational_matrix/socio_crime_correlation_2016.csv')
socio_crime_corr_2021.to_csv('output/correlational_matrix/socio_crime_correlation_2021.csv')
socio_crime_corr_change.to_csv('output/correlational_matrix/socio_crime_correlation_change.csv')

# Print summary of strongest correlation changes between crime types
print("\nStrongest correlation changes between crime types (|diff| > 0.1):")
for i in range(len(crime_types)):
    for j in range(i+1, len(crime_types)):
        diff = corr_diff.iloc[i, j]
        if abs(diff) > 0.1:
            crime1 = crime_types[i]
            crime2 = crime_types[j]
            corr_2016_val = corr_2016_clean.iloc[i, j]
            corr_2021_val = corr_2021_clean.iloc[i, j]
            print(f"{crime1} vs {crime2}: {corr_2016_val:.3f} (2016) -> {corr_2021_val:.3f} (2021), change: {diff:.3f}")

# Print summary of correlations between socioeconomic factors and crime
print("\nTop socioeconomic predictors of crime in 2016:")
for socio_var in socio_vars_2016:
    top_crime = socio_crime_corr_2016.loc[socio_var].abs().idxmax()
    corr_val = socio_crime_corr_2016.loc[socio_var, top_crime]
    print(f"{socio_var} correlates most strongly with {top_crime}: {corr_val:.3f}")

print("\nTop socioeconomic predictors of crime in 2021:")
for socio_var in socio_vars_2021:
    top_crime = socio_crime_corr_2021.loc[socio_var].abs().idxmax()
    corr_val = socio_crime_corr_2021.loc[socio_var, top_crime]
    print(f"{socio_var} correlates most strongly with {top_crime}: {corr_val:.3f}")

# Calculate and print average correlation strength
print("\nAverage correlation strength between crime types:")
print(f"2016: {np.abs(corr_2016.values[np.triu_indices_from(corr_2016.values, k=1)]).mean():.3f}")
print(f"2021: {np.abs(corr_2021.values[np.triu_indices_from(corr_2021.values, k=1)]).mean():.3f}")
print(f"Change: {np.abs(corr_2021.values[np.triu_indices_from(corr_2021.values, k=1)]).mean() - np.abs(corr_2016.values[np.triu_indices_from(corr_2016.values, k=1)]).mean():.3f}")

print("\nCorrelational matrix analysis complete - all files saved to output/correlational_matrix directory") 