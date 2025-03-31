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

# Define columns for correlation analysis
crime_cols_2016 = [f"{crime}_2016" for crime in crime_types]
crime_cols_2021 = [f"{crime}_2021" for crime in crime_types]

# 1. Create correlation matrix for 2016
corr_2016 = merged_2016[crime_cols_2016].corr()

# Create and save 2016 correlation matrix plot
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_2016, dtype=bool))
sns.heatmap(corr_2016, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True,
            mask=mask)
plt.title('Correlation Matrix - Crime Types (2016)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_2016.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create correlation matrix for 2021
corr_2021 = merged_2021[crime_cols_2021].corr()

# Create and save 2021 correlation matrix plot
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_2021, dtype=bool))
sns.heatmap(corr_2021, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True,
            mask=mask)
plt.title('Correlation Matrix - Crime Types (2021)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_2021.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Calculate correlation changes
# Create clean labels for the correlation matrices
corr_2016_clean = corr_2016.copy()
corr_2021_clean = corr_2021.copy()

# Update the index and column names to remove the year suffix
corr_2016_clean.index = [col.replace('_2016', '') for col in corr_2016_clean.index]
corr_2016_clean.columns = [col.replace('_2016', '') for col in corr_2016_clean.columns]
corr_2021_clean.index = [col.replace('_2021', '') for col in corr_2021_clean.index]
corr_2021_clean.columns = [col.replace('_2021', '') for col in corr_2021_clean.columns]

# Calculate correlation differences
corr_diff = corr_2021_clean - corr_2016_clean

# Create and save correlation change heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_diff, dtype=bool))
sns.heatmap(corr_diff, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True,
            mask=mask)
plt.title('Change in Correlation Between Crime Types (2021 - 2016)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_change.png', dpi=300, bbox_inches='tight')
plt.close()

# Save correlation matrices to CSV
corr_2016.to_csv('output/correlational_matrix/correlation_matrix_2016.csv')
corr_2021.to_csv('output/correlational_matrix/correlation_matrix_2021.csv')
corr_diff.to_csv('output/correlational_matrix/correlation_change_matrix.csv')

# Print summary of strongest correlation changes
print("\nStrongest correlation changes (|diff| > 0.1):")
for i in range(len(crime_types)):
    for j in range(i+1, len(crime_types)):
        diff = corr_diff.iloc[i, j]
        if abs(diff) > 0.1:
            crime1 = crime_types[i]
            crime2 = crime_types[j]
            corr_2016_val = corr_2016_clean.iloc[i, j]
            corr_2021_val = corr_2021_clean.iloc[i, j]
            print(f"{crime1} vs {crime2}: {corr_2016_val:.3f} (2016) -> {corr_2021_val:.3f} (2021), change: {diff:.3f}")

# Calculate and print average correlation strength
print("\nAverage correlation strength:")
print(f"2016: {np.abs(corr_2016.values[np.triu_indices_from(corr_2016.values, k=1)]).mean():.3f}")
print(f"2021: {np.abs(corr_2021.values[np.triu_indices_from(corr_2021.values, k=1)]).mean():.3f}")
print(f"Change: {np.abs(corr_2021.values[np.triu_indices_from(corr_2021.values, k=1)]).mean() - np.abs(corr_2016.values[np.triu_indices_from(corr_2016.values, k=1)]).mean():.3f}")

print("\nCorrelational matrix analysis complete - all files saved to output/correlational_matrix directory")
