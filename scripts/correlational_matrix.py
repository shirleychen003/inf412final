import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the cleaned data files
neighborhood_data = pd.read_csv("cleaned_data/cleaned_neighbourhood_data.csv")
crime_data = pd.read_csv("cleaned_data/cleaned_crime_data.csv")

# Clean column names by stripping whitespace
neighborhood_data.columns = neighborhood_data.columns.str.strip()
crime_data.columns = crime_data.columns.str.strip()

# Clean neighborhood names
def clean_name(name):
    # Strip whitespace and standardize some common differences
    name = name.strip()
    name = name.replace("St. ", "St.")  # Standardize St. vs St
    name = name.replace("St ", "St.")
    name = name.replace("-East ", "-East")  # Fix specific cases
    name = name.replace("O`Connor", "O'Connor")  # Standardize apostrophes
    name = name.replace(" - ", "-")  # Standardize hyphens
    return name

# Apply cleaning to neighborhood names
neighborhood_data['Neighbourhood Name'] = neighborhood_data['Neighbourhood Name'].apply(clean_name)
crime_data['AREA_NAME'] = crime_data['AREA_NAME'].apply(clean_name)

# Print unique neighborhood names from both datasets to debug
print("Neighborhood names from demographic data:")
print(neighborhood_data['Neighbourhood Name'].tolist())
print("\nNeighborhood names from crime data:")
print(crime_data['AREA_NAME'].tolist())

# Check for exact matches
matches = set(neighborhood_data['Neighbourhood Name']) & set(crime_data['AREA_NAME'])
print("\nNumber of exact matches:", len(matches))
print("Example matches:", list(matches)[:5] if matches else "No matches")

# Print some example non-matches
print("\nExample neighborhood names that don't match:")
print("From demographic data:", list(set(neighborhood_data['Neighbourhood Name']) - set(crime_data['AREA_NAME']))[:5])
print("From crime data:", list(set(crime_data['AREA_NAME']) - set(neighborhood_data['Neighbourhood Name']))[:5])

# Merge the datasets on neighborhood name
merged_df = pd.merge(neighborhood_data, crime_data, 
                    left_on='Neighbourhood Name', 
                    right_on='AREA_NAME', 
                    how='inner')

print(f"Successfully merged {len(merged_df)} neighborhoods")

# Define crime types to analyze (both counts and rates)
crime_types = {
    'ASSAULT_2020': 'ASSAULT_RATE_2020',
    'AUTOTHEFT_2020': 'AUTOTHEFT_RATE_2020',
    'BIKETHEFT_2020': 'BIKETHEFT_RATE_2020',
    'BREAKENTER_2020': 'BREAKENTER_RATE_2020',
    'ROBBERY_2020': 'ROBBERY_RATE_2020',
    'THEFTFROMMV_2020': 'THEFTFROMMV_RATE_2020',
    'THEFTOVER_2020': 'THEFTOVER_RATE_2020'
}

# Create correlation matrix for counts
count_columns = list(crime_types.keys())
count_corr = merged_df[count_columns].corr()

# Create correlation matrix for rates
rate_columns = list(crime_types.values())
rate_corr = merged_df[rate_columns].corr()

# Create and save count correlation matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(count_corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('Correlation Matrix - Crime Counts')
plt.tight_layout()
plt.savefig('output/correlational_matrix/crime_counts_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Create and save rate correlation matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(rate_corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('Correlation Matrix - Crime Rates')
plt.tight_layout()
plt.savefig('output/correlational_matrix/crime_rates_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Save correlation matrices to CSV
count_corr.to_csv('output/correlational_matrix/crime_counts_correlation_matrix.csv')
rate_corr.to_csv('output/correlational_matrix/crime_rates_correlation_matrix.csv')

# Print summary of strongest correlations
print("\nStrongest correlations in crime counts (|r| > 0.5):")
for i in range(len(count_columns)):
    for j in range(i+1, len(count_columns)):
        corr = count_corr.iloc[i, j]
        if abs(corr) > 0.5:
            print(f"{count_columns[i]} vs {count_columns[j]}: {corr:.3f}")

print("\nStrongest correlations in crime rates (|r| > 0.5):")
for i in range(len(rate_columns)):
    for j in range(i+1, len(rate_columns)):
        corr = rate_corr.iloc[i, j]
        if abs(corr) > 0.5:
            print(f"{rate_columns[i]} vs {rate_columns[j]}: {corr:.3f}")

# Calculate and print average correlations
print("\nAverage correlations:")
print(f"Count correlations: {count_corr.values[np.triu_indices_from(count_corr.values, k=1)].mean():.3f}")
print(f"Rate correlations: {rate_corr.values[np.triu_indices_from(rate_corr.values, k=1)].mean():.3f}")

# Calculate and print correlation differences
print("\nCorrelation differences (Rate - Count):")
for count_col, rate_col in crime_types.items():
    count_corr_with_others = count_corr[count_col].drop(count_col).mean()
    rate_corr_with_others = rate_corr[rate_col].drop(rate_col).mean()
    diff = rate_corr_with_others - count_corr_with_others
    print(f"{count_col}: {diff:.3f}")
