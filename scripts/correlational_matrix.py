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

# Select numeric columns for correlation analysis
numeric_columns = [
    # Demographic indicators
    'Total_Population',
    'Youth_Ratio',
    'Working_Age_Ratio',
    'Senior_Ratio',
    'Average_Age',
    'Median_Age',
    
    # Income indicators
    'Median_Income_2020',
    'Average_Income_2020',
    'Median_After_Tax_Income_2020',
    'Average_After_Tax_Income_2020',
    
    # Crime counts
    'ASSAULT_2020',
    'AUTOTHEFT_2020',
    'BIKETHEFT_2020',
    'BREAKENTER_2020',
    'HOMICIDE_2020',
    'ROBBERY_2020',
    'SHOOTING_2020',
    'THEFTFROMMV_2020',
    'THEFTOVER_2020',
    'TOTAL_CRIME_COUNT',
    
    # Crime rates
    'ASSAULT_RATE_2020',
    'AUTOTHEFT_RATE_2020',
    'BIKETHEFT_RATE_2020',
    'BREAKENTER_RATE_2020',
    'HOMICIDE_RATE_2020',
    'ROBBERY_RATE_2020',
    'SHOOTING_RATE_2020',
    'THEFTFROMMV_RATE_2020',
    'THEFTOVER_RATE_2020'
]

print("\nShape of merged dataframe:", merged_df.shape)
print("\nColumns in merged dataframe:", merged_df.columns.tolist())

# Verify all columns exist in merged dataframe
missing_columns = [col for col in numeric_columns if col not in merged_df.columns]
if missing_columns:
    print("\nWarning: Missing columns:", missing_columns)
    numeric_columns = [col for col in numeric_columns if col in merged_df.columns]

# Calculate correlation matrix
correlation_matrix = merged_df[numeric_columns].corr()

# Create a figure with a larger size
plt.figure(figsize=(20, 16))

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='RdBu',  # Red-Blue colormap
            center=0,     # Center the colormap at 0
            fmt='.2f',   # Format correlation values to 2 decimal places
            square=True,  # Make cells square
            cbar_kws={"shrink": .5})  # Adjust colorbar size

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('cleaned_data/correlation_matrix.png', dpi=300, bbox_inches='tight')

# Print some key findings
print("\nKey Correlations with Total Crime Count:")
total_crime_corr = correlation_matrix['TOTAL_CRIME_COUNT'].sort_values(ascending=False)
print(total_crime_corr)

# Calculate average correlations for demographic and socioeconomic factors with crime rates
demographic_vars = ['Youth_Ratio', 'Working_Age_Ratio', 'Senior_Ratio', 'Average_Age', 'Median_Age']
income_vars = ['Median_Income_2020', 'Average_Income_2020', 'Median_After_Tax_Income_2020', 'Average_After_Tax_Income_2020']
crime_rates = [col for col in numeric_columns if 'RATE' in col]

print("\nAverage correlation between demographic factors and crime rates:")
for demo_var in demographic_vars:
    avg_corr = correlation_matrix.loc[demo_var, crime_rates].mean()
    print(f"{demo_var}: {avg_corr:.3f}")

print("\nAverage correlation between income factors and crime rates:")
for income_var in income_vars:
    avg_corr = correlation_matrix.loc[income_var, crime_rates].mean()
    print(f"{income_var}: {avg_corr:.3f}")

# Save detailed correlations to CSV
correlation_matrix.to_csv('cleaned_data/correlation_matrix.csv')

# Print some example correlations
print("\nStrong correlations (|r| > 0.5):")
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.5:
            strong_correlations.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr
            ))

# Sort by absolute correlation value
strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 10 strongest correlations
for var1, var2, corr in strong_correlations[:10]:
    print(f"{var1} vs {var2}: {corr:.3f}")
