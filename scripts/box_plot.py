import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read and merge the cleaned data
neighborhood_data = pd.read_csv("cleaned_data/cleaned_neighbourhood_data.csv")
crime_data = pd.read_csv("cleaned_data/cleaned_crime_data.csv")

# Clean column names and neighborhood names
def clean_name(name):
    name = name.strip()
    name = name.replace("St. ", "St.")
    name = name.replace("St ", "St.")
    name = name.replace("-East ", "-East")
    name = name.replace("O`Connor", "O'Connor")
    name = name.replace(" - ", "-")
    return name

neighborhood_data.columns = neighborhood_data.columns.str.strip()
crime_data.columns = crime_data.columns.str.strip()
neighborhood_data['Neighbourhood Name'] = neighborhood_data['Neighbourhood Name'].apply(clean_name)
crime_data['AREA_NAME'] = crime_data['AREA_NAME'].apply(clean_name)

# Merge datasets
merged_df = pd.merge(neighborhood_data, crime_data,
                    left_on='Neighbourhood Name',
                    right_on='AREA_NAME',
                    how='inner')

# Create income groups using quantiles
merged_df['Income_Group'] = pd.qcut(merged_df['Median_Income_2020'], 
                                  q=5, 
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

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

# Create box plots for counts
plt.figure(figsize=(20, 15))
for i, (count_code, rate_code) in enumerate(crime_types.items(), 1):
    plt.subplot(3, 3, i)
    
    # Create box plot for counts
    sns.boxplot(data=merged_df,
                x='Income_Group',
                y=count_code,
                hue='Income_Group',
                legend=False)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel('Income Group')
    plt.ylabel('Number of Crimes')
    plt.title(f'{count_code.replace("_2020", "")} Counts by Income Group')

# Adjust layout
plt.tight_layout()

# Save the count plots
plt.savefig('output/box_plot/crime_counts_by_income.png', dpi=300, bbox_inches='tight')

# Create box plots for rates
plt.figure(figsize=(20, 15))
for i, (count_code, rate_code) in enumerate(crime_types.items(), 1):
    plt.subplot(3, 3, i)
    
    # Create box plot for rates
    sns.boxplot(data=merged_df,
                x='Income_Group',
                y=rate_code,
                hue='Income_Group',
                legend=False)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel('Income Group')
    plt.ylabel('Rate per 100,000 population')
    plt.title(f'{rate_code.replace("_RATE_2020", "")} Rates by Income Group')

# Adjust layout
plt.tight_layout()

# Save the rate plots
plt.savefig('output/box_plot/crime_rates_by_income.png', dpi=300, bbox_inches='tight')

# Calculate summary statistics for both counts and rates
count_stats = pd.DataFrame()
rate_stats = pd.DataFrame()

for count_code, rate_code in crime_types.items():
    # Calculate mean counts by income group
    count_means = merged_df.groupby('Income_Group', observed=True)[count_code].mean()
    count_stats[count_code.replace('_2020', '')] = count_means
    
    # Calculate mean rates by income group
    rate_means = merged_df.groupby('Income_Group', observed=True)[rate_code].mean()
    rate_stats[rate_code.replace('_RATE_2020', '')] = rate_means

# Save summary statistics
count_stats.to_csv('output/box_plot/crime_counts_summary.csv')
rate_stats.to_csv('output/box_plot/crime_rates_summary.csv')

# Print summary statistics
print("\nMean Crime Counts by Income Group:")
print(count_stats.round(2))
print("\nMean Crime Rates by Income Group:")
print(rate_stats.round(2))

# Calculate percentage differences between highest and lowest income groups
print("\nPercentage differences between Very High and Very Low Income Groups:")
for count_code, rate_code in crime_types.items():
    count_high = merged_df[merged_df['Income_Group'] == 'Very High'][count_code].mean()
    count_low = merged_df[merged_df['Income_Group'] == 'Very Low'][count_code].mean()
    rate_high = merged_df[merged_df['Income_Group'] == 'Very High'][rate_code].mean()
    rate_low = merged_df[merged_df['Income_Group'] == 'Very Low'][rate_code].mean()
    
    count_diff = ((count_high - count_low) / count_low) * 100
    rate_diff = ((rate_high - rate_low) / rate_low) * 100
    
    print(f"\n{count_code.replace('_2020', '')}:")
    print(f"Count difference: {count_diff:.1f}%")
    print(f"Rate difference: {rate_diff:.1f}%")

# Calculate correlations for both counts and rates
count_correlations = []
rate_correlations = []

for count_code, rate_code in crime_types.items():
    count_corr = merged_df['Median_Income_2020'].corr(merged_df[count_code])
    rate_corr = merged_df['Median_Income_2020'].corr(merged_df[rate_code])
    
    count_correlations.append({
        'Crime Type': count_code.replace('_2020', ''),
        'Correlation with Income': count_corr
    })
    
    rate_correlations.append({
        'Crime Type': rate_code.replace('_RATE_2020', ''),
        'Correlation with Income': rate_corr
    })

# Create and save correlation DataFrames
count_corr_df = pd.DataFrame(count_correlations)
rate_corr_df = pd.DataFrame(rate_correlations)

# Sort by absolute correlation
count_corr_df['Abs_Correlation'] = count_corr_df['Correlation with Income'].abs()
rate_corr_df['Abs_Correlation'] = rate_corr_df['Correlation with Income'].abs()

count_corr_df = count_corr_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
rate_corr_df = rate_corr_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)

# Save correlations
count_corr_df.to_csv('output/box_plot/income_crime_count_correlations.csv', index=False)
rate_corr_df.to_csv('output/box_plot/income_crime_rate_correlations.csv', index=False)

print("\nCorrelations between Income and Crime Counts:")
print(count_corr_df.round(3))
print("\nCorrelations between Income and Crime Rates:")
print(rate_corr_df.round(3))
