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

# Define crime types to analyze
crime_types = {
    'ASSAULT_RATE_2020': 'Assault Rate',
    'AUTOTHEFT_RATE_2020': 'Auto Theft Rate',
    'BIKETHEFT_RATE_2020': 'Bike Theft Rate',
    'BREAKENTER_RATE_2020': 'Break & Enter Rate',
    'ROBBERY_RATE_2020': 'Robbery Rate',
    'THEFTFROMMV_RATE_2020': 'Theft from Vehicle Rate',
    'THEFTOVER_RATE_2020': 'Theft Over Rate'
}

# Create box plots
plt.figure(figsize=(20, 15))

for i, (crime_code, crime_name) in enumerate(crime_types.items(), 1):
    plt.subplot(3, 3, i)
    
    # Create box plot
    sns.boxplot(data=merged_df,
                x='Income_Group',
                y=crime_code,
                hue='Income_Group',
                legend=False)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel('Income Group')
    plt.ylabel('Rate per 100,000 population')
    plt.title(f'{crime_name} by Income Group')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('cleaned_data/box_plot/crime_rates_by_income.png', dpi=300, bbox_inches='tight')

# Calculate summary statistics
summary_stats = pd.DataFrame()

for crime_code, crime_name in crime_types.items():
    # Calculate mean rates by income group
    means = merged_df.groupby('Income_Group', observed=True)[crime_code].mean()
    summary_stats[crime_name] = means

# Save summary statistics
summary_stats.to_csv('cleaned_data/box_plot/crime_rates_summary.csv')

# Print summary statistics
print("\nMean Crime Rates by Income Group:")
print(summary_stats.round(2))

# Calculate percentage differences between highest and lowest income groups
print("\nPercentage difference in crime rates (Very High vs Very Low Income):")
for crime_name in summary_stats.columns:
    very_high = summary_stats.loc['Very High', crime_name]
    very_low = summary_stats.loc['Very Low', crime_name]
    pct_diff = ((very_high - very_low) / very_low) * 100
    print(f"{crime_name}: {pct_diff:.1f}%")

# Additional analysis: Correlation between income and crime rates
correlations_list = []

for crime_code, crime_name in crime_types.items():
    correlation = merged_df['Median_Income_2020'].corr(merged_df[crime_code])
    correlations_list.append({
        'Crime Type': crime_name,
        'Correlation with Income': correlation
    })

correlations = pd.DataFrame(correlations_list)

# Sort by absolute correlation
correlations['Abs_Correlation'] = correlations['Correlation with Income'].abs()
correlations = correlations.sort_values('Abs_Correlation', ascending=False)
correlations = correlations.drop('Abs_Correlation', axis=1)

print("\nCorrelations between Income and Crime Rates:")
print(correlations.round(3))

# Save correlations
correlations.to_csv('cleaned_data/box_plot/income_crime_correlations.csv', index=False)
