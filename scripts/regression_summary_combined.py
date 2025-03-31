import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
from matplotlib.gridspec import GridSpec

# Create output directories if they don't exist
os.makedirs('output', exist_ok=True)

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

# Define crime types to analyze (both counts and rates)
crime_types = [
    ('ASSAULT', 'ASSAULT_RATE'),
    ('AUTOTHEFT', 'AUTOTHEFT_RATE'),
    ('BIKETHEFT', 'BIKETHEFT_RATE'),
    ('BREAKENTER', 'BREAKENTER_RATE'),
    ('ROBBERY', 'ROBBERY_RATE'),
    ('THEFTFROMMV', 'THEFTFROMMV_RATE'),
    ('THEFTOVER', 'THEFTOVER_RATE'),
    ('TOTAL_CRIME_COUNT', None)
]

# Define independent variables
# 2016 has fewer variables available, so we'll use what's common between both datasets
independent_vars_2016 = ['total_population', 'low_income_percent']
independent_vars_2021 = ['total_population', 'low_income_percent']

# Lists to store regression results
results_2016 = []
results_2021 = []

# Perform regression analysis for 2016 data
for count_type, rate_type in crime_types:
    # For 2016
    X_2016 = merged_2016[independent_vars_2016]
    
    # Crime count regression
    count_col_2016 = f"{count_type}_2016" if count_type != 'TOTAL_CRIME_COUNT' else 'TOTAL_CRIME_COUNT'
    y_count_2016 = merged_2016[count_col_2016]
    
    count_model_2016 = LinearRegression()
    count_model_2016.fit(X_2016, y_count_2016)
    count_r2_2016 = count_model_2016.score(X_2016, y_count_2016)
    
    # Store results
    count_result_2016 = {
        'Year': 2016,
        'Crime_Type': count_type,
        'Model_Type': 'Count',
        'R_squared': count_r2_2016,
        'Intercept': count_model_2016.intercept_
    }
    
    # Add coefficients
    for var, coef in zip(independent_vars_2016, count_model_2016.coef_):
        count_result_2016[f'{var}_coef'] = coef
    
    results_2016.append(count_result_2016)
    
    # Rate regression (if available)
    if rate_type:
        rate_col_2016 = f"{rate_type}_2016"
        if rate_col_2016 in merged_2016.columns:
            y_rate_2016 = merged_2016[rate_col_2016]
            
            rate_model_2016 = LinearRegression()
            rate_model_2016.fit(X_2016, y_rate_2016)
            rate_r2_2016 = rate_model_2016.score(X_2016, y_rate_2016)
            
            # Store results
            rate_result_2016 = {
                'Year': 2016,
                'Crime_Type': count_type,
                'Model_Type': 'Rate',
                'R_squared': rate_r2_2016,
                'Intercept': rate_model_2016.intercept_
            }
            
            # Add coefficients
            for var, coef in zip(independent_vars_2016, rate_model_2016.coef_):
                rate_result_2016[f'{var}_coef'] = coef
            
            results_2016.append(rate_result_2016)

# Perform regression analysis for 2021 data
for count_type, rate_type in crime_types:
    # For 2021
    X_2021 = merged_2021[independent_vars_2021]
    
    # Crime count regression
    count_col_2021 = f"{count_type}_2021" if count_type != 'TOTAL_CRIME_COUNT' else 'TOTAL_CRIME_COUNT'
    y_count_2021 = merged_2021[count_col_2021]
    
    count_model_2021 = LinearRegression()
    count_model_2021.fit(X_2021, y_count_2021)
    count_r2_2021 = count_model_2021.score(X_2021, y_count_2021)
    
    # Store results
    count_result_2021 = {
        'Year': 2021,
        'Crime_Type': count_type,
        'Model_Type': 'Count',
        'R_squared': count_r2_2021,
        'Intercept': count_model_2021.intercept_
    }
    
    # Add coefficients
    for var, coef in zip(independent_vars_2021, count_model_2021.coef_):
        count_result_2021[f'{var}_coef'] = coef
    
    results_2021.append(count_result_2021)
    
    # Rate regression (if available)
    if rate_type:
        rate_col_2021 = f"{rate_type}_2021"
        if rate_col_2021 in merged_2021.columns:
            y_rate_2021 = merged_2021[rate_col_2021]
            
            rate_model_2021 = LinearRegression()
            rate_model_2021.fit(X_2021, y_rate_2021)
            rate_r2_2021 = rate_model_2021.score(X_2021, y_rate_2021)
            
            # Store results
            rate_result_2021 = {
                'Year': 2021,
                'Crime_Type': count_type,
                'Model_Type': 'Rate',
                'R_squared': rate_r2_2021,
                'Intercept': rate_model_2021.intercept_
            }
            
            # Add coefficients
            for var, coef in zip(independent_vars_2021, rate_model_2021.coef_):
                rate_result_2021[f'{var}_coef'] = coef
            
            results_2021.append(rate_result_2021)

# Combine all results
all_results = results_2016 + results_2021
regression_summary = pd.DataFrame(all_results)

# Save the regression summary
regression_summary.to_csv('regression_summary_2016_2021.csv', index=False)

# Print summary statistics
print("\nRegression Summary Statistics:")
print(f"Total number of models: {len(regression_summary)}")
print("\n2016 Average R-squared values:")
print(f"Count models: {regression_summary[(regression_summary['Year'] == 2016) & (regression_summary['Model_Type'] == 'Count')]['R_squared'].mean():.3f}")
print(f"Rate models: {regression_summary[(regression_summary['Year'] == 2016) & (regression_summary['Model_Type'] == 'Rate')]['R_squared'].mean():.3f}")
print("\n2021 Average R-squared values:")
print(f"Count models: {regression_summary[(regression_summary['Year'] == 2021) & (regression_summary['Model_Type'] == 'Count')]['R_squared'].mean():.3f}")
print(f"Rate models: {regression_summary[(regression_summary['Year'] == 2021) & (regression_summary['Model_Type'] == 'Rate')]['R_squared'].mean():.3f}")

# Create visualizations
plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2)

# Income vs Crime plot for both years
ax1 = plt.subplot(gs[0, :])
sns.scatterplot(data=merged_2016, x='low_income_percent', y='TOTAL_CRIME_COUNT', 
               label='2016', alpha=0.7, color='blue', ax=ax1)
sns.regplot(data=merged_2016, x='low_income_percent', y='TOTAL_CRIME_COUNT', 
           scatter=False, color='blue', line_kws={'linestyle':'--'}, ax=ax1)

sns.scatterplot(data=merged_2021, x='low_income_percent', y='TOTAL_CRIME_COUNT', 
               label='2021', alpha=0.7, color='red', ax=ax1)
sns.regplot(data=merged_2021, x='low_income_percent', y='TOTAL_CRIME_COUNT', 
           scatter=False, color='red', ax=ax1)

ax1.set_title('Low Income Percentage vs Total Crime Count (2016 vs 2021)', fontsize=14)
ax1.set_xlabel('Low Income Percentage (%)', fontsize=12)
ax1.set_ylabel('Total Crime Count', fontsize=12)
ax1.legend()

# R-squared comparison for crime counts
ax2 = plt.subplot(gs[1, 0])
count_r2 = regression_summary[regression_summary['Model_Type'] == 'Count'].pivot(
    index='Crime_Type', columns='Year', values='R_squared')

count_r2.plot(kind='bar', ax=ax2)
ax2.set_title('R-squared Comparison: Crime Count Models', fontsize=14)
ax2.set_xlabel('Crime Type', fontsize=12)
ax2.set_ylabel('R-squared Value', fontsize=12)
ax2.set_ylim(0, 1)  # R-squared range 0-1
ax2.legend(title='Year')

# R-squared comparison for crime rates
ax3 = plt.subplot(gs[1, 1])
rate_r2 = regression_summary[regression_summary['Model_Type'] == 'Rate'].pivot(
    index='Crime_Type', columns='Year', values='R_squared')

rate_r2.plot(kind='bar', ax=ax3)
ax3.set_title('R-squared Comparison: Crime Rate Models', fontsize=14)
ax3.set_xlabel('Crime Type', fontsize=12)
ax3.set_ylabel('R-squared Value', fontsize=12)
ax3.set_ylim(0, 1)  # R-squared range 0-1
ax3.legend(title='Year')

plt.tight_layout()
plt.savefig('regression_analysis_2016_2021.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a correlation analysis visualization
# Compute the coefficient change for low income percentage
coef_changes = []

for crime_type in regression_summary['Crime_Type'].unique():
    for model_type in ['Count', 'Rate']:
        # Get 2016 coefficient
        coef_2016 = regression_summary[
            (regression_summary['Year'] == 2016) & 
            (regression_summary['Crime_Type'] == crime_type) & 
            (regression_summary['Model_Type'] == model_type)
        ]['low_income_percent_coef'].values
        
        # Get 2021 coefficient
        coef_2021 = regression_summary[
            (regression_summary['Year'] == 2021) & 
            (regression_summary['Crime_Type'] == crime_type) & 
            (regression_summary['Model_Type'] == model_type)
        ]['low_income_percent_coef'].values
        
        # Only proceed if we have both coefficients
        if len(coef_2016) > 0 and len(coef_2021) > 0:
            coef_changes.append({
                'Crime_Type': crime_type,
                'Model_Type': model_type,
                'Coef_2016': coef_2016[0],
                'Coef_2021': coef_2021[0],
                'Change': coef_2021[0] - coef_2016[0]
            })

# Convert to DataFrame and plot
coef_change_df = pd.DataFrame(coef_changes)

plt.figure(figsize=(12, 6))
sns.barplot(data=coef_change_df, x='Crime_Type', y='Change', hue='Model_Type')
plt.title('Change in Low Income Impact on Crime (2016 to 2021)', fontsize=14)
plt.xlabel('Crime Type', fontsize=12)
plt.ylabel('Change in Coefficient', fontsize=12)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('income_coefficient_changes.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Files created:")
print("- regression_summary_2016_2021.csv")
print("- regression_analysis_2016_2021.png")
print("- income_coefficient_changes.png") 