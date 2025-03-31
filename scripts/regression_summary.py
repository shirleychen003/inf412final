import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
from matplotlib.gridspec import GridSpec

# Create output directories if they don't exist
os.makedirs('output', exist_ok=True)

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

# Define crime types to analyze (both counts and rates)
crime_types = {
    'ASSAULT_2020': 'ASSAULT_RATE_2020',
    'AUTOTHEFT_2020': 'AUTOTHEFT_RATE_2020',
    'BIKETHEFT_2020': 'BIKETHEFT_RATE_2020',
    'BREAKENTER_2020': 'BREAKENTER_RATE_2020',
    'ROBBERY_2020': 'ROBBERY_RATE_2020',
    'THEFTFROMMV_2020': 'THEFTFROMMV_RATE_2020',
    'THEFTOVER_2020': 'THEFTOVER_RATE_2020',
    'TOTAL_CRIME_COUNT_2020': 'TOTAL_CRIME_RATE_2020'
}

# Define independent variables
independent_vars = [
    'Total_Population',
    'Youth_Ratio',
    'Working_Age_Ratio',
    'Senior_Ratio',
    'Average_Age',
    'Median_Age',
    'Median_Income_2020',
    'Average_Income_2020',
    'Median_After_Tax_Income_2020',
    'Average_After_Tax_Income_2020'
]

# Create a list for storing all regression results
regression_results = []

# Perform regression analysis for both counts and rates
for crime_type, rate_type in crime_types.items():
    crime_display_name = crime_type.replace('_2020', '')
    rate_display_name = rate_type.replace('_2020', '')
    
    X = merged_df[independent_vars]
    
    # Count regression
    y_count = merged_df[crime_type]
    count_model = LinearRegression()
    count_model.fit(X, y_count)
    count_r2 = count_model.score(X, y_count)
    count_coeffs = count_model.coef_
    
    # Rate regression
    y_rate = merged_df[rate_type]
    rate_model = LinearRegression()
    rate_model.fit(X, y_rate)
    rate_r2 = rate_model.score(X, y_rate)
    rate_coeffs = rate_model.coef_
    
    # Store results from count model
    count_result = {
        'Crime_Type': crime_display_name,
        'Model_Type': 'Count',
        'R_squared': count_r2,
        'Intercept': count_model.intercept_
    }
    
    # Add coefficients
    for var, coef in zip(independent_vars, count_coeffs):
        count_result[f'{var}_coef'] = coef
    
    regression_results.append(count_result)
    
    # Store results from rate model
    rate_result = {
        'Crime_Type': crime_display_name,
        'Model_Type': 'Rate',
        'R_squared': rate_r2,
        'Intercept': rate_model.intercept_
    }
    
    # Add coefficients
    for var, coef in zip(independent_vars, rate_coeffs):
        rate_result[f'{var}_coef'] = coef
    
    regression_results.append(rate_result)

# Convert results to DataFrame and save as CSV
regression_summary = pd.DataFrame(regression_results)
regression_summary.to_csv('regression_summary.csv', index=False)

# Print some basic statistics
print("Regression summary statistics:")
print(f"Number of models: {len(regression_summary)}")
print(f"Average R-squared (count models): {regression_summary[regression_summary['Model_Type'] == 'Count']['R_squared'].mean():.3f}")
print(f"Average R-squared (rate models): {regression_summary[regression_summary['Model_Type'] == 'Rate']['R_squared'].mean():.3f}")

# Now create the regression analysis visualization
# Focus on key relationships - we'll use Income vs Crime Rate and Population vs Crime Rate
plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2, height_ratios=[1, 1])

# 1. Top predictor: Income vs Crime
ax1 = plt.subplot(gs[0, :])
sns.regplot(data=merged_df, x='Median_Income_2020', y='TOTAL_CRIME_COUNT_2020', 
           scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax1)
ax1.set_title('Median Income vs Total Crime Count (2020)', fontsize=14)
ax1.set_xlabel('Median Income ($)', fontsize=12)
ax1.set_ylabel('Crime Count', fontsize=12)

# 2. Bottom-left: Income vs Crime Rate
ax2 = plt.subplot(gs[1, 0])
sns.regplot(data=merged_df, x='Median_Income_2020', y='TOTAL_CRIME_RATE_2020',
           scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax2)
ax2.set_title('Median Income vs Crime Rate', fontsize=14)
ax2.set_xlabel('Median Income ($)', fontsize=12)
ax2.set_ylabel('Crime Rate (per 100,000)', fontsize=12)

# 3. Bottom-right: Population vs Crime Count
ax3 = plt.subplot(gs[1, 1])
sns.regplot(data=merged_df, x='Total_Population', y='TOTAL_CRIME_COUNT_2020',
           scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax3)
ax3.set_title('Population vs Total Crime Count', fontsize=14)
ax3.set_xlabel('Total Population', fontsize=12)
ax3.set_ylabel('Crime Count', fontsize=12)

# Highlight improvement areas if that column exists
if 'is_improvement_area' in merged_df.columns:
    improvement_areas = merged_df[merged_df['is_improvement_area'] == 1]
    sns.scatterplot(data=improvement_areas, x='Median_Income_2020', y='TOTAL_CRIME_COUNT_2020', 
                   color='red', s=80, marker='x', label='Improvement Area', ax=ax1)
    sns.scatterplot(data=improvement_areas, x='Median_Income_2020', y='TOTAL_CRIME_RATE_2020', 
                   color='red', s=80, marker='x', label='Improvement Area', ax=ax2)
    sns.scatterplot(data=improvement_areas, x='Total_Population', y='TOTAL_CRIME_COUNT_2020', 
                   color='red', s=80, marker='x', label='Improvement Area', ax=ax3)

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Regression analysis complete - files created:")
print("- regression_summary.csv")
print("- regression_analysis.png") 