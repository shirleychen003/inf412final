import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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
    'THEFTOVER_2020': 'THEFTOVER_RATE_2020'
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

# Create results DataFrames
count_results = []
rate_results = []

# Perform regression analysis for both counts and rates
for count_code, rate_code in crime_types.items():
    # Prepare data
    X = merged_df[independent_vars]
    
    # Count regression
    y_count = merged_df[count_code]
    count_model = LinearRegression()
    count_model.fit(X, y_count)
    count_r2 = count_model.score(X, y_count)
    
    # Rate regression
    y_rate = merged_df[rate_code]
    rate_model = LinearRegression()
    rate_model.fit(X, y_rate)
    rate_r2 = rate_model.score(X, y_rate)
    
    # Store results
    count_results.append({
        'Crime Type': count_code.replace('_2020', ''),
        'R-squared': count_r2,
        'Coefficients': dict(zip(independent_vars, count_model.coef_))
    })
    
    rate_results.append({
        'Crime Type': rate_code.replace('_RATE_2020', ''),
        'R-squared': rate_r2,
        'Coefficients': dict(zip(independent_vars, rate_model.coef_))
    })
    
    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot count vs income
    sns.scatterplot(data=merged_df, x='Median_Income_2020', y=count_code, ax=ax1)
    ax1.set_title(f'{count_code.replace("_2020", "")} Count vs Income')
    ax1.set_xlabel('Median Income')
    ax1.set_ylabel('Number of Crimes')
    
    # Plot rate vs income
    sns.scatterplot(data=merged_df, x='Median_Income_2020', y=rate_code, ax=ax2)
    ax2.set_title(f'{rate_code.replace("_RATE_2020", "")} Rate vs Income')
    ax2.set_xlabel('Median Income')
    ax2.set_ylabel('Rate per 100,000 population')
    
    plt.tight_layout()
    plt.savefig(f'output/lin_reg/{count_code.replace("_2020", "")}_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

# Convert results to DataFrames
count_df = pd.DataFrame(count_results)
rate_df = pd.DataFrame(rate_results)

# Save results to CSV
count_df.to_csv('output/lin_reg/crime_counts_regression_results.csv', index=False)
rate_df.to_csv('output/lin_reg/crime_rates_regression_results.csv', index=False)

# Print summary of results
print("\nRegression Results Summary:")
print("\nCount Regression R-squared values:")
print(count_df[['Crime Type', 'R-squared']].round(3))
print("\nRate Regression R-squared values:")
print(rate_df[['Crime Type', 'R-squared']].round(3))

# Calculate and print average R-squared values
print("\nAverage R-squared values:")
print(f"Count regressions: {count_df['R-squared'].mean():.3f}")
print(f"Rate regressions: {rate_df['R-squared'].mean():.3f}")

# Print strongest predictors for each crime type
print("\nStrongest predictors (|coefficient| > 0.1) for each crime type:")
for idx, row in count_df.iterrows():
    print(f"\n{row['Crime Type']} (Count):")
    coefs = row['Coefficients']
    strong_predictors = {k: v for k, v in coefs.items() if abs(v) > 0.1}
    for pred, coef in strong_predictors.items():
        print(f"  {pred}: {coef:.3f}")

for idx, row in rate_df.iterrows():
    print(f"\n{row['Crime Type']} (Rate):")
    coefs = row['Coefficients']
    strong_predictors = {k: v for k, v in coefs.items() if abs(v) > 0.1}
    for pred, coef in strong_predictors.items():
        print(f"  {pred}: {coef:.3f}")

# Create comparison plot of R-squared values
plt.figure(figsize=(12, 6))
x = np.arange(len(crime_types))
width = 0.35

plt.bar(x - width/2, count_df['R-squared'], width, label='Counts')
plt.bar(x + width/2, rate_df['R-squared'], width, label='Rates')

plt.xlabel('Crime Type')
plt.ylabel('R-squared Value')
plt.title('Comparison of R-squared Values: Counts vs Rates')
plt.xticks(x, [code.replace('_2020', '') for code in crime_types.keys()], rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('output/lin_reg/r_squared_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
