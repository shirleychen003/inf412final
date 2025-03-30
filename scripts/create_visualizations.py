import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

# Create output directories if they don't exist
os.makedirs('output/box_plot', exist_ok=True)
os.makedirs('output/correlational_matrix', exist_ok=True)
os.makedirs('output/linear_regression', exist_ok=True)

# Load the unified data
print("Loading data...")
data = pd.read_csv('cleaned_data/reconciled_data/unified_data.csv')

# Clean data - remove rows with missing values for key metrics
print("Cleaning data...")
clean_data = data.dropna(subset=[
    'total_population_2016', 'low_income_percent_2016', 
    'total_population_2021', 'low_income_percent_2021',
    'TOTAL_CRIME_COUNT_2016', 'TOTAL_CRIME_COUNT_2021'
])

print(f"Original data rows: {len(data)}")
print(f"Clean data rows: {len(clean_data)}")

# 1. Box Plots
print("Creating box plots...")

# Crime counts box plot
plt.figure(figsize=(14, 8))
crime_cols_2016 = ['ASSAULT_2016', 'ROBBERY_2016', 'AUTOTHEFT_2016', 'BREAKENTER_2016', 'THEFTFROMMV_2016']
crime_cols_2021 = ['ASSAULT_2021', 'ROBBERY_2021', 'AUTOTHEFT_2021', 'BREAKENTER_2021', 'THEFTFROMMV_2021']

# Convert to long format for box plots
crime_2016_df = clean_data[crime_cols_2016].copy()
crime_2016_df = crime_2016_df.rename(columns={col: col.replace('_2016', '') for col in crime_cols_2016})
crime_2016_df['Year'] = '2016'
crime_2016_df = pd.melt(crime_2016_df, id_vars=['Year'], var_name='Crime Type', value_name='Count')

crime_2021_df = clean_data[crime_cols_2021].copy()
crime_2021_df = crime_2021_df.rename(columns={col: col.replace('_2021', '') for col in crime_cols_2021})
crime_2021_df['Year'] = '2021'
crime_2021_df = pd.melt(crime_2021_df, id_vars=['Year'], var_name='Crime Type', value_name='Count')

crime_df = pd.concat([crime_2016_df, crime_2021_df])

plt.figure(figsize=(12, 8))
sns.boxplot(x='Crime Type', y='Count', hue='Year', data=crime_df)
plt.title('Crime Counts by Type (2016 vs 2021)', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/box_plot/crime_counts_comparison.png')
plt.close()

# Population and low income percentage box plot
plt.figure(figsize=(10, 6))
pop_boxplot_data = pd.DataFrame({
    'Population 2016': clean_data['total_population_2016'],
    'Population 2021': clean_data['total_population_2021'],
    'Low Income % 2016': clean_data['low_income_percent_2016'],
    'Low Income % 2021': clean_data['low_income_percent_2021']
})

pop_boxplot_data_melted = pd.melt(pop_boxplot_data, var_name='Metric', value_name='Value')
plt.figure(figsize=(12, 8))
sns.boxplot(x='Metric', y='Value', data=pop_boxplot_data_melted)
plt.title('Population and Low Income Percentage (2016 vs 2021)', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/box_plot/population_income_comparison.png')
plt.close()

# Crime percent change box plot
crime_change_cols = [col for col in clean_data.columns if '_percent_change' in col and 'population' not in col and 'low_income' not in col]
crime_change_df = clean_data[crime_change_cols].copy()
crime_change_df = pd.melt(crime_change_df, var_name='Crime Type', value_name='Percent Change')
crime_change_df['Crime Type'] = crime_change_df['Crime Type'].str.replace('_percent_change', '')

plt.figure(figsize=(14, 8))
sns.boxplot(x='Crime Type', y='Percent Change', data=crime_change_df)
plt.title('Crime Percent Change (2016 to 2021)', fontsize=16)
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='-')
plt.ylim(-100, 300)  # Limiting to reasonable range
plt.tight_layout()
plt.savefig('output/box_plot/crime_percent_change.png')
plt.close()

# 2. Correlation Matrices
print("Creating correlation matrices...")

# 2016 correlation matrix
correlation_cols_2016 = ['total_population_2016', 'low_income_percent_2016', 'is_improvement_area_2016',
                         'ASSAULT_2016', 'AUTOTHEFT_2016', 'BREAKENTER_2016', 
                         'ROBBERY_2016', 'THEFTFROMMV_2016', 'TOTAL_CRIME_COUNT_2016']

corr_2016 = clean_data[correlation_cols_2016].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_2016, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - 2016 Data', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_2016.png')
plt.close()

# 2021 correlation matrix
correlation_cols_2021 = ['total_population_2021', 'low_income_percent_2021', 'is_improvement_area_2021',
                         'ASSAULT_2021', 'AUTOTHEFT_2021', 'BREAKENTER_2021', 
                         'ROBBERY_2021', 'THEFTFROMMV_2021', 'TOTAL_CRIME_COUNT_2021']

corr_2021 = clean_data[correlation_cols_2021].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_2021, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - 2021 Data', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_2021.png')
plt.close()

# Change correlation matrix
change_cols = ['population_percent_change', 'low_income_percent_change', 
               'ASSAULT_percent_change', 'AUTOTHEFT_percent_change', 'BREAKENTER_percent_change',
               'ROBBERY_percent_change', 'THEFTFROMMV_percent_change', 'TOTAL_CRIME_COUNT_percent_change']

# Filter out extreme values for better visualization
change_data = clean_data[change_cols].copy()
for col in change_cols:
    # Cap extreme values at 95th percentile
    upper_limit = np.percentile(change_data[col].dropna(), 95)
    lower_limit = np.percentile(change_data[col].dropna(), 5)
    change_data[col] = change_data[col].clip(lower_limit, upper_limit)

corr_change = change_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_change, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - Change Metrics (2016 to 2021)', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlational_matrix/correlation_change.png')
plt.close()

# 3. Linear Regressions
print("Creating linear regression plots...")

def create_regression_plot(x_data, y_data, x_label, y_label, title, filename):
    """Create a scatter plot with regression line and save it"""
    # Drop rows with missing values
    data_to_plot = pd.DataFrame({x_label: x_data, y_label: y_data}).dropna()
    x = data_to_plot[x_label].values.reshape(-1, 1)
    y = data_to_plot[y_label].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Create scatter plot with regression line
    plt.figure(figsize=(10, 8))
    sns.regplot(x=x_label, y=y_label, data=data_to_plot, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Add text with R² and RMSE values
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\ny = {model.coef_[0]:.3f}x + {model.intercept_:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'output/linear_regression/{filename}.png')
    plt.close()

# 2016 low income vs crime
create_regression_plot(
    clean_data['low_income_percent_2016'], 
    clean_data['TOTAL_CRIME_COUNT_2016'],
    'Low Income Percentage (2016)', 
    'Total Crime Count (2016)',
    'Low Income vs Crime (2016)',
    'low_income_vs_crime_2016'
)

# 2021 low income vs crime
create_regression_plot(
    clean_data['low_income_percent_2021'], 
    clean_data['TOTAL_CRIME_COUNT_2021'],
    'Low Income Percentage (2021)', 
    'Total Crime Count (2021)',
    'Low Income vs Crime (2021)',
    'low_income_vs_crime_2021'
)

# Change in low income vs change in crime
create_regression_plot(
    clean_data['low_income_percent_change'], 
    clean_data['TOTAL_CRIME_COUNT_percent_change'],
    'Low Income Percentage Change', 
    'Total Crime Count Percentage Change',
    'Change in Low Income vs Change in Crime (2016-2021)',
    'change_low_income_vs_change_crime'
)

# Population vs crime for both years
create_regression_plot(
    clean_data['total_population_2016'], 
    clean_data['TOTAL_CRIME_COUNT_2016'],
    'Total Population (2016)', 
    'Total Crime Count (2016)',
    'Population vs Crime (2016)',
    'population_vs_crime_2016'
)

create_regression_plot(
    clean_data['total_population_2021'], 
    clean_data['TOTAL_CRIME_COUNT_2021'],
    'Total Population (2021)', 
    'Total Crime Count (2021)',
    'Population vs Crime (2021)',
    'population_vs_crime_2021'
)

# Compare assault rates between improvement areas and non-improvement areas
improvement_2016 = clean_data[clean_data['is_improvement_area_2016'] == 1]['ASSAULT_2016']
non_improvement_2016 = clean_data[clean_data['is_improvement_area_2016'] == 0]['ASSAULT_2016']

improvement_2021 = clean_data[clean_data['is_improvement_area_2021'] == 1]['ASSAULT_2021']
non_improvement_2021 = clean_data[clean_data['is_improvement_area_2021'] == 0]['ASSAULT_2021']

plt.figure(figsize=(12, 8))
data = {
    '2016 - Not Improvement Area': non_improvement_2016,
    '2016 - Improvement Area': improvement_2016,
    '2021 - Not Improvement Area': non_improvement_2021,
    '2021 - Improvement Area': improvement_2021
}
sns.boxplot(data=data)
plt.title('Assault Counts: Improvement Areas vs Non-Improvement Areas', fontsize=16)
plt.ylabel('Assault Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/box_plot/improvement_area_assault_comparison.png')
plt.close()

print("All visualizations complete! Files saved to the 'output' directory.") 