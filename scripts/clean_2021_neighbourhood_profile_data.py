import pandas as pd
import numpy as np

# Load the data from CSV file
print("Loading raw data...")
raw_data = pd.read_csv('../raw_data/neighbourhood-profiles-2021-158-model.csv')

# First row contains neighbourhood names
neighbourhood_names = raw_data.columns[1:]  # Skip the first column which is the feature name

# Create a new DataFrame for cleaned data
print("Creating cleaned data frame...")
cleaned_data = pd.DataFrame({'neighbourhood_name': neighbourhood_names})

# Extract neighbourhood IDs from the row that contains them
neighbourhood_id_row = raw_data[raw_data.iloc[:, 0] == 'Neighbourhood Number']
if not neighbourhood_id_row.empty:
    # Convert row to list, skip the first column which is the row name
    neighbourhood_ids = neighbourhood_id_row.iloc[0, 1:].tolist()
    cleaned_data['neighbourhood_id'] = neighbourhood_ids
else:
    print("Warning: Could not find Neighbourhood Number row")

# Extract TSNS designation
tsns_row = raw_data[raw_data.iloc[:, 0] == 'TSNS 2020 Designation']
if not tsns_row.empty:
    cleaned_data['tsns_designation'] = tsns_row.iloc[0, 1:].tolist()

# Extract demographic data
print("Extracting demographic data...")
total_population_row = raw_data[raw_data.iloc[:, 0] == 'Total - Age groups of the population - 25% sample data']
if not total_population_row.empty:
    cleaned_data['total_population'] = pd.to_numeric(total_population_row.iloc[0, 1:].values, errors='coerce')

# Extract income-related metrics
print("Extracting income-related metrics...")
# Median income
median_income_row = raw_data[raw_data.iloc[:, 0] == 'Median total income in 2020  among recipients ($)']
if not median_income_row.empty:
    cleaned_data['median_income'] = pd.to_numeric(median_income_row.iloc[0, 1:].values, errors='coerce')

# Average income
avg_income_row = raw_data[raw_data.iloc[:, 0] == 'Average total income in 2020 among recipients ($)']
if not avg_income_row.empty:
    cleaned_data['average_income'] = pd.to_numeric(avg_income_row.iloc[0, 1:].values, errors='coerce')

# Median after-tax income
median_after_tax_row = raw_data[raw_data.iloc[:, 0] == 'Median after-tax income in 2020 among recipients ($)']
if not median_after_tax_row.empty:
    cleaned_data['median_after_tax_income'] = pd.to_numeric(median_after_tax_row.iloc[0, 1:].values, errors='coerce')

# Average after-tax income
avg_after_tax_row = raw_data[raw_data.iloc[:, 0] == 'Average after-tax income in 2020 among recipients ($)']
if not avg_after_tax_row.empty:
    cleaned_data['average_after_tax_income'] = pd.to_numeric(avg_after_tax_row.iloc[0, 1:].values, errors='coerce')

# Income composition percentages
# Market income
market_income_row = raw_data[raw_data.iloc[:, 0] == 'Market income (%)']
if not market_income_row.empty:
    cleaned_data['market_income_percent'] = pd.to_numeric(market_income_row.iloc[0, 1:].values, errors='coerce')

# Employment income
employment_income_row = raw_data[raw_data.iloc[:, 0] == 'Employment income (%)']
if not employment_income_row.empty:
    cleaned_data['employment_income_percent'] = pd.to_numeric(employment_income_row.iloc[0, 1:].values, errors='coerce')

# Government support
govt_support_row = raw_data[raw_data.iloc[:, 0] == 'COVID-19 - Government income support and benefits (%)']
if not govt_support_row.empty:
    cleaned_data['government_support_percent'] = pd.to_numeric(govt_support_row.iloc[0, 1:].values, errors='coerce')

# Additional income metrics
# Low income measure
print("Extracting low income metrics...")
low_income_row = raw_data[raw_data.iloc[:, 0] == 'Prevalence of low income based on the Low-income measure, after tax (LIM-AT) (%)']
if not low_income_row.empty:
    cleaned_data['low_income_percent'] = pd.to_numeric(low_income_row.iloc[0, 1:].values, errors='coerce')

# Income groups
print("Extracting income distribution data...")
try:
    # Income groups data
    without_income_row = raw_data[raw_data.iloc[:, 0] == 'Without total income']
    with_income_row = raw_data[raw_data.iloc[:, 0] == 'With total income']
    
    if not without_income_row.empty and not with_income_row.empty:
        without_income = pd.to_numeric(without_income_row.iloc[0, 1:].values, errors='coerce')
        with_income = pd.to_numeric(with_income_row.iloc[0, 1:].values, errors='coerce')
        
        # Calculate percentages
        total = without_income + with_income
        cleaned_data['percent_without_income'] = (without_income / total) * 100
    
    # Income brackets - extracting high income percentage as a potential indicator
    income_100k_plus_row = raw_data[raw_data.iloc[:, 0] == '$100,000 and over']
    if not income_100k_plus_row.empty:
        income_100k_plus = pd.to_numeric(income_100k_plus_row.iloc[0, 1:].values, errors='coerce')
        cleaned_data['percent_high_income'] = (income_100k_plus / with_income) * 100
    
    # Calculate income inequality metrics
    if 'median_income' in cleaned_data.columns and 'average_income' in cleaned_data.columns:
        cleaned_data['median_to_average_ratio'] = cleaned_data['median_income'] / cleaned_data['average_income']
    
except Exception as e:
    print(f"Warning: Could not extract some income distribution data: {e}")

# Create binary indicator for NIA (Neighbourhood Improvement Area)
if 'tsns_designation' in cleaned_data.columns:
    cleaned_data['is_improvement_area'] = cleaned_data['tsns_designation'].str.contains('Neighbourhood Improvement Area').astype(int)

# Fill missing values with 0 or appropriate value
cleaned_data = cleaned_data.fillna(0)

# Print sample of the data to verify
print("\nSample of processed data:")
print(cleaned_data.head())

# Print summary statistics
print("\nSummary statistics for key metrics:")
numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
print(cleaned_data[numeric_columns].describe().T[['mean', 'min', 'max']])

# Export cleaned data to CSV
print("\nSaving cleaned data...")
cleaned_data.to_csv('../cleaned_data/2021/cleaned_neighbourhood_income_data_2021.csv', index=False)

print(f"Done! Cleaned data saved for {len(cleaned_data)} neighbourhoods.")
