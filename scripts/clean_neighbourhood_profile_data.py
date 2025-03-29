import pandas as pd
import numpy as np

# Read the raw data
df = pd.read_csv("raw_data/neighbourhood-profiles-2021-158-model.csv")

# Print the first few rows to understand the structure
print("Original data shape:", df.shape)
print("\nFirst few rows of original data:")
print(df.head())

# Get the column names (neighborhood names)
neighborhoods = df.columns[1:].tolist()
print("\nNumber of neighborhoods:", len(neighborhoods))

# Create a new dataframe with neighborhoods as rows
cleaned_df = pd.DataFrame()
cleaned_df['Neighbourhood Name'] = neighborhoods

# Find the correct row indices for each indicator
row_indices = {
    'Neighbourhood Number': df[df.iloc[:, 0] == 'Neighbourhood Number'].index[0],
    'TSNS 2020 Designation': df[df.iloc[:, 0] == 'TSNS 2020 Designation'].index[0],
    'Total Population': df[df.iloc[:, 0] == 'Total - Age groups of the population - 25% sample data'].index[0],
    'Population 0-14': df[df.iloc[:, 0] == '0 to 14 years'].index[0],
    'Population 15-64': df[df.iloc[:, 0] == '15 to 64 years'].index[0],
    'Population 65+': df[df.iloc[:, 0] == '65 years and over'].index[0],
    'Average Age': df[df.iloc[:, 0] == 'Average age of the population'].index[0],
    'Median Age': df[df.iloc[:, 0] == 'Median age of the population'].index[0],
    'Median Income': df[df.iloc[:, 0] == 'Median total income in 2020  among recipients ($)'].index[0],
    'Average Income': df[df.iloc[:, 0] == 'Average total income in 2020 among recipients ($)'].index[0],
    'Median After-Tax Income': df[df.iloc[:, 0] == 'Median after-tax income in 2020 among recipients ($)'].index[0],
    'Average After-Tax Income': df[df.iloc[:, 0] == 'Average after-tax income in 2020 among recipients ($)'].index[0]
}

# Get the values for each indicator
cleaned_df['Neighbourhood Number'] = pd.to_numeric(df.iloc[row_indices['Neighbourhood Number'], 1:].values, errors='coerce')
cleaned_df['TSNS 2020 Designation'] = df.iloc[row_indices['TSNS 2020 Designation'], 1:].values
cleaned_df['Total_Population'] = pd.to_numeric(df.iloc[row_indices['Total Population'], 1:].values, errors='coerce')
cleaned_df['Population_0_14'] = pd.to_numeric(df.iloc[row_indices['Population 0-14'], 1:].values, errors='coerce')
cleaned_df['Population_15_64'] = pd.to_numeric(df.iloc[row_indices['Population 15-64'], 1:].values, errors='coerce')
cleaned_df['Population_65_plus'] = pd.to_numeric(df.iloc[row_indices['Population 65+'], 1:].values, errors='coerce')
cleaned_df['Average_Age'] = pd.to_numeric(df.iloc[row_indices['Average Age'], 1:].values, errors='coerce')
cleaned_df['Median_Age'] = pd.to_numeric(df.iloc[row_indices['Median Age'], 1:].values, errors='coerce')
cleaned_df['Median_Income_2020'] = pd.to_numeric(df.iloc[row_indices['Median Income'], 1:].values, errors='coerce')
cleaned_df['Average_Income_2020'] = pd.to_numeric(df.iloc[row_indices['Average Income'], 1:].values, errors='coerce')
cleaned_df['Median_After_Tax_Income_2020'] = pd.to_numeric(df.iloc[row_indices['Median After-Tax Income'], 1:].values, errors='coerce')
cleaned_df['Average_After_Tax_Income_2020'] = pd.to_numeric(df.iloc[row_indices['Average After-Tax Income'], 1:].values, errors='coerce')

# Calculate additional demographic metrics
cleaned_df['Youth_Ratio'] = (cleaned_df['Population_0_14'] / cleaned_df['Total_Population'] * 100).round(1)
cleaned_df['Working_Age_Ratio'] = (cleaned_df['Population_15_64'] / cleaned_df['Total_Population'] * 100).round(1)
cleaned_df['Senior_Ratio'] = (cleaned_df['Population_65_plus'] / cleaned_df['Total_Population'] * 100).round(1)

# Sort by neighbourhood number
cleaned_df = cleaned_df.sort_values('Neighbourhood Number')

# Export cleaned data
cleaned_df.to_csv("cleaned_data/cleaned_neighbourhood_data.csv", index=False)

print("\nNeighbourhood data cleaning completed. File saved as 'cleaned_neighbourhood_data.csv'")

# Print some basic statistics
print("\nBasic Statistics:")
print(f"Number of neighbourhoods: {len(cleaned_df)}")
print(f"Average median income: ${cleaned_df['Median_Income_2020'].mean():,.2f}")
print(f"Average population per neighbourhood: {cleaned_df['Total_Population'].mean():,.0f}")
print(f"Number of Neighbourhood Improvement Areas: {len(cleaned_df[cleaned_df['TSNS 2020 Designation'].str.contains('Improvement Area', na=False)])}")

# Print sample of the cleaned data
print("\nSample of cleaned data (first 5 rows):")
print(cleaned_df.head())
