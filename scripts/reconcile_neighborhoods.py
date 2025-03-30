import pandas as pd
import os

# Create output directory if it doesn't exist
os.makedirs('cleaned_data/reconciled', exist_ok=True)

# Load the data
n_data_2016 = pd.read_csv('cleaned_data/2016/cleaned_neighbourhood_income_data_2016.csv')
c_data_2016 = pd.read_csv('cleaned_data/2016/cleaned_crime_data_2016.csv')
n_data_2021 = pd.read_csv('cleaned_data/2021/cleaned_neighbourhood_income_data_2021.csv')
c_data_2021 = pd.read_csv('cleaned_data/2021/cleaned_crime_data_2021.csv')

# Print basic info about each dataset
print(f"2016 Neighborhood Income Data: {len(n_data_2016)} rows")
print(f"2016 Crime Data: {len(c_data_2016)} rows")
print(f"2021 Neighborhood Income Data: {len(n_data_2021)} rows")
print(f"2021 Crime Data: {len(c_data_2021)} rows")

# Print column names to debug
print("\n2021 Neighborhood Data Columns:")
print(n_data_2021.columns.tolist())
print("\n2016 Neighborhood Data Columns:")
print(n_data_2016.columns.tolist())

# Get unique neighborhood names from each dataset
n_names_2016 = set(n_data_2016['neighbourhood_name'])
c_names_2016 = set(c_data_2016['AREA_NAME'])
n_names_2021 = set(n_data_2021['neighbourhood_name'])  # Using correct column name
c_names_2021 = set(c_data_2021['AREA_NAME'])

print(f"\nUnique neighborhoods in 2016 income data: {len(n_names_2016)}")
print(f"Unique neighborhoods in 2016 crime data: {len(c_names_2016)}")
print(f"Unique neighborhoods in 2021 income data: {len(n_names_2021)}")
print(f"Unique neighborhoods in 2021 crime data: {len(c_names_2021)}")

# Find neighborhoods in 2021 that aren't in 2016 income data
new_n_2021 = n_names_2021 - n_names_2016
new_c_2021 = c_names_2021 - c_names_2016

print(f"\nNeighborhoods in 2021 income data not found in 2016 income data ({len(new_n_2021)}):")
print(sorted(list(new_n_2021)))
print(f"\nNeighborhoods in 2021 crime data not found in 2016 crime data ({len(new_c_2021)}):")
print(sorted(list(new_c_2021)))

# Print the exact name from 2016 dataset for Yonge-St. Clair to debug
print("\nExact name from 2016 dataset:")
for name in n_names_2016:
    if 'Yonge' in name and 'Clair' in name:
        print(f"Found: '{name}'")

# Define a mapping from 2021 neighborhoods to 2016 neighborhoods
# This maps neighborhoods that were split or renamed
neighborhood_mapping = {
    # 2021 name -> 2016 name
    'Dovercourt Village': 'Dovercourt-Wallace Emerson-Junction',
    'Junction-Wallace Emerson': 'Dovercourt-Wallace Emerson-Junction',
    'Yonge-Bay Corridor': 'Bay Street Corridor',
    'Bay-Cloverhill': 'Bay Street Corridor',
    'Bendale-Glen Andrew': 'Bendale',
    'Downsview': 'Downsview-Roding-CFB',
    'Oakdale-Beverley Heights': 'Downsview-Roding-CFB',
    'East Willowdale': 'Willowdale East',
    'Yonge-Doris': 'Willowdale East',
    'Fenside-Parkwoods': 'Parkwoods-Donalda',
    'Parkwoods-O\'Connor Hills': 'Parkwoods-Donalda',
    'East L\'Amoreaux': 'L\'Amoreaux',
    'L\'Amoreaux West': 'L\'Amoreaux',
    'Malvern East': 'Malvern',
    'Malvern West': 'Malvern',
    'Woburn North': 'Woburn',
    'Golfdale-Cedarbrae-Woburn': 'Woburn',
    'Downtown Yonge East': 'Church-Yonge Corridor',
    'Church-Wellesley': 'Church-Yonge Corridor',
    'St Lawrence-East Bayfront-The Islands': 'Waterfront Communities-The Island',
    'Harbourfront-CityPlace': 'Waterfront Communities-The Island',
    'Wellington Place': 'Waterfront Communities-The Island',
    'Fort York-Liberty Village': 'Niagara',
    'West Queen West': 'Niagara',
    'Humber Bay Shores': 'Mimico (includes Humber Bay Shores)',
    'Mimico-Queensway': 'Mimico (includes Humber Bay Shores)',
    'Etobicoke City Centre': 'Islington-City Centre West',
    'Islington': 'Islington-City Centre West',
    'Bendale South': 'Bendale',
    'South Eglinton-Davisville': 'Mount Pleasant West',
    'North Toronto': 'Mount Pleasant West',
    # Additional mappings for remaining unmatched neighborhoods
    'Avondale': 'Willowdale West',
    'Danforth-East York': 'Danforth East York',
    'East End Danforth': 'East End-Danforth',
    'Morningside Heights': 'Rouge',
    'O`Connor Parkview': 'O\'Connor-Parkview',
    'Taylor Massey': 'Taylor-Massey',
    'West Rouge': 'Rouge',
    'Yonge-St.Clair': 'Yonge-St.Clair',
    'Yonge-St. Clair': 'Yonge-St.Clair'  # Fix for Yonge-St. Clair (with space)
}

# Create an inverse mapping for 2016 to 2021 standardized names
inverse_mapping = {}
for new_name, old_name in neighborhood_mapping.items():
    if old_name not in inverse_mapping:
        inverse_mapping[old_name] = []
    inverse_mapping[old_name].append(new_name)

# Create copies of the data for reconciliation
n_data_reconciled_2021 = n_data_2021.copy()
c_data_reconciled_2021 = c_data_2021.copy()
n_data_reconciled_2016 = n_data_2016.copy()
c_data_reconciled_2016 = c_data_2016.copy()

# Create a new column for reconciled neighborhood names
n_data_reconciled_2021['Reconciled_Neighbourhood'] = n_data_reconciled_2021['neighbourhood_name'].map(
    lambda x: neighborhood_mapping.get(x, x)
)
c_data_reconciled_2021['Reconciled_Neighbourhood'] = c_data_reconciled_2021['AREA_NAME'].map(
    lambda x: neighborhood_mapping.get(x, x)
)
n_data_reconciled_2016['Reconciled_Neighbourhood'] = n_data_reconciled_2016['neighbourhood_name']
c_data_reconciled_2016['Reconciled_Neighbourhood'] = c_data_reconciled_2016['AREA_NAME']

# Group by reconciled neighborhoods and aggregate data
# For neighborhood data, we'll take the sum for numeric columns and first value for categorical
def aggregate_neighborhood_data_2021(group):
    # First, handle the numeric columns
    # Calculate the weighted average for percentages and medians
    numeric_cols = [col for col in group.columns if col not in ['neighbourhood_name', 'Reconciled_Neighbourhood']]
    
    # Get population column for weighting
    population_col = 'total_population'
    
    result = {}
    # Use weighted average for numeric columns
    for col in numeric_cols:
        if group[col].dtype != 'object':
            # For population, just sum
            if col == population_col:
                result[col] = group[col].sum()
            else:
                # For other numeric, use population-weighted average
                result[col] = (group[col] * group[population_col]).sum() / group[population_col].sum()
    
    # Keep the first value for non-numeric columns
    for col in group.columns:
        if col not in result and col not in ['neighbourhood_name', 'Reconciled_Neighbourhood']:
            result[col] = group[col].iloc[0]
    
    return pd.Series(result)

def aggregate_neighborhood_data_2016(group):
    # First, handle the numeric columns
    # Calculate the weighted average for percentages and medians
    numeric_cols = [col for col in group.columns if col not in ['neighbourhood_name', 'Reconciled_Neighbourhood']]
    
    # Get population column for weighting
    population_col = 'total_population'
    
    result = {}
    # Use weighted average for numeric columns
    for col in numeric_cols:
        if group[col].dtype != 'object':
            # For population, just sum
            if col == population_col:
                result[col] = group[col].sum()
            else:
                # For other numeric, use population-weighted average
                result[col] = (group[col] * group[population_col]).sum() / group[population_col].sum()
    
    # Keep the first value for non-numeric columns
    for col in group.columns:
        if col not in result and col not in ['neighbourhood_name', 'Reconciled_Neighbourhood']:
            result[col] = group[col].iloc[0]
    
    return pd.Series(result)

# Aggregate crime data by summing counts and calculating population-weighted averages for rates
def aggregate_crime_data_2021(group):
    result = {}
    
    # Handle count columns (sum)
    count_cols = [col for col in group.columns if 'RATE' not in col and col not in ['AREA_NAME', 'Reconciled_Neighbourhood']]
    for col in count_cols:
        result[col] = group[col].sum()
    
    # Handle rate columns (population-weighted average)
    # Since we don't have population in the crime data, we'll recalculate rates based on the summed counts
    # We'll do this later when we merge with the neighborhood data
    
    return pd.Series(result)

def aggregate_crime_data_2016(group):
    result = {}
    
    # Handle count columns (sum)
    count_cols = [col for col in group.columns if 'RATE' not in col and col not in ['AREA_NAME', 'Reconciled_Neighbourhood']]
    for col in count_cols:
        result[col] = group[col].sum()
    
    # Handle rate columns (population-weighted average)
    # Since we don't have population in the crime data, we'll recalculate rates based on the summed counts
    # We'll do this later when we merge with the neighborhood data
    
    return pd.Series(result)

# Aggregate the data for 2021
n_data_aggregated_2021 = n_data_reconciled_2021.groupby('Reconciled_Neighbourhood').apply(aggregate_neighborhood_data_2021).reset_index()
c_data_aggregated_2021 = c_data_reconciled_2021.groupby('Reconciled_Neighbourhood').apply(aggregate_crime_data_2021).reset_index()

# Aggregate the data for 2016 (no need for complex aggregation since no merging is happening)
n_data_aggregated_2016 = n_data_reconciled_2016.copy()
c_data_aggregated_2016 = c_data_reconciled_2016.copy()

# Recalculate crime rates based on the aggregated data for 2021
# First, merge with neighborhood data to get population
merged_data_2021 = pd.merge(
    c_data_aggregated_2021,
    n_data_aggregated_2021[['Reconciled_Neighbourhood', 'total_population']],
    on='Reconciled_Neighbourhood',
    how='left'
)

# No need to recalculate rates for 2016 since no neighborhoods were combined

# Recalculate crime rates for 2021
rate_columns_2021 = [col for col in merged_data_2021.columns if 'RATE' in col]
count_columns_2021 = [col.replace('_RATE', '') for col in rate_columns_2021]

for i, rate_col in enumerate(rate_columns_2021):
    count_col = count_columns_2021[i]
    if count_col in merged_data_2021.columns:
        merged_data_2021[rate_col] = (merged_data_2021[count_col] / merged_data_2021['total_population']) * 100000

# Save the reconciled data
n_data_aggregated_2021.to_csv('cleaned_data/reconciled/reconciled_neighbourhood_data_2021.csv', index=False)
merged_data_2021.to_csv('cleaned_data/reconciled/reconciled_crime_data_2021.csv', index=False)
n_data_aggregated_2016.to_csv('cleaned_data/reconciled/reconciled_neighbourhood_data_2016.csv', index=False)
c_data_aggregated_2016.to_csv('cleaned_data/reconciled/reconciled_crime_data_2016.csv', index=False)

# Print summary stats
print(f"\nReconciled 2021 data now has {len(n_data_aggregated_2021)} neighborhoods")
print(f"Original 2016 data had {len(n_data_2016)} neighborhoods")

# Check if the reconciliation was effective
reconciled_names_2021 = set(n_data_aggregated_2021['Reconciled_Neighbourhood'])
original_names_2016 = set(n_data_2016['neighbourhood_name'])
still_different = reconciled_names_2021 - original_names_2016

print(f"\nNeighborhoods still not matching between reconciled 2021 and 2016 data ({len(still_different)}):")
print(sorted(list(still_different)))

# Report on the reconciliation
print(f"\nReconciliation Summary:")
print(f"- Original 2021 neighborhood count: {len(n_data_2021)}")
print(f"- Reconciled 2021 neighborhood count: {len(n_data_aggregated_2021)}")
print(f"- Number of neighborhoods merged: {len(n_data_2021) - len(n_data_aggregated_2021)}")
print(f"- 2016 neighborhood count: {len(n_data_2016)}")
print(f"- Difference after reconciliation: {abs(len(n_data_aggregated_2021) - len(n_data_2016))}")

# Now we'll create a unified dataset with common columns for both years
print("\nCreating unified datasets with data from both years...")

# First, let's standardize the column names between the two years
# For neighborhood data
n_data_2016_cols = {
    'neighbourhood_name': 'neighbourhood_name',
    'total_population': 'total_population_2016',
    'low_income_percent': 'low_income_percent_2016',
    'is_improvement_area': 'is_improvement_area_2016',
    'Reconciled_Neighbourhood': 'Reconciled_Neighbourhood'
}

n_data_2021_cols = {
    'neighbourhood_name': 'neighbourhood_name_2021',
    'total_population': 'total_population_2021',
    'low_income_percent': 'low_income_percent_2021',
    'is_improvement_area': 'is_improvement_area_2021',
    'Reconciled_Neighbourhood': 'Reconciled_Neighbourhood'
}

# For crime data, we'll keep the original column names since they already include the year

# Rename columns in the reconciled data
n_data_2016_renamed = n_data_aggregated_2016.rename(columns=n_data_2016_cols)
n_data_2021_renamed = n_data_aggregated_2021.rename(columns=n_data_2021_cols)

# Select only the columns we need
n_data_2016_selected = n_data_2016_renamed[['Reconciled_Neighbourhood', 'total_population_2016', 'low_income_percent_2016', 'is_improvement_area_2016']]
n_data_2021_selected = n_data_2021_renamed[['Reconciled_Neighbourhood', 'total_population_2021', 'low_income_percent_2021', 'is_improvement_area_2021']]

# Merge the neighborhood data from both years
unified_neighborhood_data = pd.merge(
    n_data_2016_selected,
    n_data_2021_selected,
    on='Reconciled_Neighbourhood',
    how='outer'
)

# Calculate population change and low income change
unified_neighborhood_data['population_change'] = unified_neighborhood_data['total_population_2021'] - unified_neighborhood_data['total_population_2016']
unified_neighborhood_data['population_percent_change'] = (unified_neighborhood_data['population_change'] / unified_neighborhood_data['total_population_2016']) * 100
unified_neighborhood_data['low_income_percent_change'] = unified_neighborhood_data['low_income_percent_2021'] - unified_neighborhood_data['low_income_percent_2016']

# Save the unified neighborhood data
unified_neighborhood_data.to_csv('cleaned_data/reconciled/unified_neighbourhood_data.csv', index=False)

# Merge the crime data from both years
# First, we need to select only the count columns from each dataset
c_data_2016_counts = c_data_aggregated_2016[[col for col in c_data_aggregated_2016.columns if 'RATE' not in col or col == 'Reconciled_Neighbourhood']]
c_data_2021_counts = merged_data_2021[[col for col in merged_data_2021.columns if 'RATE' not in col or col == 'Reconciled_Neighbourhood']]

# Merge the crime data
unified_crime_data = pd.merge(
    c_data_2016_counts,
    c_data_2021_counts,
    on='Reconciled_Neighbourhood',
    how='outer',
    suffixes=('_2016', '_2021')
)

# Calculate crime change
for crime_type in ['ASSAULT', 'AUTOTHEFT', 'BIKETHEFT', 'BREAKENTER', 'HOMICIDE', 'ROBBERY', 'SHOOTING', 'THEFTFROMMV', 'THEFTOVER', 'TOTAL_CRIME_COUNT']:
    col_2016 = f"{crime_type}_2016"
    col_2021 = f"{crime_type}_2021"
    if col_2016 in unified_crime_data.columns and col_2021 in unified_crime_data.columns:
        unified_crime_data[f"{crime_type}_change"] = unified_crime_data[col_2021] - unified_crime_data[col_2016]
        unified_crime_data[f"{crime_type}_percent_change"] = (unified_crime_data[f"{crime_type}_change"] / unified_crime_data[col_2016]) * 100

# Save the unified crime data
unified_crime_data.to_csv('cleaned_data/reconciled/unified_crime_data.csv', index=False)

# Finally, merge the unified neighborhood and crime data
unified_data = pd.merge(
    unified_neighborhood_data,
    unified_crime_data,
    on='Reconciled_Neighbourhood',
    how='outer'
)

# Save the comprehensive unified dataset
unified_data.to_csv('cleaned_data/reconciled/unified_data.csv', index=False)

print(f"Created unified datasets with data from both 2016 and 2021.")
print(f"- Unified neighborhood data: {len(unified_neighborhood_data)} neighborhoods")
print(f"- Unified crime data: {len(unified_crime_data)} neighborhoods")
print(f"- Comprehensive unified data: {len(unified_data)} neighborhoods") 