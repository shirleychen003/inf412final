import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories if they don't exist
os.makedirs('output/box_plot', exist_ok=True)

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

# Define crime types to analyze
crime_types = [
    'ASSAULT',
    'AUTOTHEFT',
    'BIKETHEFT',
    'BREAKENTER',
    'ROBBERY',
    'THEFTFROMMV',
    'THEFTOVER',
    'TOTAL_CRIME_COUNT'
]

# Create income groups for 2021 data using quartiles
merged_2021['Income_Group'] = pd.qcut(merged_2021['median_income'], 
                                q=4, 
                                labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# For 2016, we'll use low_income_percent to create groups
merged_2016['Income_Group'] = pd.qcut(merged_2016['low_income_percent'], 
                                q=4, 
                                labels=['High', 'Medium-High', 'Medium-Low', 'Low'])  # Inverted because higher % = lower income

# Check for improvement area column
improvement_col_2016 = 'is_improvement_area'
improvement_col_2021 = 'is_improvement_area'

# 1. Create Crime Counts Comparison boxplot
plt.figure(figsize=(14, 10))
crime_counts = []

for i, crime_type in enumerate(crime_types[:7]):  # Exclude total crime count
    for year, df in [('2016', merged_2016), ('2021', merged_2021)]:
        col = f"{crime_type}_{year}"
        if col in df.columns:
            for _, row in df.iterrows():
                crime_counts.append({
                    'Crime Type': crime_type,
                    'Year': year,
                    'Count': row[col]
                })

crime_counts_df = pd.DataFrame(crime_counts)
sns.boxplot(data=crime_counts_df, x='Crime Type', y='Count', hue='Year')
plt.title('Crime Counts Comparison (2016 vs 2021)', fontsize=14)
plt.xlabel('Crime Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/box_plot/crime_counts_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create Percent Change boxplot
plt.figure(figsize=(14, 8))
percent_changes = []

# Get neighborhoods present in both datasets
common_neighborhoods = set(merged_2016['AREA_NAME']) & set(merged_2021['AREA_NAME'])
print(f"Number of neighborhoods present in both years: {len(common_neighborhoods)}")

# Calculate percent change for each neighborhood and crime type
for neighborhood in common_neighborhoods:
    nbh_2016 = merged_2016[merged_2016['AREA_NAME'] == neighborhood].iloc[0]
    nbh_2021 = merged_2021[merged_2021['AREA_NAME'] == neighborhood].iloc[0]
    
    for crime_type in crime_types[:7]:  # Exclude total crime count
        col_2016 = f"{crime_type}_2016"
        col_2021 = f"{crime_type}_2021"
        
        if col_2016 in nbh_2016 and col_2021 in nbh_2021:
            count_2016 = nbh_2016[col_2016]
            count_2021 = nbh_2021[col_2021]
            
            # Calculate percent change, handling division by zero
            if count_2016 > 0:
                pct_change = ((count_2021 - count_2016) / count_2016) * 100
            else:
                if count_2021 > 0:
                    pct_change = 100  # 100% increase if 2016 was zero
                else:
                    pct_change = 0  # No change if both are zero
            
            percent_changes.append({
                'Neighborhood': neighborhood,
                'Crime Type': crime_type,
                'Percent Change': pct_change
            })

percent_change_df = pd.DataFrame(percent_changes)
sns.boxplot(data=percent_change_df, x='Crime Type', y='Percent Change')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Percent Change in Crime Counts (2016 to 2021)', fontsize=14)
plt.xlabel('Crime Type', fontsize=12)
plt.ylabel('Percent Change (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/box_plot/crime_percent_change.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Population and Income Comparison
plt.figure(figsize=(14, 6))

# Population comparison
plt.subplot(1, 2, 1)
pop_data = []
for year, df in [('2016', merged_2016), ('2021', merged_2021)]:
    pop_col = 'total_population'
    for _, row in df.iterrows():
        pop_data.append({
            'Year': year,
            'Population': row[pop_col]
        })

pop_df = pd.DataFrame(pop_data)
sns.boxplot(data=pop_df, x='Year', y='Population')
plt.title('Neighborhood Population Comparison', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=12)

# Income comparison - use low_income_percent since it's available in both years
plt.subplot(1, 2, 2)
income_data = []
for year, df in [('2016', merged_2016), ('2021', merged_2021)]:
    income_col = 'low_income_percent'
    for _, row in df.iterrows():
        income_data.append({
            'Year': year,
            'Low Income %': row[income_col]
        })

income_df = pd.DataFrame(income_data)
sns.boxplot(data=income_df, x='Year', y='Low Income %')
plt.title('Neighborhood Low Income Percentage Comparison', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Low Income Percentage', fontsize=12)

plt.tight_layout()
plt.savefig('output/box_plot/population_income_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Improvement Area Assault Comparison
if improvement_col_2016 in merged_2016.columns and improvement_col_2021 in merged_2021.columns:
    plt.figure(figsize=(14, 6))
    
    # 2016 comparison
    plt.subplot(1, 2, 1)
    assault_data_2016 = []
    for _, row in merged_2016.iterrows():
        assault_data_2016.append({
            'Improvement Area': 'Yes' if row[improvement_col_2016] == 1 else 'No',
            'Assault Count': row['ASSAULT_2016']
        })
    
    assault_df_2016 = pd.DataFrame(assault_data_2016)
    sns.boxplot(data=assault_df_2016, x='Improvement Area', y='Assault Count')
    plt.title('Assault Count by Improvement Area (2016)', fontsize=14)
    plt.xlabel('Improvement Area', fontsize=12)
    plt.ylabel('Assault Count', fontsize=12)
    
    # 2021 comparison
    plt.subplot(1, 2, 2)
    assault_data_2021 = []
    for _, row in merged_2021.iterrows():
        assault_data_2021.append({
            'Improvement Area': 'Yes' if row[improvement_col_2021] == 1 else 'No',
            'Assault Count': row['ASSAULT_2021']
        })
    
    assault_df_2021 = pd.DataFrame(assault_data_2021)
    sns.boxplot(data=assault_df_2021, x='Improvement Area', y='Assault Count')
    plt.title('Assault Count by Improvement Area (2021)', fontsize=14)
    plt.xlabel('Improvement Area', fontsize=12)
    plt.ylabel('Assault Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/box_plot/improvement_area_assault_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Box plot analysis complete - all files saved to output/box_plot directory")
