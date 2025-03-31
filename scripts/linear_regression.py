import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

# Create output directories if they don't exist
os.makedirs('output/linear_regression', exist_ok=True)

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

# 1. Low income vs crime regression for 2016
plt.figure(figsize=(15, 10))
for i, crime_type in enumerate(crime_types[:7]):  # Exclude total crime count
    plt.subplot(3, 3, i+1)
    col = f"{crime_type}_2016"
    
    X = merged_2016[['low_income_percent']]
    y = merged_2016[col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Plot data points
    plt.scatter(merged_2016['low_income_percent'], merged_2016[col], alpha=0.6, color='blue')
    
    # Plot regression line
    plt.plot(merged_2016['low_income_percent'], y_pred, color='red', linewidth=2)
    
    # Add regression equation and R² to plot
    equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
    plt.text(0.05, 0.95, f"{equation}\nR² = {r2:.3f}", transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top')
    
    plt.title(f"{crime_type} vs Low Income (2016)", fontsize=10)
    plt.xlabel("Low Income Percentage")
    plt.ylabel(f"{crime_type} Count")
    
plt.tight_layout()
plt.savefig('output/linear_regression/low_income_vs_crime_2016.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Low income vs crime regression for 2021
plt.figure(figsize=(15, 10))
for i, crime_type in enumerate(crime_types[:7]):  # Exclude total crime count
    plt.subplot(3, 3, i+1)
    col = f"{crime_type}_2021"
    
    X = merged_2021[['low_income_percent']]
    y = merged_2021[col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Plot data points
    plt.scatter(merged_2021['low_income_percent'], merged_2021[col], alpha=0.6, color='blue')
    
    # Plot regression line
    plt.plot(merged_2021['low_income_percent'], y_pred, color='red', linewidth=2)
    
    # Add regression equation and R² to plot
    equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
    plt.text(0.05, 0.95, f"{equation}\nR² = {r2:.3f}", transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top')
    
    plt.title(f"{crime_type} vs Low Income (2021)", fontsize=10)
    plt.xlabel("Low Income Percentage")
    plt.ylabel(f"{crime_type} Count")
    
plt.tight_layout()
plt.savefig('output/linear_regression/low_income_vs_crime_2021.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Population vs crime regression for 2016
plt.figure(figsize=(15, 10))
for i, crime_type in enumerate(crime_types[:7]):  # Exclude total crime count
    plt.subplot(3, 3, i+1)
    col = f"{crime_type}_2016"
    
    X = merged_2016[['total_population']]
    y = merged_2016[col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Plot data points
    plt.scatter(merged_2016['total_population'], merged_2016[col], alpha=0.6, color='blue')
    
    # Plot regression line
    plt.plot(merged_2016['total_population'], y_pred, color='red', linewidth=2)
    
    # Add regression equation and R² to plot
    equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.2f}"
    plt.text(0.05, 0.95, f"{equation}\nR² = {r2:.3f}", transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top')
    
    plt.title(f"{crime_type} vs Population (2016)", fontsize=10)
    plt.xlabel("Population")
    plt.ylabel(f"{crime_type} Count")
    
plt.tight_layout()
plt.savefig('output/linear_regression/population_vs_crime_2016.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Population vs crime regression for 2021
plt.figure(figsize=(15, 10))
for i, crime_type in enumerate(crime_types[:7]):  # Exclude total crime count
    plt.subplot(3, 3, i+1)
    col = f"{crime_type}_2021"
    
    X = merged_2021[['total_population']]
    y = merged_2021[col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Plot data points
    plt.scatter(merged_2021['total_population'], merged_2021[col], alpha=0.6, color='blue')
    
    # Plot regression line
    plt.plot(merged_2021['total_population'], y_pred, color='red', linewidth=2)
    
    # Add regression equation and R² to plot
    equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.2f}"
    plt.text(0.05, 0.95, f"{equation}\nR² = {r2:.3f}", transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top')
    
    plt.title(f"{crime_type} vs Population (2021)", fontsize=10)
    plt.xlabel("Population")
    plt.ylabel(f"{crime_type} Count")
    
plt.tight_layout()
plt.savefig('output/linear_regression/population_vs_crime_2021.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Change in low income vs change in crime
# Get neighborhoods present in both datasets
common_neighborhoods = set(merged_2016['AREA_NAME']) & set(merged_2021['AREA_NAME'])
print(f"Number of neighborhoods present in both years: {len(common_neighborhoods)}")

# Create DataFrame for changes
change_df = pd.DataFrame()
change_df['neighborhood'] = list(common_neighborhoods)

# Calculate change in low income percentage
for neighborhood in common_neighborhoods:
    nbh_2016 = merged_2016[merged_2016['AREA_NAME'] == neighborhood].iloc[0]
    nbh_2021 = merged_2021[merged_2021['AREA_NAME'] == neighborhood].iloc[0]
    
    # Get low income percentage change
    change_df.loc[change_df['neighborhood'] == neighborhood, 'low_income_change'] = (
        nbh_2021['low_income_percent'] - nbh_2016['low_income_percent']
    )
    
    # Calculate crime changes for each type
    for crime_type in crime_types[:7]:  # Exclude total crime count
        col_2016 = f"{crime_type}_2016"
        col_2021 = f"{crime_type}_2021"
        
        if col_2016 in nbh_2016 and col_2021 in nbh_2021:
            change_df.loc[change_df['neighborhood'] == neighborhood, f'{crime_type}_change'] = (
                nbh_2021[col_2021] - nbh_2016[col_2016]
            )

# Plot change in low income vs change in crime
plt.figure(figsize=(15, 10))
for i, crime_type in enumerate(crime_types[:7]):  # Exclude total crime count
    plt.subplot(3, 3, i+1)
    
    # Get data
    X = change_df[['low_income_change']].dropna()
    y = change_df[f'{crime_type}_change'].dropna()
    
    # Filter for neighborhoods with both data points
    valid_data = change_df.dropna(subset=['low_income_change', f'{crime_type}_change'])
    
    if len(valid_data) > 5:  # Only run regression if we have enough data points
        X = valid_data[['low_income_change']]
        y = valid_data[f'{crime_type}_change']
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Plot data points
        plt.scatter(valid_data['low_income_change'], valid_data[f'{crime_type}_change'], alpha=0.6, color='blue')
        
        # Plot regression line
        plt.plot(X, y_pred, color='red', linewidth=2)
        
        # Add regression equation and R² to plot
        equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
        plt.text(0.05, 0.95, f"{equation}\nR² = {r2:.3f}", transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top')
    else:
        # Just plot the scatter points if not enough data for regression
        plt.scatter(change_df['low_income_change'], change_df[f'{crime_type}_change'], alpha=0.6, color='blue')
    
    plt.title(f"Change in {crime_type} vs Change in Low Income", fontsize=10)
    plt.xlabel("Change in Low Income Percentage")
    plt.ylabel(f"Change in {crime_type} Count")
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
plt.tight_layout()
plt.savefig('output/linear_regression/change_low_income_vs_change_crime.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Create a subdirectory for detailed regression analysis
os.makedirs('output/linear_regression/regression_analysis', exist_ok=True)

# Store regression models and metrics for all combinations
regression_results = []

# Crime types and predictor variables
predictors = ['low_income_percent', 'total_population']

# Build regression models for different combinations
for year in [2016, 2021]:
    for crime_type in crime_types:
        for predictor in predictors:
            # Get the appropriate dataframe and column names
            if year == 2016:
                df = merged_2016
                crime_col = f"{crime_type}_{year}"
            else:
                df = merged_2021
                crime_col = f"{crime_type}_{year}"
            
            # Only proceed if the crime column exists
            if crime_col in df.columns:
                X = df[[predictor]]
                y = df[crime_col]
                
                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Calculate metrics
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                
                # Store results
                regression_results.append({
                    'Year': year,
                    'Crime_Type': crime_type,
                    'Predictor': predictor,
                    'Coefficient': model.coef_[0],
                    'Intercept': model.intercept_,
                    'R_squared': r2,
                    'MSE': mse
                })

# Convert results to DataFrame and save
regression_df = pd.DataFrame(regression_results)
regression_df.to_csv('output/linear_regression/regression_analysis/regression_summary.csv', index=False)

# Generate summary statistics by year and predictor
print("\nRegression Summary Statistics")
print("============================")

# Group by year and calculate mean R-squared
year_summary = regression_df.groupby('Year')['R_squared'].mean().reset_index()
print("\nAverage R-squared by Year:")
print(year_summary)

# Group by predictor and calculate mean R-squared
predictor_summary = regression_df.groupby('Predictor')['R_squared'].mean().reset_index()
print("\nAverage R-squared by Predictor:")
print(predictor_summary)

# Group by crime type and calculate mean R-squared
crime_summary = regression_df.groupby('Crime_Type')['R_squared'].mean().reset_index()
print("\nAverage R-squared by Crime Type:")
print(crime_summary)

# Identify strongest relationships (highest R-squared)
print("\nTop 5 Strongest Relationships (by R-squared):")
top_models = regression_df.sort_values('R_squared', ascending=False).head(5)
for _, row in top_models.iterrows():
    print(f"{row['Crime_Type']} ({row['Year']}) ~ {row['Predictor']}: R² = {row['R_squared']:.3f}")

print("\nLinear regression analysis complete - all files saved to output/linear_regression directory")
