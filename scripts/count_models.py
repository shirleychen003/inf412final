#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count Models (Poisson and Negative Binomial) for Crime Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from scipy import stats
from scipy.stats import poisson, nbinom

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory
output_dir = 'output/count_models'
os.makedirs(output_dir, exist_ok=True)

# Helper function to clean neighborhood names
def clean_neighborhood_name(name):
    """Standardize neighborhood names for consistent matching"""
    if not isinstance(name, str):
        return name
    return name.lower().strip().replace('-', ' ').replace('  ', ' ')

# Load and prepare data
def load_data(year):
    """Load crime and income data for a specific year and merge"""
    # Load crime data
    crime_data = pd.read_csv(f'cleaned_data/{year}/cleaned_crime_data_{year}.csv')
    
    # Load income data
    income_data = pd.read_csv(f'cleaned_data/{year}/cleaned_neighbourhood_income_data_{year}.csv')
    
    # Clean neighborhood names for matching
    crime_data['neighborhood_clean'] = crime_data['AREA_NAME'].apply(clean_neighborhood_name)
    income_data['neighborhood_clean'] = income_data['neighbourhood_name'].apply(clean_neighborhood_name)
    
    # Merge datasets
    merged_data = pd.merge(
        crime_data, 
        income_data, 
        left_on='neighborhood_clean', 
        right_on='neighborhood_clean', 
        how='inner'
    )
    
    print(f"Loaded data for {year}. {len(merged_data)} neighborhoods matched.")
    
    return merged_data

# Plot dispersion check for Poisson vs NB decision
def plot_dispersion_check(data, crime_type, predictor, year):
    """
    Check if data is overdispersed (variance > mean) to determine if 
    Negative Binomial is more appropriate than Poisson
    """
    crime_count = data[f'{crime_type}_{year}']
    
    # Calculate mean and variance
    mean_val = np.mean(crime_count)
    var_val = np.var(crime_count)
    dispersion = var_val / mean_val
    
    # Plot histogram with overlaid Poisson PMF
    plt.figure(figsize=(10, 6))
    
    # Histogram of observed data
    sns.histplot(crime_count, stat='density', kde=False, alpha=0.5, label='Observed')
    
    # Poisson PMF based on mean
    x = np.arange(0, max(crime_count) + 1)
    pmf = poisson.pmf(x, mean_val)
    plt.plot(x, pmf, 'ro-', ms=8, label='Poisson PMF')
    
    plt.title(f"{crime_type} Counts vs Poisson Distribution\nMean: {mean_val:.2f}, Variance: {var_val:.2f}, Dispersion: {dispersion:.2f}")
    plt.xlabel(f"{crime_type} Count")
    plt.ylabel("Density")
    plt.legend()
    
    plt.savefig(f"{output_dir}/{crime_type}_{predictor}_{year}_dispersion_check.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_val, var_val, dispersion

# Fit Poisson model
def fit_poisson_model(data, crime_type, predictor, year):
    """Fit Poisson regression model and return results"""
    # Prepare formula
    formula = f"{crime_type}_{year} ~ {predictor}"
    
    # Fit model
    model = smf.glm(formula=formula, data=data, family=sm.families.Poisson())
    result = model.fit()
    
    print(f"\nPoisson Model: {crime_type} ~ {predictor} ({year})")
    print(f"AIC: {result.aic:.2f}")
    print(f"Log-Likelihood: {result.llf:.2f}")
    print(f"Pseudo R²: {1 - (result.deviance / result.null_deviance):.4f}")
    
    # Calculate predictions
    data['pred_poisson'] = result.predict()
    
    # Calculate Pearson residuals
    data['resid_poisson'] = result.resid_pearson
    
    return result, data

# Fit Negative Binomial model
def fit_nb_model(data, crime_type, predictor, year):
    """Fit Negative Binomial regression model and return results"""
    # Prepare formula
    formula = f"{crime_type}_{year} ~ {predictor}"
    
    # Fit model
    model = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial())
    try:
        result = model.fit()
        
        print(f"\nNegative Binomial Model: {crime_type} ~ {predictor} ({year})")
        print(f"AIC: {result.aic:.2f}")
        print(f"Log-Likelihood: {result.llf:.2f}")
        print(f"Pseudo R²: {1 - (result.deviance / result.null_deviance):.4f}")
        
        # Calculate predictions
        data['pred_nb'] = result.predict()
        
        # Calculate Pearson residuals
        data['resid_nb'] = result.resid_pearson
        
        return result, data
    except:
        print("Negative Binomial model failed to converge. Using Poisson results only.")
        return None, data

# Plot diagnostic plots
def plot_diagnostics(data, poisson_model, nb_model, crime_type, predictor, year):
    """Generate diagnostic plots for the models"""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted (Poisson)
    plt.subplot(2, 2, 1)
    plt.scatter(data[f'{crime_type}_{year}'], data['pred_poisson'], alpha=0.6)
    max_val = max(data[f'{crime_type}_{year}'].max(), data['pred_poisson'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Actual Counts')
    plt.ylabel('Predicted Counts (Poisson)')
    plt.title('Actual vs Predicted (Poisson)')
    
    # Plot 2: Pearson Residuals vs Predicted (Poisson)
    plt.subplot(2, 2, 2)
    plt.scatter(data['pred_poisson'], data['resid_poisson'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Counts')
    plt.ylabel('Pearson Residuals')
    plt.title('Residuals vs Predicted (Poisson)')
    
    # If NB model converged, show its plots too
    if nb_model is not None:
        # Plot 3: Actual vs Predicted (NB)
        plt.subplot(2, 2, 3)
        plt.scatter(data[f'{crime_type}_{year}'], data['pred_nb'], alpha=0.6)
        max_val = max(data[f'{crime_type}_{year}'].max(), data['pred_nb'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Actual Counts')
        plt.ylabel('Predicted Counts (NB)')
        plt.title('Actual vs Predicted (Negative Binomial)')
        
        # Plot 4: Pearson Residuals vs Predicted (NB)
        plt.subplot(2, 2, 4)
        plt.scatter(data['pred_nb'], data['resid_nb'], alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Counts')
        plt.ylabel('Pearson Residuals')
        plt.title('Residuals vs Predicted (Negative Binomial)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{crime_type}_{predictor}_{year}_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot model comparison
def plot_model_comparison(data, crime_type, predictor, year):
    """Compare Poisson and NB model predictions side by side"""
    # Only run this if both models are available
    if 'pred_nb' in data.columns:
        plt.figure(figsize=(12, 6))
        
        # Create a long-form dataframe for seaborn
        df_melt = pd.DataFrame({
            'Actual': data[f'{crime_type}_{year}'],
            'Poisson': data['pred_poisson'],
            'Negative Binomial': data['pred_nb']
        })
        
        df_long = pd.melt(df_melt, id_vars=[], value_vars=['Actual', 'Poisson', 'Negative Binomial'],
                         var_name='Model', value_name='Count')
        
        # Box plot
        sns.boxplot(x='Model', y='Count', data=df_long)
        plt.title(f'Distribution of Actual vs Predicted Counts\n{crime_type} ~ {predictor} ({year})')
        
        plt.savefig(f"{output_dir}/{crime_type}_{predictor}_{year}_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

# Save model results
def save_results(data, poisson_model, nb_model, crime_type, predictor, year):
    """Save model predictions and summary"""
    # Create results dataframe
    results_df = data[['AREA_NAME', f'{crime_type}_{year}', 'pred_poisson']].copy()
    results_df.rename(columns={
        'AREA_NAME': 'Neighborhood',
        f'{crime_type}_{year}': 'Actual',
        'pred_poisson': 'Predicted_Poisson'
    }, inplace=True)
    
    if nb_model is not None:
        results_df['Predicted_NB'] = data['pred_nb']
    
    # Save to CSV
    results_df.to_csv(f"{output_dir}/{crime_type}_{predictor}_{year}_predictions.csv", index=False)
    
    # Save model summaries
    with open(f"{output_dir}/{crime_type}_{predictor}_{year}_summary.txt", 'w') as f:
        f.write("POISSON MODEL SUMMARY\n")
        f.write("======================\n\n")
        f.write(str(poisson_model.summary()))
        f.write("\n\n")
        
        if nb_model is not None:
            f.write("NEGATIVE BINOMIAL MODEL SUMMARY\n")
            f.write("===============================\n\n")
            f.write(str(nb_model.summary()))

# Main analysis function for a single model
def analyze_count_model(crime_type, predictor, year):
    """Run full analysis for a specific crime type, predictor, and year"""
    print(f"\n{'='*50}")
    print(f"Analyzing {crime_type} ~ {predictor} for {year}")
    print(f"{'='*50}")
    
    # Load data
    data = load_data(year)
    
    # Check for overdispersion
    mean_val, var_val, dispersion = plot_dispersion_check(data, crime_type, predictor, year)
    
    # Determine which model to use based on dispersion
    if dispersion > 1.5:
        print(f"Data is overdispersed (dispersion = {dispersion:.2f}). Fitting both Poisson and Negative Binomial models.")
        fit_nb = True
    else:
        print(f"Data is not heavily overdispersed (dispersion = {dispersion:.2f}). Fitting Poisson model only.")
        fit_nb = False
    
    # Fit Poisson model
    poisson_model, data = fit_poisson_model(data, crime_type, predictor, year)
    
    # Fit Negative Binomial model if needed
    if fit_nb:
        nb_model, data = fit_nb_model(data, crime_type, predictor, year)
    else:
        nb_model = None
    
    # Generate diagnostic plots
    plot_diagnostics(data, poisson_model, nb_model, crime_type, predictor, year)
    
    # Plot model comparison if both models are available
    if nb_model is not None:
        plot_model_comparison(data, crime_type, predictor, year)
    
    # Save results
    save_results(data, poisson_model, nb_model, crime_type, predictor, year)
    
    return {
        'crime_type': crime_type,
        'predictor': predictor,
        'year': year,
        'dispersion': dispersion,
        'poisson_aic': poisson_model.aic,
        'poisson_pseudo_r2': 1 - (poisson_model.deviance / poisson_model.null_deviance),
        'nb_aic': nb_model.aic if nb_model is not None else None,
        'nb_pseudo_r2': 1 - (nb_model.deviance / nb_model.null_deviance) if nb_model is not None else None
    }

# Create summary comparison of all models
def create_summary_heatmap(all_results):
    """Create a heatmap showing pseudo R² values for all models"""
    # Convert results to dataframe
    results_df = pd.DataFrame(all_results)
    
    # Create pivot table for Poisson model results
    pivot_poisson = results_df.pivot_table(
        index='crime_type', 
        columns=['predictor', 'year'], 
        values='poisson_pseudo_r2'
    )
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_poisson, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Poisson Model Pseudo R² by Crime Type and Predictor')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/poisson_pseudo_r2_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create pivot table for NB model results if available
    if not all(pd.isna(results_df['nb_pseudo_r2'])):
        pivot_nb = results_df.pivot_table(
            index='crime_type', 
            columns=['predictor', 'year'], 
            values='nb_pseudo_r2'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_nb, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('Negative Binomial Model Pseudo R² by Crime Type and Predictor')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nb_pseudo_r2_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save summary CSV
    results_df.to_csv(f"{output_dir}/count_models_summary.csv", index=False)

if __name__ == "__main__":
    # Define crime types and predictors to analyze
    crime_types = ['ASSAULT', 'BREAKENTER', 'ROBBERY', 'AUTOTHEFT']
    predictors = ['total_population', 'low_income_percent']
    years = ['2016', '2021']
    
    all_results = []
    
    # Run analyses
    for crime_type in crime_types:
        for predictor in predictors:
            for year in years:
                try:
                    result = analyze_count_model(crime_type, predictor, year)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {crime_type} ~ {predictor} for {year}: {e}")
    
    # Create summary visualizations
    if all_results:
        create_summary_heatmap(all_results)
        
    print(f"\nCount models analysis complete. Results saved to {output_dir}/") 