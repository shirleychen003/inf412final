#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count Model Fitness Check
Evaluates the fitness of Poisson and Negative Binomial regression models
by implementing additional goodness-of-fit tests and assumption checks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from scipy import stats
from scipy.stats import poisson, nbinom, chi2
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import functions from count_models module
from count_models import load_data, clean_neighborhood_name

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory
output_dir = 'output/count_model_fitness'
os.makedirs(output_dir, exist_ok=True)

def check_deviance_goodness_of_fit(model):
    """
    Perform deviance goodness-of-fit test.
    H0: The model fits the data well.
    H1: The model does not fit the data well.
    """
    # Calculate degrees of freedom
    df = model.df_resid
    
    # Get deviance
    deviance = model.deviance
    
    # Compute p-value based on chi-square distribution
    p_value = 1 - chi2.cdf(deviance, df)
    
    return {
        'deviance': deviance,
        'df': df,
        'p_value': p_value,
        'conclusion': 'Good fit' if p_value > 0.05 else 'Poor fit'
    }

def check_pearson_chi2_test(model, observed):
    """
    Perform Pearson chi-square goodness-of-fit test.
    H0: The model fits the data well.
    H1: The model does not fit the data well.
    """
    # Calculate predicted values
    predicted = model.predict()
    
    # Calculate Pearson residuals
    pearson_resid = (observed - predicted) / np.sqrt(predicted)
    
    # Calculate Pearson chi-square statistic
    pearson_chi2 = np.sum(pearson_resid**2)
    
    # Calculate degrees of freedom
    df = model.df_resid
    
    # Compute p-value based on chi-square distribution
    p_value = 1 - chi2.cdf(pearson_chi2, df)
    
    return {
        'pearson_chi2': pearson_chi2,
        'df': df,
        'p_value': p_value,
        'conclusion': 'Good fit' if p_value > 0.05 else 'Poor fit'
    }

def check_zero_inflation(observed):
    """Check if data has excess zeros indicating zero-inflation"""
    num_zeros = np.sum(observed == 0)
    prop_zeros = num_zeros / len(observed)
    
    # Calculate expected zeros under Poisson
    lambda_mle = np.mean(observed)
    expected_zero_prop = np.exp(-lambda_mle)
    
    return {
        'observed_zero_prop': prop_zeros,
        'expected_zero_prop': expected_zero_prop,
        'zero_inflated': prop_zeros > 1.5 * expected_zero_prop
    }

def check_aic_bic(poisson_model, nb_model=None):
    """Compare AIC and BIC between models"""
    result = {
        'poisson_aic': poisson_model.aic,
        'poisson_bic': poisson_model.bic
    }
    
    if nb_model is not None:
        result.update({
            'nb_aic': nb_model.aic,
            'nb_bic': nb_model.bic,
            'preferred_model': 'Negative Binomial' if nb_model.aic < poisson_model.aic else 'Poisson'
        })
    else:
        result.update({
            'preferred_model': 'Poisson'
        })
    
    return result

def check_rmse(observed, poisson_pred, nb_pred=None):
    """Calculate RMSE for each model"""
    poisson_rmse = np.sqrt(mean_squared_error(observed, poisson_pred))
    
    result = {
        'poisson_rmse': poisson_rmse
    }
    
    if nb_pred is not None:
        nb_rmse = np.sqrt(mean_squared_error(observed, nb_pred))
        result.update({
            'nb_rmse': nb_rmse,
            'preferred_model': 'Negative Binomial' if nb_rmse < poisson_rmse else 'Poisson'
        })
    
    return result

def plot_qq_residuals(model, observed, title, output_path):
    """Generate QQ plot of residuals to check distributional assumptions"""
    # Calculate Pearson residuals
    predicted = model.predict()
    pearson_resid = (observed - predicted) / np.sqrt(predicted)
    
    # Create QQ plot
    plt.figure(figsize=(10, 6))
    stats.probplot(pearson_resid, dist="norm", plot=plt)
    plt.title(f'QQ Plot of Pearson Residuals - {title}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_fitness(crime_type, predictor, year):
    """Run comprehensive fitness checks for a given model"""
    print(f"\n{'='*50}")
    print(f"Analyzing fitness of {crime_type} ~ {predictor} for {year}")
    print(f"{'='*50}")
    
    # Load data
    data = load_data(year)
    
    # Get observed counts
    observed = data[f'{crime_type}_{year}']
    
    # Calculate dispersion to decide on model
    mean_val = np.mean(observed)
    var_val = np.var(observed)
    dispersion = var_val / mean_val
    
    print(f"Mean: {mean_val:.2f}, Variance: {var_val:.2f}")
    print(f"Dispersion (Variance/Mean): {dispersion:.2f}")
    
    # Fit Poisson model
    formula = f"{crime_type}_{year} ~ {predictor}"
    poisson_model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()
    
    # Check zero-inflation
    zero_check = check_zero_inflation(observed)
    print(f"\nZero-Inflation Check:")
    print(f"Observed zero proportion: {zero_check['observed_zero_prop']:.3f}")
    print(f"Expected zero proportion under Poisson: {zero_check['expected_zero_prop']:.3f}")
    print(f"Zero-inflated: {zero_check['zero_inflated']}")
    
    # Poisson goodness-of-fit tests
    deviance_test = check_deviance_goodness_of_fit(poisson_model)
    print(f"\nPoisson Deviance GoF Test:")
    print(f"Deviance: {deviance_test['deviance']:.2f}, df: {deviance_test['df']}")
    print(f"p-value: {deviance_test['p_value']:.4f} ({deviance_test['conclusion']})")
    
    pearson_test = check_pearson_chi2_test(poisson_model, observed)
    print(f"\nPoisson Pearson Chi² Test:")
    print(f"Pearson Chi²: {pearson_test['pearson_chi2']:.2f}, df: {pearson_test['df']}")
    print(f"p-value: {pearson_test['p_value']:.4f} ({pearson_test['conclusion']})")
    
    # Store model predictions
    poisson_pred = poisson_model.predict()
    
    # QQ plot for Poisson
    plot_qq_residuals(
        poisson_model, 
        observed, 
        f"Poisson Model - {crime_type} ~ {predictor} ({year})", 
        f"{output_dir}/{crime_type}_{predictor}_{year}_poisson_qq.png"
    )
    
    # Fit Negative Binomial model if there's overdispersion
    nb_model = None
    nb_pred = None
    if dispersion > 1.5:
        print(f"\nDispersion > 1.5, fitting Negative Binomial model")
        try:
            nb_model = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial()).fit()
            nb_pred = nb_model.predict()
            
            # NB goodness-of-fit tests
            nb_deviance_test = check_deviance_goodness_of_fit(nb_model)
            print(f"\nNB Deviance GoF Test:")
            print(f"Deviance: {nb_deviance_test['deviance']:.2f}, df: {nb_deviance_test['df']}")
            print(f"p-value: {nb_deviance_test['p_value']:.4f} ({nb_deviance_test['conclusion']})")
            
            nb_pearson_test = check_pearson_chi2_test(nb_model, observed)
            print(f"\nNB Pearson Chi² Test:")
            print(f"Pearson Chi²: {nb_pearson_test['pearson_chi2']:.2f}, df: {nb_pearson_test['df']}")
            print(f"p-value: {nb_pearson_test['p_value']:.4f} ({nb_pearson_test['conclusion']})")
            
            # QQ plot for NB
            plot_qq_residuals(
                nb_model, 
                observed, 
                f"Negative Binomial Model - {crime_type} ~ {predictor} ({year})", 
                f"{output_dir}/{crime_type}_{predictor}_{year}_nb_qq.png"
            )
        except:
            print("Negative Binomial model failed to converge")
    
    # Compare models using AIC/BIC
    aic_bic_comparison = check_aic_bic(poisson_model, nb_model)
    print(f"\nModel Comparison (AIC/BIC):")
    print(f"Poisson AIC: {aic_bic_comparison['poisson_aic']:.2f}, BIC: {aic_bic_comparison['poisson_bic']:.2f}")
    if nb_model is not None:
        print(f"NB AIC: {aic_bic_comparison['nb_aic']:.2f}, BIC: {aic_bic_comparison['nb_bic']:.2f}")
    print(f"Preferred model (AIC): {aic_bic_comparison['preferred_model']}")
    
    # Compare models using RMSE
    rmse_comparison = check_rmse(observed, poisson_pred, nb_pred)
    print(f"\nModel Comparison (RMSE):")
    print(f"Poisson RMSE: {rmse_comparison['poisson_rmse']:.2f}")
    if nb_pred is not None:
        print(f"NB RMSE: {rmse_comparison['nb_rmse']:.2f}")
        print(f"Preferred model (RMSE): {rmse_comparison['preferred_model']}")
    
    # Calculate R values (correlation coefficients)
    poisson_r = np.corrcoef(observed, poisson_pred)[0,1]
    result = {
        'crime_type': crime_type,
        'predictor': predictor,
        'year': year,
        'dispersion': dispersion,
        'zero_inflated': zero_check['zero_inflated'],
        'poisson_deviance_p': deviance_test['p_value'],
        'poisson_pearson_p': pearson_test['p_value'],
        'poisson_aic': poisson_model.aic,
        'poisson_bic': poisson_model.bic,
        'poisson_rmse': rmse_comparison['poisson_rmse'],
        'poisson_r': poisson_r  # Add R value
    }
    
    if nb_model is not None:
        nb_r = np.corrcoef(observed, nb_pred)[0,1]  # Calculate NB R value
        result.update({
            'nb_deviance_p': nb_deviance_test['p_value'],
            'nb_pearson_p': nb_pearson_test['p_value'],
            'nb_aic': nb_model.aic,
            'nb_bic': nb_model.bic,
            'nb_rmse': rmse_comparison['nb_rmse'],
            'nb_r': nb_r,  # Add NB R value
            'preferred_aic': aic_bic_comparison['preferred_model'],
            'preferred_rmse': rmse_comparison['preferred_model']
        })
    
    return result

def create_fitness_summary(all_results):
    """Create a comprehensive summary of all fitness checks"""
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary to CSV
    results_df.to_csv(f"{output_dir}/count_model_fitness_summary.csv", index=False)
    
    # Create heatmap of fitness metrics
    plt.figure(figsize=(12, 8))
    
    # Get only numeric columns for heatmap
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_results = results_df.pivot_table(
        index='crime_type', 
        columns=['predictor', 'year'], 
        values='dispersion'
    )
    
    sns.heatmap(numeric_results, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Dispersion (Variance/Mean) by Crime Type and Predictor')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dispersion_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar chart comparing AIC values
    if 'nb_aic' in results_df.columns:
        # Prepare data for plotting
        plot_data = []
        for _, row in results_df.iterrows():
            plot_data.append({
                'crime_type': row['crime_type'],
                'predictor': row['predictor'],
                'year': row['year'],
                'model': 'Poisson',
                'AIC': row['poisson_aic']
            })
            plot_data.append({
                'crime_type': row['crime_type'],
                'predictor': row['predictor'],
                'year': row['year'],
                'model': 'Negative Binomial',
                'AIC': row['nb_aic'] if not pd.isna(row['nb_aic']) else None
            })
        
        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df.dropna()
        
        # Create grouped bar chart
        plt.figure(figsize=(14, 10))
        g = sns.catplot(
            data=plot_df, kind="bar",
            x="crime_type", y="AIC", hue="model",
            col="predictor", row="year",
            ci=None, palette="dark", alpha=.6, height=6, aspect=1.5
        )
        g.set_axis_labels("Crime Type", "AIC Value")
        g.set_titles("{col_name} - {row_name}")
        g.tight_layout()
        plt.savefig(f"{output_dir}/aic_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # NEW: Create visualization for Pearson chi-square tests
    if 'poisson_pearson_p' in results_df.columns:
        # Create heatmap of p-values for Pearson chi-square tests
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for Poisson Pearson chi-square p-values
        pearson_poisson_results = results_df.pivot_table(
            index='crime_type', 
            columns=['predictor', 'year'], 
            values='poisson_pearson_p'
        )
        
        # Use log scale for better visualization since p-values are very small
        pearson_poisson_results = -np.log10(pearson_poisson_results + 1e-10)  # Add small value to avoid log(0)
        
        sns.heatmap(pearson_poisson_results, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Poisson Pearson Chi-Square Test (-log10 p-value)\nHigher values indicate stronger evidence against model fit')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pearson_poisson_pvalue_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # If NB results are available, create heatmap for NB Pearson chi-square p-values
        if 'nb_pearson_p' in results_df.columns:
            plt.figure(figsize=(12, 8))
            
            pearson_nb_results = results_df.pivot_table(
                index='crime_type', 
                columns=['predictor', 'year'], 
                values='nb_pearson_p'
            )
            
            pearson_nb_results = -np.log10(pearson_nb_results + 1e-10)  # Add small value to avoid log(0)
            
            sns.heatmap(pearson_nb_results, annot=True, cmap='YlOrRd', fmt='.2f')
            plt.title('Negative Binomial Pearson Chi-Square Test (-log10 p-value)\nHigher values indicate stronger evidence against model fit')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pearson_nb_pvalue_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # NEW: Create visualization for RMSE comparisons
    if 'poisson_rmse' in results_df.columns:
        # Prepare data for plotting
        rmse_data = []
        for _, row in results_df.iterrows():
            rmse_data.append({
                'crime_type': row['crime_type'],
                'predictor': row['predictor'],
                'year': row['year'],
                'model': 'Poisson',
                'RMSE': row['poisson_rmse']
            })
            if 'nb_rmse' in results_df.columns:
                rmse_data.append({
                    'crime_type': row['crime_type'],
                    'predictor': row['predictor'],
                    'year': row['year'],
                    'model': 'Negative Binomial',
                    'RMSE': row['nb_rmse'] if not pd.isna(row['nb_rmse']) else None
                })
        
        rmse_df = pd.DataFrame(rmse_data)
        rmse_df = rmse_df.dropna()
        
        # Create grouped bar chart for RMSE
        plt.figure(figsize=(14, 10))
        g = sns.catplot(
            data=rmse_df, kind="bar",
            x="crime_type", y="RMSE", hue="model",
            col="predictor", row="year",
            ci=None, palette="Set2", alpha=.7, height=6, aspect=1.5
        )
        g.set_axis_labels("Crime Type", "RMSE Value")
        g.set_titles("{col_name} - {row_name}")
        g.tight_layout()
        plt.savefig(f"{output_dir}/rmse_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap showing the percent difference in RMSE
        if 'nb_rmse' in results_df.columns:
            # Calculate percent difference: (NB RMSE - Poisson RMSE) / Poisson RMSE * 100
            results_df['rmse_percent_diff'] = (results_df['nb_rmse'] - results_df['poisson_rmse']) / results_df['poisson_rmse'] * 100
            
            plt.figure(figsize=(12, 8))
            rmse_diff_results = results_df.pivot_table(
                index='crime_type', 
                columns=['predictor', 'year'], 
                values='rmse_percent_diff'
            )
            
            # Use a diverging colormap centered at 0
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(rmse_diff_results, annot=True, cmap=cmap, center=0, fmt='.2f')
            plt.title('RMSE Percent Difference: (NB - Poisson)/Poisson\nNegative values favor Poisson, positive values favor NB')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/rmse_percent_difference_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # NEW: Create visualization for R values
    if 'poisson_r' in results_df.columns:
        # Create heatmap of R values
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for Poisson R values
        r_poisson_results = results_df.pivot_table(
            index='crime_type', 
            columns=['predictor', 'year'], 
            values='poisson_r'
        )
        
        sns.heatmap(r_poisson_results, annot=True, cmap='RdYlBu', fmt='.3f', center=0, vmin=-1, vmax=1)
        plt.title('Poisson Model Correlation Coefficients (R)\nBlue indicates positive correlation, Red indicates negative')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/poisson_r_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # If NB results are available, create heatmap for NB R values
        if 'nb_r' in results_df.columns:
            plt.figure(figsize=(12, 8))
            
            r_nb_results = results_df.pivot_table(
                index='crime_type', 
                columns=['predictor', 'year'], 
                values='nb_r'
            )
            
            sns.heatmap(r_nb_results, annot=True, cmap='RdYlBu', fmt='.3f', center=0, vmin=-1, vmax=1)
            plt.title('Negative Binomial Model Correlation Coefficients (R)\nBlue indicates positive correlation, Red indicates negative')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/nb_r_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a comparison plot showing R values side by side
            plt.figure(figsize=(14, 10))
            r_comparison_data = []
            for _, row in results_df.iterrows():
                r_comparison_data.append({
                    'crime_type': row['crime_type'],
                    'predictor': row['predictor'],
                    'year': row['year'],
                    'model': 'Poisson',
                    'R': row['poisson_r']
                })
                r_comparison_data.append({
                    'crime_type': row['crime_type'],
                    'predictor': row['predictor'],
                    'year': row['year'],
                    'model': 'Negative Binomial',
                    'R': row['nb_r'] if not pd.isna(row['nb_r']) else None
                })
            
            r_comparison_df = pd.DataFrame(r_comparison_data)
            r_comparison_df = r_comparison_df.dropna()
            
            g = sns.catplot(
                data=r_comparison_df, kind="bar",
                x="crime_type", y="R", hue="model",
                col="predictor", row="year",
                ci=None, palette="Set2", alpha=.7, height=6, aspect=1.5
            )
            g.set_axis_labels("Crime Type", "Correlation Coefficient (R)")
            g.set_titles("{col_name} - {row_name}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/r_value_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return results_df

if __name__ == "__main__":
    # Define crime types and predictors to analyze
    crime_types = ['ASSAULT', 'BREAKENTER', 'ROBBERY', 'AUTOTHEFT']
    predictors = ['total_population', 'low_income_percent']
    years = ['2016', '2021']
    
    all_results = []
    
    # Run fitness analyses
    for crime_type in crime_types:
        for predictor in predictors:
            for year in years:
                try:
                    result = analyze_fitness(crime_type, predictor, year)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error analyzing fitness of {crime_type} ~ {predictor} for {year}: {e}")
    
    # Create summary
    if all_results:
        results_df = create_fitness_summary(all_results)
        
        # Print overall conclusions
        print("\n\nOVERALL FITNESS CONCLUSIONS:")
        print("==========================")
        
        # Check proportion of models with good fit
        if 'poisson_deviance_p' in results_df.columns:
            poisson_good_fit = (results_df['poisson_deviance_p'] > 0.05).mean() * 100
            print(f"Poisson models with good fit (deviance test): {poisson_good_fit:.1f}%")
        
        if 'nb_deviance_p' in results_df.columns:
            nb_good_fit = (results_df['nb_deviance_p'] > 0.05).mean() * 100
            print(f"Negative Binomial models with good fit (deviance test): {nb_good_fit:.1f}%")
        
        # Check zero-inflation
        if 'zero_inflated' in results_df.columns:
            zero_inflated_pct = results_df['zero_inflated'].mean() * 100
            print(f"Models with potential zero-inflation: {zero_inflated_pct:.1f}%")
        
        # Check preferred model
        if 'preferred_aic' in results_df.columns:
            nb_preferred = (results_df['preferred_aic'] == 'Negative Binomial').mean() * 100
            print(f"Cases where Negative Binomial is preferred (AIC): {nb_preferred:.1f}%")
        
    print(f"\nCount model fitness analysis complete. Results saved to {output_dir}/")
