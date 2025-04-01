import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import os
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('output/fitness', exist_ok=True)

# Function to clean neighborhood names for consistent matching
def clean_name(name):
    name = str(name).strip()
    name = name.replace("St. ", "St.")
    name = name.replace("St ", "St.")
    name = name.replace("-East ", "-East")
    name = name.replace("O`Connor", "O'Connor")
    name = name.replace(" - ", "-")
    return name

# Load data
print("Loading data...")
# Use absolute paths
nbh_data_2016 = pd.read_csv("cleaned_data/2016/cleaned_neighbourhood_income_data_2016.csv")
crime_data_2016 = pd.read_csv("cleaned_data/2016/cleaned_crime_data_2016.csv")
nbh_data_2021 = pd.read_csv("cleaned_data/2021/cleaned_neighbourhood_income_data_2021.csv")
crime_data_2021 = pd.read_csv("cleaned_data/2021/cleaned_crime_data_2021.csv")

print("Data loaded successfully!")
print(f"2016 neighborhood data shape: {nbh_data_2016.shape}")
print(f"2016 crime data shape: {crime_data_2016.shape}")
print(f"2021 neighborhood data shape: {nbh_data_2021.shape}")
print(f"2021 crime data shape: {crime_data_2021.shape}")

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

# Define predictor variables
predictors = ['low_income_percent', 'total_population']

# Function to calculate additional fitness metrics
def calculate_fitness_metrics(X, y, model):
    """Calculate various model fitness metrics for the regression model"""
    # Add constant for statsmodels
    X_sm = sm.add_constant(X)
    
    # Fit statsmodels OLS for additional diagnostics
    model_sm = sm.OLS(y, X_sm).fit()
    
    # Predictions from scikit-learn model
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Basic metrics
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    # Normality of residuals (Shapiro-Wilk test)
    _, normality_p = stats.shapiro(residuals)
    
    # Homoscedasticity (Breusch-Pagan test)
    try:
        _, bp_p_value, _, _ = het_breuschpagan(residuals, X_sm)
    except:
        bp_p_value = np.nan
    
    # Durbin-Watson statistic for autocorrelation
    dw_stat = sm.stats.stattools.durbin_watson(residuals)
    
    # F-statistic and p-value
    f_stat = model_sm.fvalue
    f_pvalue = model_sm.f_pvalue
    
    # AIC and BIC
    aic = model_sm.aic
    bic = model_sm.bic
    
    return {
        'R_squared': r2,
        'Adjusted_R_squared': adj_r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Shapiro_p_value': normality_p,
        'BP_p_value': bp_p_value,
        'Durbin_Watson': dw_stat,
        'F_statistic': f_stat,
        'F_p_value': f_pvalue,
        'AIC': aic,
        'BIC': bic,
        'Residuals': residuals,
        'Predictions': y_pred,
        'Y_true': y,
        'X': X
    }

# Function to generate diagnostic plots
def generate_diagnostic_plots(metrics, year, crime_type, predictor):
    """Generate diagnostic plots for the regression model"""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{crime_type} ~ {predictor} ({year}): Diagnostic Plots", fontsize=16)
    
    # Extract data
    residuals = metrics['Residuals']
    y_pred = metrics['Predictions']
    y_true = metrics['Y_true']
    X = metrics['X']
    
    # Plot 1: Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Add a LOWESS smoothed line
    try:
        lowess = sm.nonparametric.lowess(residuals, y_pred, frac=0.6)
        axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color='red', lw=2)
    except:
        pass
    
    # Plot 2: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    # Plot 3: Scale-Location Plot (Square root of standardized residuals vs Fitted)
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location')
    
    # Try to add a LOWESS smoothed line
    try:
        lowess = sm.nonparametric.lowess(np.sqrt(np.abs(standardized_residuals)), y_pred, frac=0.6)
        axes[1, 0].plot(lowess[:, 0], lowess[:, 1], color='red', lw=2)
    except:
        pass
    
    # Plot 4: Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1, 1].set_xlabel('Actual values')
    axes[1, 1].set_ylabel('Predicted values')
    axes[1, 1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    filename = f"output/fitness/{crime_type}_{predictor}_{year}_diagnostics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# Collect all results here
fitness_results = []
plot_filenames = []

# Process each year, crime type, and predictor combination
print("Analyzing regression models...")
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
                print(f"Processing: {crime_type} ~ {predictor} ({year})")
                X = df[[predictor]]
                y = df[crime_col]
                
                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate metrics
                metrics = calculate_fitness_metrics(X, y, model)
                
                # Generate plots
                plot_filename = generate_diagnostic_plots(metrics, year, crime_type, predictor)
                plot_filenames.append(plot_filename)
                
                # Add model information
                result = {
                    'Year': year,
                    'Crime_Type': crime_type,
                    'Predictor': predictor,
                    'Coefficient': model.coef_[0],
                    'Intercept': model.intercept_,
                    **{k: v for k, v in metrics.items() if k not in ['Residuals', 'Predictions', 'Y_true', 'X']}
                }
                
                fitness_results.append(result)

# Convert results to DataFrame and save
fitness_df = pd.DataFrame(fitness_results)
fitness_df.to_csv('output/fitness/regression_fitness_metrics.csv', index=False)

# Generate a summary report
summary_df = fitness_df.groupby(['Year', 'Predictor']).agg({
    'R_squared': ['mean', 'min', 'max'],
    'Adjusted_R_squared': ['mean', 'min', 'max'],
    'RMSE': ['mean', 'min', 'max'],
    'MAE': ['mean', 'min', 'max'],
    'Shapiro_p_value': ['mean', 'min', 'max'],
    'BP_p_value': ['mean', 'min', 'max'],
    'Durbin_Watson': ['mean', 'min', 'max']
}).reset_index()

summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
summary_df.to_csv('output/fitness/regression_fitness_summary.csv', index=False)

# Create a heatmap of R-squared values
plt.figure(figsize=(12, 8))
pivot_table = fitness_df.pivot_table(
    index='Crime_Type', 
    columns=['Year', 'Predictor'], 
    values='R_squared'
)

# Sort crime types by average R-squared
crime_order = fitness_df.groupby('Crime_Type')['R_squared'].mean().sort_values(ascending=False).index

# Reorder pivot table
pivot_table = pivot_table.reindex(crime_order)

# Create heatmap
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
plt.title('R-squared Values by Crime Type, Year, and Predictor', fontsize=16)
plt.tight_layout()
plt.savefig('output/fitness/r_squared_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a bar chart comparing different model fit metrics
metrics_to_compare = ['R_squared', 'Adjusted_R_squared']
filtered_df = fitness_df[fitness_df['Crime_Type'] == 'TOTAL_CRIME_COUNT']

plt.figure(figsize=(12, 8))
filtered_df_melted = pd.melt(
    filtered_df, 
    id_vars=['Year', 'Predictor'], 
    value_vars=metrics_to_compare,
    var_name='Metric', 
    value_name='Value'
)

# Create grouped bar chart
sns.barplot(
    data=filtered_df_melted,
    x='Predictor', 
    y='Value', 
    hue='Year',
    palette='viridis'
)

plt.title('Comparison of Model Fit Metrics for Total Crime Count', fontsize=16)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('output/fitness/total_crime_metrics_comparison.png', dpi=300)
plt.close()

# Generate composite plot for residual analysis
plt.figure(figsize=(15, 10))

for i, predictor in enumerate(predictors):
    for j, year in enumerate([2016, 2021]):
        # Get total crime data for this predictor and year
        filtered_data = fitness_df[(fitness_df['Crime_Type'] == 'TOTAL_CRIME_COUNT') & 
                                   (fitness_df['Predictor'] == predictor) & 
                                   (fitness_df['Year'] == year)]
        
        if not filtered_data.empty:
            # Calculate normality test results
            shapiro_p = filtered_data['Shapiro_p_value'].values[0]
            bp_p = filtered_data['BP_p_value'].values[0]
            dw = filtered_data['Durbin_Watson'].values[0]
            
            plt.subplot(2, 2, i*2 + j + 1)
            
            # Add titles with fitness metrics
            plt.title(f"Total Crime ~ {predictor} ({year})\n" +
                      f"Shapiro p={shapiro_p:.3f} | BP p={bp_p:.3f} | DW={dw:.2f}", 
                      fontsize=12)
            
            # Add annotations explaining the tests
            annotation_text = (
                "Shapiro-Wilk: tests normality of residuals (p>0.05 = normal)\n"
                "Breusch-Pagan: tests homoscedasticity (p>0.05 = equal variance)\n"
                "Durbin-Watson: tests autocorrelation (closer to 2 = no autocorrelation)"
            )
            plt.annotate(annotation_text, xy=(0.5, 0.05), xycoords='axes fraction', 
                        ha='center', va='bottom', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Add a visual indicator of fitness
            is_normal = shapiro_p > 0.05
            is_homoscedastic = bp_p > 0.05
            is_no_autocorr = 1.5 < dw < 2.5
            
            plt.text(0.5, 0.8, f"Model Assumptions Met:", 
                    ha='center', transform=plt.gca().transAxes, fontsize=11)
            
            plt.text(0.5, 0.75, f"Normality: {'✅' if is_normal else '❌'}", 
                    ha='center', transform=plt.gca().transAxes, fontsize=10)
            
            plt.text(0.5, 0.7, f"Equal Variance: {'✅' if is_homoscedastic else '❌'}", 
                    ha='center', transform=plt.gca().transAxes, fontsize=10)
            
            plt.text(0.5, 0.65, f"No Autocorrelation: {'✅' if is_no_autocorr else '❌'}", 
                    ha='center', transform=plt.gca().transAxes, fontsize=10)
            
            # Remove axes for a cleaner look
            plt.axis('off')

plt.suptitle('Model Assumptions Check for Total Crime Regressions', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('output/fitness/model_assumptions_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAnalysis complete. Results saved to the 'output/fitness' directory.")
print(f"Generated {len(plot_filenames)} diagnostic plot files.")
print(f"Summary metrics saved to regression_fitness_metrics.csv and regression_fitness_summary.csv")
print(f"Check r_squared_heatmap.png for a visual comparison of model performance.") 