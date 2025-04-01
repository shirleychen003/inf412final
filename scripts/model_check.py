import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot

# Set random seed for reproducibility
np.random.seed(888)

# Create output directory
os.makedirs('output/model_check', exist_ok=True)

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
nbh_data_2016 = pd.read_csv("cleaned_data/2016/cleaned_neighbourhood_income_data_2016.csv")
crime_data_2016 = pd.read_csv("cleaned_data/2016/cleaned_crime_data_2016.csv")

# Clean column names and neighborhood names
nbh_data_2016.columns = nbh_data_2016.columns.str.strip()
crime_data_2016.columns = crime_data_2016.columns.str.strip()

# Apply name cleaning
nbh_data_2016['neighbourhood_name'] = nbh_data_2016['neighbourhood_name'].apply(clean_name)
crime_data_2016['AREA_NAME'] = crime_data_2016['AREA_NAME'].apply(clean_name)

# Merge datasets
merged_2016 = pd.merge(
    nbh_data_2016, 
    crime_data_2016,
    left_on='neighbourhood_name',
    right_on='AREA_NAME',
    how='inner'
)

print(f"Number of matched neighborhoods: {len(merged_2016)}")

# Creating the most important model: BREAKENTER with population (highest R² in 2016)
X = merged_2016[['total_population']]
y = merged_2016['BREAKENTER_2016']

# Add constant for statsmodels (equivalent to intercept)
X_sm = sm.add_constant(X)

# Fit the model using statsmodels for more detailed diagnostics
model = sm.OLS(y, X_sm).fit()
print("\nModel Summary:")
print(model.summary())

# Generate predictions and residuals
y_pred = model.predict(X_sm)
residuals = y - y_pred

# Create a DataFrame with actual values, predictions, and residuals
results_df = pd.DataFrame({
    'Neighborhood': merged_2016['neighbourhood_name'],
    'Population': merged_2016['total_population'],
    'Actual': y,
    'Predicted': y_pred,
    'Residuals': residuals
})

# Save results to CSV
results_df.to_csv('output/model_check/breakenter_population_predictions.csv', index=False)

# 1. Posterior Predictive Check Equivalent: Bootstrap simulation of possible datasets
# This is a frequentist analog to Bayesian posterior predictive checks
n_simulations = 100
n_samples = len(y)
simulated_datasets = []

for i in range(n_simulations):
    # Resample residuals with replacement
    resampled_residuals = np.random.choice(residuals, size=n_samples, replace=True)
    # Generate new y values by adding resampled residuals to predictions
    simulated_y = y_pred + resampled_residuals
    simulated_datasets.append(simulated_y)

# Plot the equivalent of pp_check
plt.figure(figsize=(10, 6))
# Plot simulated datasets
for i, sim_data in enumerate(simulated_datasets):
    if i < 50:  # Only plot 50 simulations to avoid overcrowding
        sns.kdeplot(sim_data, color='lightgrey', alpha=0.5, linewidth=0.5)

# Plot actual data density
sns.kdeplot(y, color='black', linewidth=2, label='Actual Data')

plt.title('Model Check: Distribution of Actual vs. Simulated Break & Enter Counts', fontsize=14)
plt.xlabel('Break & Enter Count', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('output/model_check/predictive_check.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Coefficient Uncertainty: Bootstrap confidence intervals
# This is a frequentist analog to posterior vs. prior plots
n_bootstrap = 1000
bootstrap_coefs = np.zeros((n_bootstrap, 2))  # For intercept and slope

for i in range(n_bootstrap):
    # Bootstrap sample
    indices = np.random.choice(range(len(y)), size=len(y), replace=True)
    X_boot = X.values[indices]
    y_boot = y.values[indices]
    
    # Add constant
    X_boot_sm = np.column_stack((np.ones(len(X_boot)), X_boot))
    
    # Fit model and store coefficients
    model_boot = sm.OLS(y_boot, X_boot_sm).fit()
    bootstrap_coefs[i, 0] = model_boot.params[0]  # Intercept
    bootstrap_coefs[i, 1] = model_boot.params[1]  # Slope

# Plot coefficient distributions
plt.figure(figsize=(10, 8))

# Plot coefficient distributions
plt.subplot(2, 1, 1)
sns.histplot(bootstrap_coefs[:, 0], kde=True, color='blue')
plt.axvline(model.params[0], color='red', linestyle='--', label=f'Point Estimate: {model.params[0]:.4f}')
plt.title('Bootstrap Distribution of Intercept', fontsize=12)
plt.xlabel('Intercept Value', fontsize=10)
plt.legend()

plt.subplot(2, 1, 2)
sns.histplot(bootstrap_coefs[:, 1], kde=True, color='green')
plt.axvline(model.params[1], color='red', linestyle='--', label=f'Point Estimate: {model.params[1]:.4f}')
plt.title('Bootstrap Distribution of Population Coefficient', fontsize=12)
plt.xlabel('Coefficient Value', fontsize=10)
plt.legend()

plt.tight_layout()
plt.savefig('output/model_check/coefficient_uncertainty.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Diagnostic Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Diagnostic Plots for Break & Enter ~ Population Model', fontsize=16)

# Residuals vs Fitted
axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Add smoothed line
try:
    lowess = sm.nonparametric.lowess(residuals, y_pred, frac=0.6)
    axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color='red', lw=2)
except:
    pass

# QQ plot of residuals
probplot = ProbPlot(residuals)
probplot.qqplot(ax=axes[0, 1], line='45')
axes[0, 1].set_title('Q-Q Plot')

# Scale-Location plot
standardized_residuals = residuals / np.std(residuals)
axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Standardized residuals|')
axes[1, 0].set_title('Scale-Location')

# Add smoothed line
try:
    lowess = sm.nonparametric.lowess(np.sqrt(np.abs(standardized_residuals)), y_pred, frac=0.6)
    axes[1, 0].plot(lowess[:, 0], lowess[:, 1], color='red', lw=2)
except:
    pass

# Actual vs Predicted
axes[1, 1].scatter(y, y_pred, alpha=0.6)
axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[1, 1].set_xlabel('Actual Break & Enter Count')
axes[1, 1].set_ylabel('Predicted Break & Enter Count')
axes[1, 1].set_title('Actual vs Predicted')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
plt.savefig('output/model_check/diagnostic_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Model evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"R²: {model.rsquared:.4f}")
print(f"Adjusted R²: {model.rsquared_adj:.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
print(f"F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.8f}")

# Normality test of residuals
stat, p_value = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test for normality of residuals:")
print(f"Statistic: {stat:.4f}, p-value: {p_value:.8f}")
print(f"Normality assumption {'satisfied' if p_value > 0.05 else 'violated'}")

# Heteroscedasticity test
_, bp_p_value, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X_sm)
print(f"\nBreusch-Pagan test for heteroscedasticity:")
print(f"p-value: {bp_p_value:.8f}")
print(f"Homoscedasticity assumption {'satisfied' if bp_p_value > 0.05 else 'violated'}")

# Autocorrelation test
dw_stat = sm.stats.stattools.durbin_watson(residuals)
print(f"\nDurbin-Watson test for autocorrelation:")
print(f"Statistic: {dw_stat:.4f}")
print(f"Independence assumption {'satisfied' if 1.5 < dw_stat < 2.5 else 'potentially violated'}")

print("\nModel check complete. Results saved to the 'output/model_check' directory.") 