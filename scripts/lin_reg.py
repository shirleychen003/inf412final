import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Read and merge the cleaned data
neighborhood_data = pd.read_csv("cleaned_data/cleaned_neighbourhood_data.csv")
crime_data = pd.read_csv("cleaned_data/cleaned_crime_data.csv")

# Clean column names and neighborhood names
def clean_name(name):
    name = name.strip()
    name = name.replace("St. ", "St.")
    name = name.replace("St ", "St.")
    name = name.replace("-East ", "-East")
    name = name.replace("O`Connor", "O'Connor")
    name = name.replace(" - ", "-")
    return name

neighborhood_data.columns = neighborhood_data.columns.str.strip()
crime_data.columns = crime_data.columns.str.strip()
neighborhood_data['Neighbourhood Name'] = neighborhood_data['Neighbourhood Name'].apply(clean_name)
crime_data['AREA_NAME'] = crime_data['AREA_NAME'].apply(clean_name)

# Merge datasets
merged_df = pd.merge(neighborhood_data, crime_data,
                    left_on='Neighbourhood Name',
                    right_on='AREA_NAME',
                    how='inner')

# Define features for analysis
demographic_features = [
    'Total_Population',
    'Youth_Ratio',
    'Working_Age_Ratio',
    'Senior_Ratio',
    'Average_Age',
    'Median_Age',
    'Median_Income_2020',
    'Average_Income_2020'
]

# Create figure for scatter plots
plt.figure(figsize=(20, 15))

# Perform linear regression for each feature against total crime count
for i, feature in enumerate(demographic_features, 1):
    plt.subplot(3, 3, i)
    
    # Prepare data
    X = merged_df[feature].values.reshape(-1, 1)
    y = merged_df['TOTAL_CRIME_COUNT'].values
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate R-squared
    r2 = r2_score(y, y_pred)
    
    # Create scatter plot
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, y_pred, color='red', linewidth=2)
    
    # Add labels and title
    plt.xlabel(feature)
    plt.ylabel('Total Crime Count')
    plt.title(f'{feature} vs Crime Count\nR² = {r2:.3f}')
    
    # Print regression results
    print(f"\nRegression Results for {feature}:")
    print(f"R-squared: {r2:.3f}")
    print(f"Coefficient: {model.coef_[0]:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('cleaned_data/linear_regression/regression_analysis.png', dpi=300, bbox_inches='tight')

# Perform multiple linear regression
print("\nMultiple Linear Regression Analysis:")
X_multi = merged_df[demographic_features]
y_multi = merged_df['TOTAL_CRIME_COUNT']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Create and fit the model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Make predictions
y_pred_multi = model_multi.predict(X_test)

# Calculate metrics
r2_multi = r2_score(y_test, y_pred_multi)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_multi))

print("\nMultiple Linear Regression Results:")
print(f"R-squared: {r2_multi:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")
print("\nFeature Coefficients:")
for feature, coef in zip(demographic_features, model_multi.coef_):
    print(f"{feature}: {coef:.3f}")

# Create a summary DataFrame of the regression results
regression_summary = pd.DataFrame({
    'Feature': demographic_features,
    'Coefficient': model_multi.coef_,
    'Absolute Impact': abs(model_multi.coef_)
})

# Sort by absolute impact
regression_summary = regression_summary.sort_values('Absolute Impact', ascending=False)

# Save regression summary
regression_summary.to_csv('cleaned_data/regression_summary.csv', index=False)

print("\nRegression summary saved to 'cleaned_data/regression_summary.csv'")
