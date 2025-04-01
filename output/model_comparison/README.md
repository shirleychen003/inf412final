# Linear vs. Count Models Comparison

This directory contains comparative analysis of Linear Regression vs. Negative Binomial models for crime data analysis.

## Key Files

- **model_comparison_metrics.csv**: Summary of all comparison metrics
- **rmse_improvement_heatmap.png**: Visualization of RMSE improvement percentages
- Individual model files with pattern `{CRIME_TYPE}_{PREDICTOR}_{YEAR}_*`:
  - **_model_comparison.png**: Side-by-side diagnostic plots
  - **_distribution_comparison.png**: Distribution of actual vs predicted values
  - **_errors_by_{predictor}.png**: How prediction errors relate to predictor values
  - **_predictions.csv**: Actual vs. predicted counts for each model
  - **_model_summaries.txt**: Detailed statistical output for both models

## Conclusions

1. Negative Binomial models consistently outperform Linear Regression for crime count data
2. Linear models violate normality assumptions (all Shapiro-Wilk p-values < 0.05)
3. Linear models show heteroscedasticity (increasing error variance with predicted values)
4. Count models provide more accurate predictions, especially for neighborhoods with high crime counts
5. The improvement is particularly notable for highly overdispersed crime types (ASSAULT)

These results confirm that count models are the appropriate choice for modeling crime data.
