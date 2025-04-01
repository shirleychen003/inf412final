# Final Summary: Linear vs. Count Models for Crime Data Analysis

This document summarizes our findings from comparing linear regression models with count models (Poisson and Negative Binomial) for analyzing Toronto neighborhood crime data.

## Why Linear Models Are Inadequate for Crime Data

Our analysis clearly demonstrates that linear regression models are inappropriate for modeling crime count data for several key reasons:

1. **Violated Assumptions**: All linear models showed violations of key assumptions:
   - Non-normality of residuals (Shapiro-Wilk p < 0.05 for all models)
   - Heteroscedasticity (increasing error variance with predicted values)
   - Negative predictions for some neighborhoods (crime counts cannot be negative)

2. **Poor Fit to Data Distribution**: Crime data follows a right-skewed distribution that is more naturally modeled by count distributions than by normal distributions assumed in linear regression.

3. **Underprediction of High-Crime Areas**: Linear models consistently underpredicted crime in high-crime neighborhoods, particularly for ASSAULT.

## Benefits of Count Models

The Negative Binomial models provided significant improvements:

1. **Appropriate for Count Data**: Count models are specifically designed for non-negative integer outcomes like crime counts.

2. **Handling Overdispersion**: All crime types showed substantial overdispersion (variance > mean), with dispersion ratios ranging from 14 to 82. Negative Binomial models appropriately handled this overdispersion, which Poisson models could not.

3. **Better Model Fit**: Negative Binomial models showed:
   - Lower AIC values compared to both Poisson and linear models
   - Better predictions, especially in high-crime neighborhoods
   - More appropriate residual patterns

4. **Log-Link Interpretation**: The log-link function in count models ensures predictions are always positive and allows for intuitive interpretation of coefficients as percentage changes.

## Key Findings About Crime Predictors

1. **Population as a Predictor**:
   - Strong association with all crime types (Pseudo R² values 0.24-0.42)
   - The effect is approximately linear in log-space, meaning a proportional increase in population leads to a proportional increase in crime

2. **Low Income Percentage as a Predictor**:
   - Strong relationship with ASSAULT (R² ~0.34) and ROBBERY (R² ~0.23)
   - Weaker relationship with property crimes like BREAKENTER and AUTOTHEFT (R² <0.12)
   - From the Negative Binomial model for ASSAULT in 2021, each additional percentage point of low income is associated with a 10.1% increase in assault incidents (exp(0.0961) = 1.101)

## Recommendations for Future Crime Modeling

1. **Use Negative Binomial Models**: For all crime types, Negative Binomial regression is clearly the appropriate modeling approach due to the significant overdispersion in the data.

2. **Consider Multivariate Models**: Combining predictors (population, low income percentage, and potentially other socioeconomic factors) may provide even better predictive power.

3. **Incorporate Spatial Effects**: Crime often exhibits spatial autocorrelation, where neighboring areas have similar crime rates. Spatial regression models could further improve predictions.

4. **Model Selection**: For any future crime modeling, we recommend:
   - Testing for overdispersion to determine whether Poisson or Negative Binomial is appropriate
   - Using appropriate diagnostics to assess model fit
   - Considering the substantive interpretation of coefficients, not just statistical fit

## Conclusion

This analysis clearly demonstrates that count models, particularly Negative Binomial regression, are vastly superior to linear regression for modeling crime data. The consistent patterns of overdispersion, the violation of linear model assumptions, and the improved fit of count models all support this conclusion. These findings align with best practices in criminology and provide a more robust foundation for understanding the relationships between neighborhood characteristics and crime rates in Toronto. 