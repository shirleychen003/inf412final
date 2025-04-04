# Toronto Neighborhood Crime Analysis (2016-2021)

## Project Overview
This project analyzes the relationship between socioeconomic factors and crime rates across Toronto neighborhoods, comparing data from 2016 and 2021. The analysis focuses particularly on exploring how low-income percentages correlate with different types of crime, and how these relationships may have changed over the five-year period.

## Research Questions
- How do socioeconomic factors (particularly income levels) correlate with crime rates across Toronto neighborhoods?
- Did these relationships change between 2016 and 2021?
- Are there significant differences between crime count models and crime rate models?
- How do Neighborhood Improvement Areas (NIAs) compare to other neighborhoods in terms of crime statistics?

## Data Sources
- Toronto Neighborhood Profiles (2016 and 2021)
- Toronto Police Service Crime Data (2016 and 2021)

## Project Structure
- `/raw_data/` - Original data files
- `/cleaned_data/` - Processed data files separated by year
  - `/2016/` - Cleaned 2016 data
  - `/2021/` - Cleaned 2021 data
  - `/reconciled_data/` - Matched data across years
- `/scripts/` - Python scripts for data processing and analysis
  - **Data Cleaning Scripts**
    - `clean_2016_neighbourhood_profile_data.py` - Extracts and cleans neighborhood demographic data from 2016
    - `clean_2021_neighbourhood_profile_data.py` - Extracts and cleans neighborhood demographic data from 2021
    - `clean_2016_crime_data.py` - Processes crime statistics from 2016, calculating totals and rates
    - `clean_2021_crime_data.py` - Processes crime statistics from 2021, calculating totals and rates
    - `clean_crime_data.py` - General crime data cleaning script for other years
    - `convert_xlsx.py` - Converts Excel files to CSV format for easier processing
  - **Data Analysis Scripts**
    - `correlational_matrix.py` - Generates correlation matrices between socioeconomic factors and crime metrics
    - `count_model_fitness.py` - Evaluates fitness metrics for count regression models
    - `count_models.py` - Implements Poisson and negative binomial regression models for crime counts
    - `linear_regression.py` - Builds linear regression models for various crime types and predictors
    - `model_check.py` - Performs diagnostic checks on statistical models including posterior predictive checks
    - `model_comparison.py` - Compares different statistical models for each crime type
    - `regression_fitness.py` - Evaluates fitness metrics for general regression models
    - `regression_summary_combined.py` - Creates summary visualizations of regression results across years
    - `reconcile_neighborhoods.py` - Matches and harmonizes neighborhood data between different years
  - **Visualization Scripts**
    - `create_visualizations.py` - Generates a variety of visualizations for exploratory data analysis
    - `box_plot_2016.py` - Creates box plots for 2016 data showing crime distributions by income group
    - `box_plot_2021.py` - Creates box plots for 2021 data with comparisons to 2016
    - `toronto_map.py` - Generates interactive and static maps of Toronto neighborhoods with crime data
- `/output/` - Generated visualizations and analysis results
  - `/box_plot/` - Boxplots comparing crime types and demographics
  - `/correlational_matrix/` - Correlation heatmaps between variables
  - `/linear_regression/` - Regression analysis plots and results
  - `/model_comparison/` - Comparison of statistical models
- `inf412final_report.ipynb` - Jupyter notebook containing the final report

## Key Features
- Correlation analysis between income levels and various crime types
- Linear regression models for crime counts and crime rates
- Geographical visualization of low-income areas and crime hotspots
- Comparative analysis across years (2016 vs 2021)
- Special focus on Neighborhood Improvement Areas (NIAs)

## How to Use This Repository

### Prerequisites
- Python 3.7+
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, geopandas, folium

### Setup
1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Analysis
1. Clean data (if needed):
   ```
   python scripts/clean_2021_neighbourhood_profile_data.py
   python scripts/clean_2021_crime_data.py
   ```
2. Generate visualizations:
   ```
   python scripts/create_visualizations.py
   ```
3. Run regression models:
   ```
   python scripts/linear_regression.py
   python scripts/count_models.py
   ```
4. Open the Jupyter notebook to view the complete analysis:
   ```
   jupyter notebook inf412final_report.ipynb
   ```

## Main Findings
- Strong correlation between low-income percentages and certain crime types, particularly assault
- Population size is a strong predictor for total crime counts
- The relationship between socioeconomic factors and crime rates changed between 2016 and 2021
- Neighborhood Improvement Areas show distinct patterns in both crime rates and income metrics

## Authors
- Shirley Chen
- David James Dimalanta
- Michael Fang
- Harrison Huang