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
  - Data cleaning scripts (e.g., `clean_2021_neighbourhood_profile_data.py`)
  - Visualization scripts (e.g., `create_visualizations.py`, `box_plot_2016.py`)
  - Statistical analysis (e.g., `correlational_matrix.py`, `linear_regression.py`)
  - Modeling scripts (e.g., `count_models.py`, `model_comparison.py`)
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