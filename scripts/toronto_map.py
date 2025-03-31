import pandas as pd
import geopandas as gpd
import folium
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import matplotlib.patches as mpatches
import re

# Step 1: Load Toronto Neighborhood Boundaries
# Option 1: Direct download from GitHub
toronto_geojson_url = "https://raw.githubusercontent.com/jasonicarter/toronto-geojson/master/toronto_crs84.geojson"
neighborhoods = gpd.read_file(toronto_geojson_url)

# Load reconciled data
unified_data = pd.read_csv('cleaned_data/reconciled_data/unified_data.csv')

# Extract relevant columns for the map tooltip
hover_data = unified_data.loc[:, [
    'Reconciled_Neighbourhood', 
    'is_improvement_area_2021',
    'total_population_2021', 
    'low_income_percent_2021',
    'TOTAL_CRIME_COUNT_2021',
    'ASSAULT_2021',
    'AUTOTHEFT_2021',
    'ROBBERY_2021'
]].copy()

# Clean the neighborhood names in GeoJSON by removing the ID in parentheses
neighborhoods['CLEAN_NAME'] = neighborhoods['AREA_NAME'].apply(lambda x: re.sub(r'\s*\(\d+\)$', '', x).strip())

# Ensure both dataframes have consistently formatted neighborhood names
neighborhoods['CLEAN_NAME'] = neighborhoods['CLEAN_NAME'].str.strip()
hover_data['Reconciled_Neighbourhood'] = hover_data['Reconciled_Neighbourhood'].str.strip()

# Create a mapping dictionary for mismatched names
name_mapping = {
    # Add specific mappings for neighborhoods that might still have different names
    'Cabbagetown-South St.James Town': 'Cabbagetown-South St. James Town',
    'North St.James Town': 'North St. James Town',
    "O'Connor-Parkview": "O'Connor-Parkview",
    "Weston-Pellam Park": "Weston-Pelham Park",
    "L'Amoreaux": "L'Amoreaux",
    "Tam O'Shanter-Sullivan": "Tam O'Shanter-Sullivan"
}

# Apply name mapping for special cases
neighborhoods['MAPPED_NAME'] = neighborhoods['CLEAN_NAME'].map(lambda x: name_mapping.get(x, x))

# Merge GeoJSON with the hover data
merged_data = neighborhoods.merge(hover_data, 
                               left_on='MAPPED_NAME', 
                               right_on='Reconciled_Neighbourhood',
                               how='left')

# Fill NaN values with 0
merged_data['is_improvement_area_2021'] = merged_data['is_improvement_area_2021'].fillna(0)
numeric_cols = ['total_population_2021', 'low_income_percent_2021', 'TOTAL_CRIME_COUNT_2021', 
                'ASSAULT_2021', 'AUTOTHEFT_2021', 'ROBBERY_2021']
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(0)

# Calculate crime rates per 100,000 population
for crime_type in ['TOTAL_CRIME_COUNT_2021', 'ASSAULT_2021', 'AUTOTHEFT_2021', 'ROBBERY_2021']:
    rate_col = f"{crime_type.replace('_2021', '')}_RATE"
    merged_data[rate_col] = (merged_data[crime_type] / merged_data['total_population_2021'] * 100000).round(1)
    # Handle division by zero
    merged_data[rate_col] = merged_data[rate_col].replace([float('inf'), float('nan')], 0)

# Step 3: Create a Static Map using GeoPandas and Matplotlib
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Create a colormap: improvement areas in red, others in blue
colors = ListedColormap(['lightblue', 'red'])

# Plot the map
merged_data.plot(column='is_improvement_area_2021', 
                 ax=ax,
                 cmap=colors, 
                 legend=True)

# Manually add a custom legend
blue_patch = mpatches.Patch(color='lightblue', label='Regular Neighborhood')
red_patch = mpatches.Patch(color='red', label='Improvement Area')
ax.legend(handles=[blue_patch, red_patch], title='Neighborhood Status')

# Add title and remove axis
ax.set_title('Toronto Neighborhoods with Improvement Areas Highlighted', fontsize=20)
ax.set_axis_off()

# Add neighborhood names for context (optional)
for idx, row in merged_data.iterrows():
    # Only label improvement areas to avoid clutter
    if row['is_improvement_area_2021'] == 1:
        ax.annotate(text=row['CLEAN_NAME'], 
                   xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                   ha='center', fontsize=8)

# Save the static map
os.makedirs('output', exist_ok=True)  # Ensure output directory exists
plt.savefig('output/toronto_neighborhoods_improvement_areas.png', dpi=300, bbox_inches='tight')
plt.close()

# Step 4: Create an Interactive Map using Folium
m = folium.Map(location=[43.7417, -79.3733], zoom_start=10)

# Define a function to style neighborhoods based on improvement area status
def style_function(feature):
    is_improvement_area = feature['properties']['is_improvement_area_2021']
    
    # Style for improvement areas
    if is_improvement_area == 1:
        return {
            'fillColor': '#FF4136',
            'color': '#85144b',
            'weight': 1,
            'fillOpacity': 0.7
        }
    # Style for regular neighborhoods
    else:
        return {
            'fillColor': '#7FDBFF',
            'color': '#0074D9',
            'weight': 1,
            'fillOpacity': 0.4
        }

# Add the is_improvement_area to the properties for the GeoJSON
merged_data = merged_data.copy()  # Make a copy to avoid SettingWithCopyWarning
for col in merged_data.columns:
    if col not in ['geometry', 'AREA_NAME', 'CLEAN_NAME', 'MAPPED_NAME', 'Reconciled_Neighbourhood']:
        merged_data.loc[:, col] = merged_data[col].astype(float)  # Convert to float for proper JSON encoding

# Create enhanced tooltip with more data
tooltip_fields = [
    'CLEAN_NAME', 
    'is_improvement_area_2021',
    'total_population_2021', 
    'low_income_percent_2021',
    'TOTAL_CRIME_COUNT_RATE', 
    'ASSAULT_RATE',
    'AUTOTHEFT_RATE',
    'ROBBERY_RATE'
]

tooltip_aliases = [
    'Neighborhood', 
    'Improvement Area',
    'Population (2021)', 
    'Low Income %',
    'Total Crime Rate', 
    'Assault Rate',
    'Auto Theft Rate',
    'Robbery Rate'
]

# Add neighborhood polygons to the map with enhanced tooltip
folium.GeoJson(
    merged_data,
    name='Improvement_Areas',  # Add a proper name for the layer
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        localize=True,
        sticky=True,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px 3px 3px #888888;
            padding: 10px;
            font-size: 14px;
        """
    )
).add_to(m)

# Add a legend
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 200px; height: 120px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     border-radius: 5px;">
     <p><strong>Neighborhood Status</strong></p>
     <p><i style="background: #7FDBFF; width: 15px; height: 15px; display: inline-block;"></i> Regular Neighborhood</p>
     <p><i style="background: #FF4136; width: 15px; height: 15px; display: inline-block;"></i> Improvement Area</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Add a title
title_html = '''
<h3 align="center" style="font-size:16px">Toronto Neighborhoods with Improvement Areas (2021)</h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Add a source note
source_html = '''
<div style="position: fixed; bottom: 10px; left: 10px; background-color: white; 
            padding: 5px; border-radius: 5px; font-size: 12px;">
  Data sources: City of Toronto Open Data, Toronto Neighborhood Profiles 2021
</div>
'''
m.get_root().html.add_child(folium.Element(source_html))

# Add instructions for users
instruction_html = '''
<div style="position: fixed; top: 10px; right: 10px; background-color: white; 
            padding: 10px; border-radius: 5px; font-size: 12px; border: 1px solid #ccc; max-width: 250px;">
  <strong>How to use:</strong> Hover over any neighborhood to see demographic and crime data. 
  Crime rates are per 100,000 population.
</div>
'''
m.get_root().html.add_child(folium.Element(instruction_html))

# Add low income choropleth layer
folium.Choropleth(
    geo_data=merged_data,
    name='Low Income',
    data=merged_data,
    columns=['CLEAN_NAME', 'low_income_percent_2021'],
    key_on='feature.properties.CLEAN_NAME',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Low Income Percentage (2021)'
).add_to(m)

# Add crime rate choropleth layer
folium.Choropleth(
    geo_data=merged_data,
    name='Crime Rate',
    data=merged_data,
    columns=['CLEAN_NAME', 'TOTAL_CRIME_COUNT_RATE'],
    key_on='feature.properties.CLEAN_NAME',
    fill_color='PuRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Crime Rate per 100,000 (2021)'
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save the interactive map
os.makedirs('output', exist_ok=True)  # Ensure output directory exists
m.save('output/toronto_neighborhoods_improvement_areas_interactive.html')

print("Map creation complete!")
