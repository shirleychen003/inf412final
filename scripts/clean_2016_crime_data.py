import pandas as pd

# Load raw data
crime_data = pd.read_csv("../raw_data/neighbourhood-crime-rates - 4326.csv")

# Filter out the columns to keep for 2016
columns_to_keep = ["AREA_NAME", "HOOD_ID", "ASSAULT_2016", "ASSAULT_RATE_2016", "AUTOTHEFT_2016",
                   "AUTOTHEFT_RATE_2016", "BIKETHEFT_2016", "BIKETHEFT_RATE_2016",
                   "BREAKENTER_2016", "BREAKENTER_RATE_2016", "HOMICIDE_2016",
                   "HOMICIDE_RATE_2016", "ROBBERY_2016",
                   "ROBBERY_RATE_2016", "SHOOTING_2016", "SHOOTING_RATE_2016",
                   "THEFTFROMMV_2016", "THEFTFROMMV_RATE_2016", "THEFTOVER_2016",
                   "THEFTOVER_RATE_2016"]

cleaned_data = crime_data[columns_to_keep]

# Replace empty columns with 0
cleaned_data = cleaned_data.fillna(0)

# Make numbers ints except for rates. Round the rates to 2 decimal places.
for column in cleaned_data.columns[1:]:
    if 'RATE' not in column:
        cleaned_data[column] = cleaned_data[column].astype(int)
    else:
        cleaned_data[column] = cleaned_data[column].round(2)

# Add up all the crime counts
columns_to_sum = ["ASSAULT_2016", "AUTOTHEFT_2016", "BIKETHEFT_2016", "BREAKENTER_2016",
                  "HOMICIDE_2016", "ROBBERY_2016", "SHOOTING_2016", "THEFTFROMMV_2016",
                  "THEFTOVER_2016"]

# Create a new column 'TOTAL_CRIME_COUNT' that sums the values across the specified columns
cleaned_data['TOTAL_CRIME_COUNT'] = cleaned_data[columns_to_sum].sum(axis=1)

# Export cleaned data
cleaned_data.to_csv("../cleaned_data/2016/cleaned_crime_data_2016.csv", index=False)

print(f"Crime data for 2016 has been cleaned for {len(cleaned_data)} neighborhoods.")
print(f"Sample of cleaned data:")
print(cleaned_data.head()) 