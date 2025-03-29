import pandas as pd

# Load raw data
crime_data = pd.read_csv("../raw_data/neighbourhood-crime-rates - 4326.csv")

# Filter out the columns to keep
columns_to_keep = ["AREA_NAME", "HOOD_ID", "ASSAULT_2020", "AUTOTHEFT_2020",
                   "AUTOTHEFT_RATE_2020", "BIKETHEFT_2020", "BIKETHEFT_RATE_2020",
                   "BREAKENTER_2020", "BREAKENTER_RATE_2020", "HOMICIDE_2020",
                   "ASSAULT_2020", "HOMICIDE_RATE_2020", "ROBBERY_2020",
                   "ROBBERY_RATE_2020", "SHOOTING_2020", "SHOOTING_RATE_2020",
                   "THEFTFROMMV_2020", "THEFTFROMMV_RATE_2020", "THEFTOVER_2020",
                   "THEFTOVER_RATE_2020"]

cleaned_data = crime_data[columns_to_keep]

# Replace empty columns with 0
cleaned_data = cleaned_data.fillna(0)

# Make numbers ints except for rates. Round the rates to 2 decimal places.
for column in cleaned_data.columns[1:]:
    if 'RATE' not in column:
        cleaned_data[column] = cleaned_data[column].astype(int)
    else:
        cleaned_data[column] = cleaned_data[column].round(2)

# Replace empty columns with 0
cleaned_data = cleaned_data.fillna(0)

# Export cleaned data
cleaned_data.to_csv("../cleaned_data/cleaned_crime_data.csv", index=False)
