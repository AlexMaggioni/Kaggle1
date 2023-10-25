import numpy as np
import pandas as pd

location_per_latlon = {
0: [
(-31.09517601, 353.75),
(-31.09517601, 353.4375),
(-31.09517601, 353.125),
(-30.62581486, 353.125),
(-30.62581486, 353.4375),
(-30.62581486, 353.75),
(-30.86049544, 354.0625),
(-30.86049544, 354.375),
(-30.39113429, 354.375),
(-30.39113429, 354.0625),
(-30.86049544, 353.75),
(-30.86049544, 353.4375),
(-30.86049544, 353.125),
(-30.39113429, 353.125),
(-30.39113429, 353.4375),
(-30.39113429, 353.75),
(-31.09517601, 354.375),
(-31.09517601, 354.0625),
(-30.62581486, 354.375),
(-30.62581486, 354.0625)
],

1: [
(-21.00391134, 229.6875),
(-21.00391134, 229.375),
(-21.00391134, 229.0625),
(-20.5345502, 229.6875),
(-20.5345502, 229.375),
(-20.5345502, 229.0625),
(-21.23859192, 230.3125),
(-21.23859192, 230.0),
(-20.76923077, 230.0),
(-20.76923077, 230.3125),
(-21.23859192, 229.375),
(-21.23859192, 229.6875),
(-21.23859192, 229.0625),
(-20.76923077, 229.375),
(-20.76923077, 229.6875),
(-20.76923077, 229.0625),
(-21.00391134, 230.0),
(-21.00391134, 230.3125),
(-20.5345502, 230.3125),
(-20.5345502, 230.0)
],

2: [
(-24.75880052, 242.1875),
(-24.75880052, 242.5),
(-24.28943937, 242.1875),
(-24.28943937, 242.5),
(-24.52411995, 242.1875),
(-24.52411995, 242.5),
(-24.75880052, 241.25),
(-24.75880052, 241.875),
(-24.75880052, 241.5625),
(-24.28943937, 241.25),
(-24.28943937, 241.5625),
(-24.28943937, 241.875),
(-24.9934811, 242.1875),
(-24.9934811, 242.5),
(-24.52411995, 241.25),
(-24.52411995, 241.5625),
(-24.52411995, 241.875),
(-24.9934811, 241.5625),
(-24.9934811, 241.25),
(-24.9934811, 241.875)
],

3: [
(23.58539765, 277.1875),
(23.58539765, 277.5),
(23.58539765, 277.8125),
(21.47327249, 275.0),
(21.47327249, 275.3125),
(21.47327249, 275.625),
(21.47327249, 275.9375),
(21.70795306, 276.25),
(24.0547588, 277.1875),
(24.0547588, 277.5),
(24.0547588, 277.8125),
(23.58539765, 276.5625),
(23.58539765, 276.875),
(23.35071708, 277.1875),
(23.35071708, 277.5),
(23.35071708, 277.8125),
(23.82007823, 277.1875),
(23.82007823, 277.5),
(23.82007823, 277.8125),
(21.70795306, 275.625),
(21.70795306, 275.0),
(21.70795306, 275.9375),
(21.70795306, 275.3125),
(24.0547588, 276.5625),
(24.0547588, 276.875),
(23.82007823, 276.5625),
(21.47327249, 276.25),
(23.82007823, 276.875),
(23.35071708, 276.5625),
(23.35071708, 276.875)
],

4: [
(12.79009126, 252.5),
(12.79009126, 252.8125),
(13.25945241, 252.8125),
(13.25945241, 252.5),
(13.49413299, 253.75),
(13.49413299, 253.125),
(13.49413299, 253.4375),
(13.02477184, 253.75),
(13.02477184, 253.125),
(13.02477184, 253.4375),
(12.79009126, 253.4375),
(12.79009126, 253.75),
(12.79009126, 253.125),
(13.49413299, 252.5),
(13.49413299, 252.8125),
(13.25945241, 253.125),
(13.25945241, 253.4375),
(13.25945241, 253.75),
(13.02477184, 252.8125),
(13.02477184, 252.5)
],

5: [(21.94263364, 244.6875),
(21.94263364, 244.375),
(21.94263364, 244.0625),
(22.17731421, 245.0),
(21.94263364, 243.75),
(22.17731421, 244.6875),
(22.17731421, 244.0625),
(22.17731421, 244.375),
(21.94263364, 245.0),
(22.17731421, 243.75)
]}

import numpy as np

def format_data(data, is_test=False):
    # Filter all duplicates and remove SNo column
    if "SNo" in data.columns:
        data = data.drop(columns=["SNo"])

    # Remove ALL rows with same set of features but different label
    if not is_test and "Label" in data.columns:
        data = data.drop_duplicates()
        duplicate_mask = data.drop(columns="Label").duplicated(keep=False)
        data = data[~duplicate_mask].reset_index(drop=True)

    if "lat" in data.columns and "lon" in data.columns:
        data = data.rename(columns={"lat": "LAT", 'lon': "LON"})

        # Locations based on clustering shown in the notebook
        data['LOCATION'] = None
        # Iterate through rows of the dataframe
        for idx, row in data.iterrows():
            lat, lon = row['LAT'], row['LON']

            # Check each key in the location dictionary
            for location, coords in location_per_latlon.items():
                if (lat, lon) in coords:
                    data.at[idx, 'LOCATION'] = int(location)
                    break

        # Ensure LOCATION column is of type int
        data['LOCATION'] = data['LOCATION'].astype(int)
        
    if "LOCATION" in data.columns:        
        data["SOUTHERN_HEMISPHERE"] = (data["LOCATION"] <= 2).astype(int)

    if "LOCATION" in data.columns:
        # Apply one-hot encoding to the LOCATION column
        data = pd.get_dummies(data, columns=["LOCATION"], prefix="LOC")
        encoded_columns = [col for col in data.columns if "LOC_" in col]
        data[encoded_columns] = data[encoded_columns].astype(int)

    if "time" in data.columns:
        # Simplify the time column
        data['YEAR'] = data['time'].apply(lambda x: int(str(x)[:4]))
        data['MONTH'] = data['time'].apply(lambda x: int(str(x)[4:6]))
        data['DAY'] = data['time'].apply(lambda x: int(str(x)[6:]))
        data = data.drop(columns=["time"])

    if "PRECT" in data.columns:
        # Scale change for PRECT
        data["PRECT"] = data["PRECT"] * 3600 * 1000

    if all(col in data.columns for col in ["U850", "V850"]):
        # Wind magnitude and direction for 850
        data["WIND850_MAGNITUDE"] = np.sqrt(data["U850"]**2 + data["V850"]**2)
        data['WIND850_DIRECTION'] = np.arctan2(data['V850'], data['U850'])
        data['WIND850_INTERACTION'] = data['U850'] * data['V850']

    if all(col in data.columns for col in ["UBOT", "VBOT"]):
        # Wind magnitude and direction for BOT
        data["WINDBOT_MAGNITUDE"] = np.sqrt(data["UBOT"]**2 + data["VBOT"]**2)
        data['WINDBOT_DIRECTION'] = np.arctan2(data['VBOT'], data['UBOT'])
        data['WINDBOT_INTERACTION'] = data['UBOT'] * data['VBOT']

    if "PS" in data.columns and "WIND850_MAGNITUDE" in data.columns:
        data['PS_WIND850_INTERACTION'] = data['PS'] * data['WIND850_MAGNITUDE']

    if "PS" in data.columns and "WINDBOT_MAGNITUDE" in data.columns:
        data['PS_WINDBOT_INTERACTION'] = data['PS'] * data['WINDBOT_MAGNITUDE']

    if all(col in data.columns for col in ["PS", "PSL"]):
        data['PRESSURE_DIFFERENCE'] = data['PS'] - data['PSL']

    if all(col in data.columns for col in ["T200", "T500"]):
        data['TEMP_DIFFERENCE'] = data['T200'] - data['T500']

    if "WINDBOT_MAGNITUDE" in data.columns and "WIND850_MAGNITUDE" in data.columns:
        data['WIND_SHEAR'] = data['WINDBOT_MAGNITUDE'] - data['WIND850_MAGNITUDE']

    if all(col in data.columns for col in ["Z200", "Z1000"]):
        data['GEOPOTENTIAL_DIFF'] = data['Z200'] - data['Z1000']

    if "MONTH" in data.columns:
        data['SEASON'] = data['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] else ('Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Fall')))
        one_hot = pd.get_dummies(data['SEASON'], prefix="SEASON").astype(int)
        data = pd.concat([data, one_hot], axis=1)
        data = data.drop(columns=["SEASON"])

    if "TMQ" in data.columns and "WIND850_MAGNITUDE" in data.columns:
        data['TMQ_WIND850_INTERACTION'] = data['TMQ'] * data['WIND850_MAGNITUDE']

    if "TMQ" in data.columns and "WINDBOT_MAGNITUDE" in data.columns:
        data['TMQ_WINDBOT_INTERACTION'] = data['TMQ'] * data['WINDBOT_MAGNITUDE']

    if "QREFHT" in data.columns and "TREFHT" in data.columns:
        data['QREFHT_TREFHT_INTERACTION'] = data['QREFHT'] * data['TREFHT']

    # Polynomial Features: Quadratic terms
    if "TMQ" in data.columns:
        data['TMQ_SQUARE'] = data['TMQ'] ** 2

    if "WIND850_MAGNITUDE" in data.columns:
        data['WIND850_MAGNITUDE_SQUARE'] = data['WIND850_MAGNITUDE'] ** 2

    if "WINDBOT_MAGNITUDE" in data.columns:
        data['WINDBOT_MAGNITUDE_SQUARE'] = data['WINDBOT_MAGNITUDE'] ** 2

    if "TREFHT" in data.columns:
        data['TREFHT_SQUARE'] = data['TREFHT'] ** 2

    # Cyclical Encoding for Time-Related Features
    if "MONTH" in data.columns:
        data['MONTH_SIN'] = np.sin(2 * np.pi * data['MONTH'] / 12)
        data['MONTH_COS'] = np.cos(2 * np.pi * data['MONTH'] / 12)

    # Atmospheric Stability: Lapse Rate
    if "T500" in data.columns and "T200" in data.columns:
        data['LAPSE_RATE'] = (data['T500'] - data['T200']) / 300  # difference in temperature over the difference in height

    if not is_test and "Label" in data.columns:
        # Move Label column to the end only if it's not test data
        cols = list(data.columns)
        cols.remove("Label")
        cols.append("Label")
        data = data[cols]

    return data