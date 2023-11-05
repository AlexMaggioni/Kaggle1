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

def format_data(data, is_test=False):

    # ========================================================
    # Preprocessing steps and removing unnecessary features
    # ========================================================

    # Remove SNo column
    if "SNo" in data.columns:
        data = data.drop(columns=["SNo"])

    # Remove all rows with a duplicate set of features in training data
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
        data = data.drop(columns=["LAT", "LON"])
    
    if "lat" in data.columns and "lon" in data.columns:
        data = data.drop(columns=["lat", "lon"])

    if "LOCATION" in data.columns:
        data["SOUTHERN_HEMISPHERE"] = (data["LOCATION"] <= 2).astype(int)

    if "LOCATION" in data.columns:
        # Apply one-hot encoding to the LOCATION column
        data = pd.get_dummies(data, columns=["LOCATION"], prefix="LOC")
        encoded_columns = [col for col in data.columns if "LOC_" in col]
        data[encoded_columns] = data[encoded_columns].astype(int)

    if "time" in data.columns:
        # Simplify the time column
        data['MONTH'] = data['time'].apply(lambda x: int(str(x)[4:6]))
        data = data.drop(columns=["time"])

    if is_test:
        # Change scale of PRECT and transform to log scale
        data["PRECT"] = data["PRECT"].apply(lambda x: 0 if x < 0 else x)
        data["PRECT"] = np.sqrt(data["PRECT"] * 10_000_000_000)

    # ========================================================
    # General features that could be useful
    # ========================================================

    # Wind magnitude for 850 and BOT
    data["WIND850_MAGNITUDE"] = np.sqrt(data["U850"]**2 + data["V850"]**2)
    data["WINDBOT_MAGNITUDE"] = np.sqrt(data["UBOT"]**2 + data["VBOT"]**2)

    # Wind direction for 850 and BOT
    data['WIND850_DIRECTION'] = np.arctan2(data['V850'], data['U850'])
    data['WINDBOT_DIRECTION'] = np.arctan2(data['VBOT'], data['UBOT'])

    # ========================================================
    # Interaction features
    # ========================================================

    # Interaction between wind magnitudes
    data["WIND_MAGNITUDE_INTERACTION"] = data['WIND850_MAGNITUDE'] * data['WINDBOT_MAGNITUDE'] 

    # Interaction between temperature and humidity
    data['PS_TMQ_INTERACTION'] = data['PS'] * data['TMQ']

    # Interaction between precipitation rate and wind magnitude
    data["PRECT_WIND_MAGNITUDE_INTERACTION"] = data["PRECT"] * data["WIND850_MAGNITUDE"] * data["WINDBOT_DIRECTION"]

    # Interaction between wind magnitude and pressure
    data['PS_WIND850_INTERACTION'] = data['PS'] * data['WIND850_MAGNITUDE']
    data['PS_WINDBOT_INTERACTION'] = data['PS'] * data['WINDBOT_MAGNITUDE']

    # Humidity and wind magnitude interaction
    data['QREFHT_WIND850_INTERACTION'] = data['QREFHT'] * data['WIND850_MAGNITUDE']
    data['QREFHT_WINDBOT_INTERACTION'] = data['QREFHT'] * data['WINDBOT_MAGNITUDE']

    # Humidity and temperature interaction
    data['HUMIDITY_INDEX'] = data['TMQ'] * data['QREFHT']

    # Integrated Vapor Transport
    data["TMQ_WIND850_INTERACTION"] = data['TMQ'] * data["WIND850_MAGNITUDE"]
    data['TMQ_WINDBOT_INTERACTION'] = data['TMQ'] * data["WINDBOT_MAGNITUDE"]
    
    # Interaction between temperature and humidity
    data['T200_QREFHT_INTERACTION'] = data['T200'] * data['QREFHT']
    data['T500_QREFHT_INTERACTION'] = data['T500'] * data['QREFHT']

    # Interaction between temperature and wind magnitude
    data['QREFHT_TREFHT_INTERACTION'] = data['QREFHT'] * data['TREFHT']

    # ========================================================
    # Features intended to identify Tropical Cyclones
    # ========================================================
    
    # Geopotential Height Difference to provide insights into the vertical structure of the atmosphere
    data['GEOPOTENTIAL_DIFF_200_1000'] = data['Z200'] - data['Z1000']

    # Pressure difference: Difference between surface and sea-level pressure
    data['PRESSURE_DIFFERENCE'] = data['PS'] - data['PSL']

    # Temperature difference: Difference between temperatures at different pressure surfaces
    data['TEMP_DIFFERENCE'] = data['T200'] - data['T500']

    # Wind shear: Difference in wind magnitudes between levels
    data['WIND_SHEAR'] = data['WINDBOT_MAGNITUDE'] - data['WIND850_MAGNITUDE']

    # ========================================================
    # Features intended to identify Atmospheric Rivers
    # ========================================================

    # Ratio of PRECT to TMQ
    data['PRECT_TMQ_RATIO'] = data['PRECT'] / data['TMQ']

    # Sorting columns
    ordered_columns = sorted(data.columns)
    if not is_test and "Label" in data.columns:
        ordered_columns.remove("Label")
        ordered_columns.append("Label")

    data = data[ordered_columns]

    return data