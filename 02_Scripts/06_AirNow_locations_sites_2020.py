import pandas as pd


# Helper function for AQI
def get_aqi_level(row):
    """
    Computes the level of health concern based on a numeric value of the
    AQI index for a pollutant
    :param row: the AQI level of the pollutant
    :return: a string in ['good', 'moderate', 'unhealthy_sensitive_groups',
           'unhealthy', 'very_unhealthy', 'hazardous'] indicating the level
           of health concern
    """
    if 'AQI' in row['type']:
        if row['value'] <= 50:
            return 'good'
        elif row['value'] <= 100:
            return 'moderate'
        elif row['value'] <= 150:
            return 'unhealthy_sensitive_groups'
        elif row['value'] <= 200:
            return 'unhealthy'
        elif row['value'] <= 300:
            return 'very_unhealthy'
        else:
            return 'hazardous'
    else:
        return None

# Load city and site datasets
city_data = pd.read_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_cities_2020_avg_latlon.csv',
    usecols=['id', 'name', 'type', 'measurement', 'value', 'lat', 'lon'])
site_data = pd.read_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_sites_2020_avg.csv')

# Modify city dataset
city_data['location_type'] = 'city_location'
city_data = city_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value', 'lat', 'lon']]

# Modify site dataset
site_data['location_type'] = 'site'
site_data.rename(columns={'AQSID': 'id', 'SiteName': 'name',
                          'Latitude': 'lat', 'Longitude': 'lon',
                          'measurement': 'type'},
                 inplace=True)

# Add measurements to site dataset
measurement_map = pd.Series({'PM10_AQI': 'AQI', 'PM25_AQI': 'AQI',
                             'OZONE_AQI': 'AQI',
                             'NO2_AQI': 'AQI', 'PM25': 'UG/M3', 'OZONE': 'PPB',
                             'PM10': 'UG/M3', 'CO': 'PPM', 'SO2': 'PPB',
                             'NO2': 'PPB'}, name='measurement')
site_data = site_data.merge(measurement_map, how='left', left_on='type',
                            right_index=True)
site_data = site_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value', 'lat', 'lon']]

# Create a map of ID, lat and lon so as to add missing location information
# in city_data from site_data
location_map = site_data[['id', 'lat', 'lon']].copy()
city_data = city_data[['location_type', 'id', 'name', 'type',
                       'measurement', 'value']]\
    .merge(location_map, how='left', on='id')

# Concatenate city and site datasets
combined_data = pd.concat([city_data, site_data], axis=0)

# Drop missing 'value', 'lat', 'lon' rows
# Note: this drops two rows with missing location information
# and 52,887 rows with missing value information
combined_data.dropna(subset=['value', 'lat', 'lon'], axis=0, inplace=True)

# Check for duplicates
combined_data.drop_duplicates(subset=['id', 'name', 'type', 'measurement'],
                              inplace=True)

# Compute qualitative AQI level for AQI variables
combined_data['AQI_level'] = combined_data.apply(get_aqi_level, axis=1)

# Drop location_type col
combined_data.drop('location_type', axis=1, inplace=True)

# Save combined data
combined_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_locations_2020_avg.csv',
    index=False)
