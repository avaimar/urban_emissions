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
    '../../01_Data/01_Carbon_emissions/AirNow/World_sites_2020_avg_raw.csv')

# Modify city dataset
city_data['location_type'] = 'city_location'
city_data = city_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value', 'lat', 'lon']]

# Modify site dataset
# We start by dropping rows with NULL values
site_data.dropna(subset=['value'], axis=0, inplace=True)
site_data['location_type'] = 'site'
site_data.rename(columns={'AQSID': 'id', 'SiteName': 'name',
                          'Latitude': 'lat', 'Longitude': 'lon',
                          'measurement': 'type'}, inplace=True)

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
# NOTE: The majority of missings are related to 'JPN Site XXXX'
location_map = site_data[['id', 'lat', 'lon']].copy()
location_map.drop_duplicates(['id'], inplace=True)

city_data_non_missing = city_data.dropna(subset=['lat'], axis=0)
city_data_missing = city_data[city_data['lat'].isna()]

city_data_missing = city_data_missing[['location_type', 'id', 'name', 'type',
                       'measurement', 'value']]\
    .merge(location_map, how='left', on='id', validate='many_to_one')
city_data = pd.concat([city_data_non_missing, city_data_missing])

# Concatenate city and site datasets
combined_data = pd.concat([city_data, site_data], axis=0)

# Drop missing 'lat', 'lon' rows
combined_data.dropna(subset=['lat', 'lon'], axis=0, inplace=True)

# Check for duplicates
# Note: We add keep='last' so as to use information from the site dataset
# as the city dataset tends to have distinct values for some of the sites. Site
# data tends to coincide with one of these city data values
combined_data.drop_duplicates(subset=['id', 'type', 'lat', 'lon'],
                              inplace=True, keep='last')

# Note: Some sites such as MMGBU1000 share the same ID but have different
# names and coordinates as shown with the code below. Other sites appear to be
# the same; they have slightly different coordinates their names seem to be
# spelled differently.
# combined_data.groupby(['id', 'type']).count().sort_values('name', ascending=False)
# We will use the average of these sites to map {'id', 'type'} to a single value.
# This seems to be a sensible choice as it is likely that these sites' names
# changed over time, or that they are near geographically.
combined_data = combined_data.groupby(
    ['id', 'type', 'measurement'], as_index=False).agg(
    dict(name='first', value='mean', lat='mean', lon='mean'))

# Compute qualitative AQI level for AQI variables
combined_data['AQI_level'] = combined_data.apply(get_aqi_level, axis=1)

# Save combined data
combined_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_locations_2020_avg.csv',
    index=False)
