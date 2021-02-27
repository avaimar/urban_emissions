import numpy as np
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


# Helper function for lat/lon distances
def acceptable_distance(df):
    """
    Helper function to determine whether the entries in the [id, name]
    pair are acceptably close together.
    :param df: (DataFrame)
    :return:
    """
    num_entries = df.shape[0]
    if df['lat_ok'].sum() + df['lon_ok'].sum() == 2 * num_entries:
        return True
    else:
        return False


# Load city and site datasets
city_data = pd.read_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_cities_2020_avg_raw.csv')
site_data = pd.read_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_sites_2020_avg_raw.csv')

# Import reporting area information from AirNow
reporting_area = pd.read_csv(
    'https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/Site_To_ReportingArea.csv',
    encoding='ISO-8859-1')

# Modify city dataset
city_data['location_type'] = 'city_location'

# Modify site dataset
# We start by dropping rows with missing values for column 'value'
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
# Note: there are various sites which have exactly the same id, lat and lon
# but different names. These sites appear to be the same, as their names
# tend to have different spellings. We will keep the first entry in these cases.
location_map = site_data[['id', 'lat', 'lon']].copy()
location_map.drop_duplicates(['id', 'lat', 'lon'], inplace=True)

# Ensure there that Site IDs have the same lat/lon
location_count = location_map.groupby(['id'], as_index=False)['lat']\
    .count().sort_values('lat', ascending=False)
location_count.rename(columns={'lat': 'count'}, inplace=True)

location_map = location_map.merge(
    location_count, on=['id'], how='left', validate='many_to_one')

location_unique = location_map[location_map['count'] == 1].copy()
location_duplicates = location_map[location_map['count'] > 1].copy()

# There are several sites with the same ID that have coordinates differing by a few
# decimals. Check if distance for lat/lon is less than 0.1 and if so we
# keep the first entry for these pairs
location_duplicates['first_lat'] = location_duplicates.groupby(['id'])['lat'].transform('first')
location_duplicates['first_lon'] = location_duplicates.groupby(['id'])['lon'].transform('first')

location_duplicates['lat_ok'] = np.abs(location_duplicates['lat'] - location_duplicates['first_lat']) < 0.1
location_duplicates['lon_ok'] = np.abs(location_duplicates['lon'] - location_duplicates['first_lon']) < 0.1

# Note that although the location_duplicates dataset is 1203 entries long, it
# refers to only 65 unique sites (len(location_duplicates['id'].unique())). We
# will thus keep the sites that abide by the 0.1 max distance in both lat and lon
# and 'throw' away the rest. If we are in severe need for more data points we can
# come back and clean the remaining 20 sites manually.

appropriate_locs = location_duplicates.groupby('id')\
    .apply(acceptable_distance).reset_index()

location_duplicates = location_duplicates.merge(
    appropriate_locs, how='left', on=['id'], validate='many_to_one')
location_duplicates = location_duplicates[location_duplicates[0] == True]
location_duplicates = location_duplicates.groupby(['id'], as_index=False).agg('first')

# Concatenate these new unique [id, name] pairs with the already unique pairs
location_map = pd.concat(
    [location_unique[['id', 'lat', 'lon']],
     location_duplicates[['id', 'lat', 'lon']]])

# At this point we have unique [id] with unique lat/lon values. We
# merge these with the city data by [id].
city_data = city_data[['location_type', 'id', 'name', 'type',
                       'measurement', 'value']] \
    .merge(location_map, how='left',
           on=['id'], validate='many_to_one')

# Note that there are 8578 entries with lat/lon info and 8485 entries whose
# ID is not present in the location map.

# We will attempt to find the remaining missing locations with the
# ReportingArea dictionary provided by AirNow. This dictionary does not
# include all sites, however, and only has data on 1569 sites.

missing_city_data = city_data[city_data['lat'].isna()]
nonmissing_city_data = city_data[city_data['lat'].notna()]

# Note that reporting_area includes duplicate rows where the same site
# (with the same lat/lon) maps to multiple ReportingAreas. We will thus
# drop duplicates, as we are interested only in these site-lat-lon mappings.

reporting_area.drop_duplicates(subset='SiteID', inplace=True)

missing_city_data = missing_city_data[
    ['location_type', 'id', 'name', 'type', 'measurement', 'value']]\
    .merge(reporting_area[['SiteID', 'SiteLat', 'SiteLong']],
           how='left', left_on='id', right_on='SiteID', validate='many_to_one')
missing_city_data.rename(columns={'SiteLat': 'lat', 'SiteLong':'lon'}, inplace=True)
# This added data on 63 entries.

city_data = pd.concat([
    nonmissing_city_data[['location_type', 'id', 'name', 'type', 'measurement',
                          'value', 'lat', 'lon']],
    missing_city_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value', 'lat', 'lon']]])

city_data = city_data[city_data['lat'].notna()]

# We will also merge these back to the site dataset in order to ensure that
# every ID is mapped to a unique lat lon
site_data = site_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value']].merge(
    location_map, on='id', how='left', validate='many_to_one')

# Concatenate city and site datasets
city_data = city_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value', 'lat', 'lon']]
site_data = site_data[['location_type', 'id', 'name', 'type', 'measurement',
                       'value', 'lat', 'lon']]
combined_data = pd.concat([city_data, site_data], axis=0)

# Note: 962 missing lat/lon values now come from site_data. These are related
# to Sites that were dropped in the location_map because their multiple lat/lons
# were different by more than 0.1. We will drop these rows now.
combined_data = combined_data.dropna(axis=0, subset=['lat', 'lon'])

# Fix differing values for PM2.5
combined_data.loc[combined_data['type'] == 'PM25', 'type'] = 'PM2.5'

# Check for duplicates
# We first check that each ID points to a single lat/lon location
test = combined_data.groupby(['id'], as_index=False)['lat'].nunique().sort_values(ascending=False)
test = combined_data.groupby(['id'], as_index=False)['lon'].nunique().sort_values(ascending=False)

# Continuing to work with the assumption that a Site ID is a unique location
# regardless of differences in the name, we will average over the values
# for each [id, type, lat, lon] combination
combined_data = combined_data.groupby(
    ['id', 'type', 'measurement', 'lat', 'lon'], as_index=False)['value'].mean()

# Compute qualitative AQI level for AQI variables
combined_data['AQI_level'] = combined_data.apply(get_aqi_level, axis=1)

# Save combined data
combined_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_locations_2020_avg_clean.csv',
    index=False)
