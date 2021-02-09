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
                             'NO2_AQI': 'AQI', 'PM2.5': 'UG/M3', 'OZONE': 'PPB',
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
location_map = site_data[['id', 'name', 'lat', 'lon']].copy()
location_map.drop_duplicates(['id', 'name', 'lat', 'lon'], inplace=True)

# Ensure there that [id, name] pairs have the same lat/lon
location_count = location_map.groupby(['id', 'name'], as_index=False)['lat']\
    .count().sort_values('lat', ascending=False)
location_count.rename(columns={'lat': 'count'}, inplace=True)

location_map = location_map.merge(
    location_count, on=['id', 'name'], how='left', validate='many_to_one')

location_unique = location_map[location_map['count'] == 1].copy()
location_duplicates = location_map[location_map['count'] > 1].copy()

# There are several [id, name] pairs that have coordinates differing by a few
# decimals. Check if distance for lat/lon is less than 0.1 and if so we
# keep the first entry for these pairs
location_duplicates['first_lat'] = location_duplicates.groupby(['id', 'name'])['lat'].transform('first')
location_duplicates['first_lon'] = location_duplicates.groupby(['id', 'name'])['lon'].transform('first')

location_duplicates['lat_ok'] = np.abs(location_duplicates['lat'] - location_duplicates['first_lat']) < 0.1
location_duplicates['lon_ok'] = np.abs(location_duplicates['lon'] - location_duplicates['first_lon']) < 0.1

# Note that although the location_duplicates dataset is 1204 entries long, it
# refers to only 63 unique sites (len(location_duplicates['id'].unique())). We
# will thus keep the sites that abide by the 0.1 max distance in both lat and lon
# and 'throw' away the rest. If we are in severe need for more data points we can
# come back and clean these 18 sites manually.

appropriate_locs = location_duplicates.groupby(
    ['id', 'name'], as_index=False).apply(acceptable_distance).reset_index()

location_duplicates = location_duplicates.merge(
    appropriate_locs, how='left', on=['id', 'name'], validate='many_to_one')
location_duplicates = location_duplicates[location_duplicates[0] == True]
location_duplicates = location_duplicates.groupby(['id', 'name'], as_index=False).agg('first')

# Concatenate these new unique [id, name] pairs with the already unique pairs
location_map = pd.concat(
    [location_unique[['id', 'name', 'lat', 'lon']],
     location_duplicates[['id', 'name', 'lat', 'lon']]])

# At this point we have unique [id, name] pairs with unique lat/lon values. We
# merge these with the city data by [id, name].
city_data = city_data[['location_type', 'id', 'name', 'type',
                       'measurement', 'value']] \
    .merge(location_map, how='left',
           on=['id', 'name'], validate='many_to_one')

# We now check for sites with missing lat/lon in the city_data (10448 missing values).
# Check how many site IDs are present in our location_map. This may reflect a
# difference in names
non_missing_cities = city_data[city_data['lat'].notna()].copy()
missing_cities = city_data[city_data['lat'].isna()].copy()
missing_cities_ids = set(missing_cities['id'].unique())
# Note that these belong to 2210 unique sites ( missing_cities_ids.shape )

location_map_ids = set(location_map['id'].unique())
intersect_ids = missing_cities_ids.intersection(location_map_ids)

# We have 429 site IDS which are present in our location_map, under different
# names. ( len(intersect_ids) ) We will grab the locations from the location_map
# though the name does not match. It appears this is reasonable, as for example
# id 000052301 is under the name 'Saint-Faustin' or 'Saint-Faustin-Lac-Carraes'
# in the location map, but hast the name 'Saint-Faustin-Lac-Ca' in city_data
# We will grab the first matching ID
missing_cities = \
    missing_cities[['location_type', 'id', 'name', 'type', 'measurement', 'value']] \
        .merge(location_map[['id', 'lat', 'lon']], how='left', on='id', validate='many_to_one')

# NOTE: The majority of missings are related to 'JPN Site XXXX'


# city_data = city_data[['location_type', 'id', 'name', 'type', 'measurement',
#                       'value', 'lat', 'lon']]


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
