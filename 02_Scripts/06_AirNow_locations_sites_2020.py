import pandas as pd

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

# Concatenate
combined_data = pd.concat([city_data, site_data], axis=0)

# Drop missing 'value' rows
combined_data.dropna(subset=['value'], axis=0, inplace=True)

# Save combined data
combined_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_locations_2020_avg.csv',
    index=False)
