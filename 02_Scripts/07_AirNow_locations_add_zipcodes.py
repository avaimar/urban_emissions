import numpy as np
import pandas as pd


# Import data on zip code mappings from AirNow
zip_code_data = pd.read_table(
    'https://files.airnowtech.org/airnow/today/cityzipcodes.csv',
    sep="|")

# Import reporting area information from AirNow
reporting_area = pd.read_csv(
    'https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/Site_To_ReportingArea.csv',
    encoding='ISO-8859-1')

# Import clean data on monitoring sites
site_data = pd.read_csv('../01_Data/01_Carbon_emissions/AirNow/World_locations_2020_avg_clean.csv')

# Add site / zip code identifiers
site_data['Location_type'] = 'Site'
zip_code_data['Location_type'] = 'Zip_code'

# Merge zip code and Site ID via the ReportingAreaName
# There are multiple rows with the same ReportingAreaName in the reporting_area
# dataset; we need a single one in order to map zipcodes.

zip_code_data = zip_code_data.merge(
    reporting_area[['ReportingAreaName', 'SiteID']],
    how='left', left_on='City', right_on='ReportingAreaName',
    validate='many_to_one')


# Expand zip code data to account for multiple measurements
measurements = site_data['type'].unique()
complete_zip_code_data = pd.DataFrame()

for measurement in measurements:
    temp_data = zip_code_data.copy()
    temp_data['type'] = measurement
    complete_zip_code_data = complete_zip_code_data.append(temp_data)

# Add air quality data to zip codes
# Note: In site_data, a city ('name') can map to multiple sites. We will
# map zip codes to all sites in the city and later filter for the zip_code /
# site combination which is closest in lat/lon terms.
complete_zip_code_data = complete_zip_code_data.merge(
    site_data[['name', 'type', 'measurement', 'value', 'AQI_level', 'lat', 'lon']],
    how='left', left_on=['City', 'type'], right_on=['name', 'type'])

# Filter out zip codes with no related sites
complete_zip_code_data.dropna(subset=['name'], axis=0, inplace=True)

# For zip codes with unique sites, filter for those with
complete_zip_code_data['dist'] = \
    (complete_zip_code_data['Latitude'] - complete_zip_code_data['lat'])**2 +\
    (complete_zip_code_data['Longitude'] - complete_zip_code_data['lon'])**2
complete_zip_code_data.sort_values(by='dist', inplace=True)
complete_zip_code_data.drop_duplicates(
    subset=['Zipcode', 'type'], keep='first', inplace=True)

# Create IDs -- this will allow us to count unique locations even if
# a location id and zip code are the same
site_data['Unique_ID'] = 'S_' + site_data['id']
complete_zip_code_data['Unique_ID'] = 'ZC_' + complete_zip_code_data['Zipcode'].astype(str)

# Select columns and append
site_data['Zipcode'] = np.nan
selected_cols = ['Unique_ID', 'Location_type', 'Zipcode', 'name', 'type',
                 'measurement', 'value', 'lat', 'lon', 'AQI_level']
site_data = site_data[selected_cols]

# Drop the site lat/lon and keep the zipcode lat/lon
complete_zip_code_data.drop(columns=['lat', 'lon'], inplace=True)
complete_zip_code_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'},
                              inplace=True)
complete_zip_code_data = complete_zip_code_data[selected_cols]
combined_data = site_data.append(complete_zip_code_data)

# Export to csv
combined_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_locations_and_zipcodes_2020_avg.csv',
    index=False)