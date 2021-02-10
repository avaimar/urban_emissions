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

# Import county-level information from AirNow
county_data = pd.read_table(
    'https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/reporting_area_locations_V2.dat',
    sep='|', header=None,
    names=['County', 'State', 'Country', 'X', 'Y', 'lat', 'lon', 'TimeZone',
           'Z', 'Time1', 'Time2', 'ReportingAreaID', 'W', 'V'])
county_data = county_data[['County', 'ReportingAreaID', 'lat', 'lon']]

# Import clean data on monitoring sites
site_data = pd.read_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_locations_2020_avg_clean.csv')

# Add site / zip code identifiers
site_data['Location_type'] = 'Site'
zip_code_data['Location_type'] = 'Zip_code'
county_data['Location_type'] = 'County'

# Expand zip code and county data to account for multiple measurements
measurements = site_data['type'].unique()
complete_zip_code_data = pd.DataFrame()
complete_county_data = pd.DataFrame()

for measurement in measurements:
    z_temp_data = zip_code_data.copy()
    c_temp_data = county_data.copy()

    z_temp_data['type'] = measurement
    c_temp_data['type'] = measurement

    complete_zip_code_data = complete_zip_code_data.append(z_temp_data)
    complete_county_data = complete_county_data.append(c_temp_data)

# 1. Add air quality data to zip codes -----------------------------------
# Note: In reporting_area, a ReportingAreaName can map to multiple sites. We will
# map zip codes to all sites in the ReportingArea and later filter for the zip_code /
# ReportingArea combination which is closest in lat/lon terms.
complete_zip_code_data = complete_zip_code_data.merge(
    reporting_area[['ReportingAreaName', 'SiteID', 'SiteLat', 'SiteLong']],
    how='left', left_on=['City'], right_on=['ReportingAreaName'])

# Filter out zip codes with no related sites
complete_zip_code_data.dropna(subset=['SiteID'], axis=0, inplace=True)

# Compute lat/lon distance and filter from smallest to largest
complete_zip_code_data['dist'] = \
    (complete_zip_code_data['Latitude'] - complete_zip_code_data['SiteLat'])**2 +\
    (complete_zip_code_data['Longitude'] - complete_zip_code_data['SiteLong'])**2
complete_zip_code_data.sort_values(by='dist', inplace=True)
complete_zip_code_data.drop_duplicates(
    subset=['Zipcode', 'type'], keep='first', inplace=True)

# Add pollutant information
complete_zip_code_data = complete_zip_code_data.merge(
    site_data[['id', 'type', 'measurement', 'value', 'AQI_level']],
    how='left', left_on=['SiteID', 'type'], right_on=['id', 'type'],
    validate='many_to_one'
)

# Drop missing 'value' rows --- these are rows for which the zipcode could
# not be mapped to one of our sites
complete_zip_code_data.dropna(subset=['value'], inplace=True)

# Check that zipcode/type/lat/lon mappings are unique
test = complete_zip_code_data.groupby(
    ['Zipcode', 'Latitude', 'Longitude', 'type'])['SiteID'].count()
test = complete_zip_code_data.groupby('Zipcode')['Latitude'].nunique()\
    .sort_values(ascending=False)

# Get desired columns
complete_zip_code_data = complete_zip_code_data[
    ['Location_type', 'Zipcode', 'Latitude', 'Longitude', 'type',
     'measurement', 'value', 'AQI_level']]

# 2. Add air quality data to counties -----------------------------------
# Merge county and Site ID via the ReportingAreaName
# There are multiple rows with the same ReportingAreaName in the reporting_area
# dataset; we will once more use the county/ReportingArea combination with the
# closest lat/lon coordinates
complete_county_data = complete_county_data.merge(
    reporting_area[['ReportingAreaID', 'SiteID', 'SiteLat', 'SiteLong']],
    how='left', on=['ReportingAreaID'])

# Filter out counties with no related sites
complete_county_data.dropna(subset=['SiteID'], axis=0, inplace=True)

# Compute lat/lon distance and filter from smallest to largest
complete_county_data['dist'] = \
    (complete_county_data['lat'] - complete_county_data['SiteLat'])**2 +\
    (complete_county_data['lon'] - complete_county_data['SiteLong'])**2
complete_county_data.sort_values(by='dist', inplace=True)
complete_county_data.drop_duplicates(
    subset=['County', 'type'], keep='first', inplace=True)

# Add pollutant information
complete_county_data = complete_county_data.merge(
    site_data[['id', 'type', 'measurement', 'value', 'AQI_level']],
    how='left', left_on=['SiteID', 'type'], right_on=['id', 'type'],
    validate='many_to_one'
)

# Drop missing 'value' rows --- these are rows for which the county could
# not be mapped to one of our sites
complete_county_data.dropna(subset=['value'], inplace=True)

# Check that county/type/lat/lon mappings are unique
test = complete_county_data.groupby(
    ['County', 'lat', 'lon', 'type'])['SiteID'].count()
test = complete_county_data.groupby('County')['lat'].nunique()\
    .sort_values(ascending=False)

# Get desired columns
complete_county_data = complete_county_data[
    ['Location_type', 'County', 'lat', 'lon', 'type',
     'measurement', 'value', 'AQI_level']]

# Create IDs
site_data['Unique_ID'] = 'S_' + site_data['id']
complete_zip_code_data['Unique_ID'] = 'ZC_' + complete_zip_code_data['Zipcode'].astype(str)
complete_county_data['Unique_ID'] = 'C_' + complete_county_data['County']

# site_data: Select columns and append
site_data['Zipcode'] = np.nan
site_data['County'] = np.nan
selected_cols = ['Unique_ID', 'Location_type', 'Zipcode', 'County',
                 'type', 'measurement', 'value', 'lat', 'lon', 'AQI_level']
site_data = site_data[selected_cols]

# zip_code_data: Rename columns
complete_zip_code_data['County'] = np.nan
complete_zip_code_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'},
                              inplace=True)
complete_zip_code_data = complete_zip_code_data[selected_cols]

# county_data: Rename columns
complete_county_data['Zipcode'] = np.nan
complete_county_data = complete_county_data[selected_cols]

# Combine datasets
combined_data = pd.concat([site_data, complete_zip_code_data, complete_county_data])

# Export to csv
combined_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_all_locations_2020_avg_clean.csv',
    index=False)
