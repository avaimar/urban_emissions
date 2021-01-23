import pandas as pd


# Import data from EPA website
epa_counties_xlsx = 'https://www.epa.gov/sites/production/files/2020-06/ctyfactbook2019.xlsx'
df = pd.read_excel(epa_counties_xlsx, skiprows=1, na_values=['ND', 'IN'])

# Rename columns
df.columns = ['State', 'County', 'County_FIPS_Code',
              'Population_2010', 'CO_8hr_ppm', 'Pb_3mo',
              'NO2_Am_ppb', 'NO2_1h_ppb', 'O3_8hr_ppm',
              'PM10_24hr', 'PM25_AM', 'PM25_24hr', 'SO2_1hr_ppb']
df.drop([0], inplace=True)

# Drop rows that are not observations
df.dropna(axis=0, subset=['County_FIPS_Code'], inplace=True)

# Save cleaned data
df.to_csv('../01_Data/01_Carbon_emissions/US_counties_2019_clean.csv')
