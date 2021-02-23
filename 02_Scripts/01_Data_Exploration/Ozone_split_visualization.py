import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

# Load the final database and the ozone train/dev/test splits
db = pd.read_csv(
    '01_Data/01_Carbon_emissions/AirNow/World_all_locations_2020_avg_clean.csv',
    dtype={
        'Unique_ID': str, 'Location_type': str,
        'Zipcode': str, 'County': str, 'type': str,
        'measurement': str, 'value': float, 'lat': float,
        'lon': float, 'AQI_level': str})
splits = pd.read_csv('01_Data/03_Processed_data/OZONE/ozone_splits.csv')

# Filter for ozone readings and get split
ozone = db[db['type'] == 'OZONE']
ozone = ozone.merge(splits, how='left', validate='one_to_one', on='Unique_ID')

# Plot
fig = px.scatter_geo(ozone,
                     lat=ozone['lat'],
                     lon=ozone['lon'],
                     color=ozone['dataset'],
                     hover_name=ozone["Location_type"])
fig.show()
