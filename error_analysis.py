import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"
import os
import seaborn as sns

import visualization_L8SR


# Helper functions to visualize the images of a Unique ID
def visualize_sat(uid_, name):
    """
    Save the visualization of the sat image relating to a uid_ with name 'name'
    :return: None
    """
    # Get lat, lon
    lat = test_pred[test_pred['Unique_ID'] == uid_].iloc[0]['lat']
    lon = test_pred[test_pred['Unique_ID'] == uid_].iloc[0]['lon']

    # Call Sat visualization function
    visualization_L8SR.visualization(
        lat, lon,
        os.path.join('04_Results', 'sat_{}_{}'.format(name, uid_)), mask=True)


def visualize_street(uid_):
    """

    :return:
    """
    pass # TODO


# Load prediction files
test_pred = pd.read_csv(
    '01_Data/03_Processed_data/OZONE/Extracted_features/predict_test.csv')

# Load base emissions file and filter for ozone measurements
airnow = pd.read_csv(
    '01_Data/01_Carbon_emissions/AirNow/world_all_locations_2020_avg_clean.csv',
    dtype={
        'Unique_ID': 'string', 'Location_type': 'string',
        'Zipcode': 'string', 'County': 'string', 'type': 'string',
        'measurement': 'string', 'value': float, 'lat': float,
        'lon': float, 'AQI_level': 'string'}
)
airnow = airnow[airnow['type'] == 'OZONE']

# Get lat, lon and location type
test_pred = test_pred.merge(
    airnow[['Unique_ID', 'Location_type', 'lat', 'lon']],
    how='left', on='Unique_ID', validate='many_to_one')

# Compute error
test_pred['error'] = test_pred['label'] - ['prediction']
test_pred['abs_error'] = abs(test_pred['error'])

# Plot error distribution
sns.histplot(
    test_pred,
    x='error', hue='Location_type',
    multiple='stack'
)

# Visualize errors geographically
fig = px.scatter_geo(test_pred,
                     lat=test_pred['lat'],
                     lon=test_pred['lon'],
                     color=test_pred['error'],
                     hover_name=test_pred["Unique_ID"],
                     opacity=0.5)
fig.show()

# Visualize absolute errors geographically
fig = px.scatter_geo(test_pred,
                     lat=test_pred['lat'],
                     lon=test_pred['lon'],
                     color=test_pred['abs_error'],
                     hover_name=test_pred["Unique_ID"],
                     opacity=0.5)
fig.show()

# Visualize most and least accurate locations
top10 = test_pred.copy().sort_values('abs_error')
top10_uids = top10.iloc[:10]['Unique_ID'].to_list()

for i, uid in enumerate(top10_uids):
    print('Unique ID: {}'.format(uid))
    visualize_sat(uid, 'top_{}'.format(i))
    visualize_street(uid)

bot10 = test_pred.copy().sort_values('abs_error', ascending=False)
bot10_uids = bot10.iloc[:10]['Unique_ID'].to_list()
for i, uid in enumerate(bot10_uids):
    print('Unique ID: {}'.format(uid))
    visualize_sat(uid)
    visualize_street(uid)
