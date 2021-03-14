import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"
import os
import seaborn as sns


import visualization_L8SR


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

# Load street test images
street_test = h5py.File('01_Data/03_Processed_data/OZONE/street_test.hdf5')
street_test_X = street_test['X']
street_test_Y = street_test['Y']


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


def visualize_street(test_row_num_, uid_, name):
    """
    Plot the street image at row test_row_num_ in the street_test.hdf5 file
    :return:
    """
    img = street_test_X[test_row_num_, :]
    img = Image.fromarray(img)
    img.save(os.path.join('04_Results', 'street_{}_{}.png'.format(name, uid_)))


# Get lat, lon and location type
test_pred = test_pred.merge(
    airnow[['Unique_ID', 'Location_type', 'lat', 'lon']],
    how='left', on='Unique_ID', validate='many_to_one')

# Compute error
test_pred['error'] = test_pred['label'] - test_pred['prediction']
test_pred['abs_error'] = abs(test_pred['error'])

# Add image number in order to grab the correct street image for a UID
test_pred['img_num'] = test_pred.index

# Plot error distribution
error_hist = sns.histplot(
    test_pred, x='error', hue='Location_type', multiple='stack', palette='Set2')
error_hist.set(xlabel='Error')
leg = error_hist.axes.get_legend()
leg.set_title('Location type')
for t, l in zip(leg.texts, ['Site', 'Zipcode', 'County']):
    t.set_text(l)
error_hist.get_figure().savefig(os.path.join('04_Results', 'error_histogram.png'))
plt.clf()

# Plot prediction distribution
pred_hist = sns.histplot(
    test_pred, x='prediction', hue='Location_type', multiple='stack', palette='Set2')
pred_hist.set(xlabel='Prediction')
leg = pred_hist.axes.get_legend()
leg.set_title('Location type')
for t, l in zip(leg.texts, ['Site', 'Zipcode', 'County']):
    t.set_text(l)
pred_hist.get_figure().savefig(os.path.join('04_Results', 'pred_histogram.png'))
plt.clf()

# Plot label distribution
lab_hist = sns.histplot(
    test_pred, x='label', hue='Location_type', multiple='stack', palette='Set2')
lab_hist.set(xlabel='Label')
leg = lab_hist.axes.get_legend()
leg.set_title('Location type')
for t, l in zip(leg.texts, ['Site', 'Zipcode', 'County']):
    t.set_text(l)
lab_hist.get_figure().savefig(os.path.join('04_Results', 'lab_histogram.png'))
plt.clf()

# Visualize errors geographically
fig = px.scatter_geo(
    test_pred, lat=test_pred['lat'], lon=test_pred['lon'],
    color=test_pred['error'], hover_name=test_pred["Unique_ID"], opacity=0.5,
    scope='north america', labels={"error": "Error"})
fig.show()

# Visualize absolute errors geographically
fig = px.scatter_geo(
    test_pred, lat=test_pred['lat'], lon=test_pred['lon'],
    color=test_pred['abs_error'], hover_name=test_pred["Unique_ID"], opacity=0.5,
    scope='north america', labels={"abs_error": "Absolute Error"})
fig.show()

# Visualize most and least accurate locations
top10 = test_pred.copy().sort_values('abs_error')
top10_uids = top10.iloc[:10]['Unique_ID'].to_list()
top10_rows = top10.iloc[:10]['img_num'].to_list()

for i, (uid, row) in enumerate(zip(top10_uids, top10_rows)):
    print('Unique ID: {}'.format(uid))
    visualize_sat(uid, 'top_{}'.format(i))
    visualize_street(row, uid, 'top_{}'.format(i))

bot10 = test_pred.copy().sort_values('abs_error', ascending=False)
bot10_uids = bot10.iloc[:10]['Unique_ID'].to_list()
bot10_rows = bot10.iloc[:10]['img_num'].to_list()
for i, (uid, row) in enumerate(zip(bot10_uids, bot10_rows)):
    print('Unique ID: {}'.format(uid))
    visualize_sat(uid, 'bot_{}'.format(i))
    visualize_street(row, uid, 'bot_{}'.format(i))

# How are errors distributed across the labels?
# Are errors higher for lower or higher ozone values?
e_vals = sns.scatterplot(x=test_pred['label'], y=test_pred['error'],
                         hue=test_pred['Location_type'], palette='Set2')
e_vals.axhline(0)
e_vals.set(xlabel='Ozone reading (ppb)', ylabel='Prediction error')
leg = e_vals.axes.get_legend()
leg.set_title('Location type')
for t, l in zip(leg.texts, ['Site', 'Zipcode', 'County']):
    t.set_text(l)
e_vals.get_figure().savefig(os.path.join('04_Results', 'error_scatter.png'))
plt.clf()

# Close dataset
street_test.close()
