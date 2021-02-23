import argparse
from config import KEY
import google_streetview.api
import pandas as pd
from PIL import Image
import math
import numpy as np
import os
import random
import re
import shutil
import sys

PYTHON = sys.executable

# set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--split', default='val', required=True,
                    help='train, val, or test split')
parser.add_argument('-d', '--split-csv', default='../../01_Data/ozone_splits.csv',
                    required=True, help='csv file containing location Unique IDs \
                                         labeled with train, val, or test')
parser.add_argument('-l', '--lat-lon-csv',
                    default='../../01_Data/01_Carbon_emissions/Airnow/World_all_locations_2020_avg_clean.csv',
                    required=True, help='csv file containing Unique IDs and lat/lon coordinates')
parser.add_argument('-n', '--num-samples-per-grid', default=10,
                    required=True, help='Number of locations to sample per spatial grid cell')

# define parameters
earth_radius = 6271
grid_size = 6720 # m
fov = 120
res = 224

random.seed(42)

def getGridSample(lat, lon, n):
    """
    Get a random sampling of n locations within k km from (lat, lon)
    param: lat      latitude of grid center point
    param: lon      longitude of grid center point
    param: n        number of locations to sample
    return:         array of length n of latitude, longitude tuples
    """

    locations = []
    for i in range(n):
        # Get a random point in km's
        min_delta = -1 * grid_size
        max_delta = grid_size
        delta_x_kms = (np.random.rand()*(max_delta - min_delta) + min_delta)/1000 # kms
        delta_y_kms = (np.random.rand()*(max_delta - min_delta) + min_delta)/1000 # kms

        # Convert from km's to latitude/longitude
        delta_lat = (delta_x_kms/earth_radius) * 180/math.pi
        r = earth_radius*math.cos(lat*math.pi/180.0)
        delta_lon = (delta_y_kms/r)*180/math.pi

        # Get the new lat/lon point
        new_lat = lat + delta_lat
        new_lon = lon + delta_lon

        locations.append((new_lat, new_lon))

    return locations


def downloadImage(params, unique_id, index):
    """
    param: params           dictionary of parameters to pass to google streetview API
    param: unique_id        Unique_ID of location
    param: index            index out of n samples for given Unique_ID
    """
    results = google_streetview.api.results(params)
    results.download_links('%s_%s' %(re.sub('/', '-', unique_id) ,index))

def clearDirectory():
    for f in os.listdir():
        if (os.path.isdir(f)):
            shutil.rmtree(f)


def loadData(split_csv, lat_lon_csv, split, n):
    """
    param: split_csv        csv filename with Unique IDs and split name
    param: lat_lon_csv      csv filename with latitude, longitude, and Unique IDs
    param: split            name of split to take images for
    param: n                number of samples per grid
    return: numpy array of shape (num_locations_in_split * n)
    """

    # Load the Unique IDs for the split
    unique_ids = pd.read_csv(split_csv, dtype={'Unique_ID': str, 'dataset': str})
    unique_ids = unique_ids[unique_ids['dataset'] == split]

    # Load the lat/lon coords for each Unique ID
    lat_lon = pd.read_csv(lat_lon_csv)

    # Set up data Dataframe
    data = pd.DataFrame(columns=['Unique_ID', 'lat', 'lon', 'index', 'img'])
    df_index = 0

    # TODO Add chunk index

    # done - 0:10 (0), 10:110 (1), 110:210 (2), 210:310 (3), 310:410 (4),
    #        410:510 (5), 510:610 (6)
    # Loop over Unique_IDs
    for indx, row in unique_ids.iloc[510:610].iterrows():
        print('Loading row ', indx)
        unique_id = row['Unique_ID']

        # Get the corresponding lat/lon center coordinate
        coord_row = lat_lon.loc[lat_lon['Unique_ID']==unique_id].iloc[0]
        lat, lon = coord_row['lat'], coord_row['lon']

        # Get a list of n samples around that coordinate
        samples = getGridSample(lat, lon, n)

        # Delete previously downloaded images
        clearDirectory()

        print('\tDownloading images for unique id ', unique_id)
        # Download each image
        for i, sample in enumerate(samples):
            params = [{'location': '%f,%f' %(sample[0],sample[1]),
                'key': KEY,
                'size': '%dx%d' %(res,res),
                'radius': grid_size,
                'fov': fov,
                'source': 'outdoor'}]
            downloadImage(params, unique_id, i)

        print('Concatenating images')
        # Concatenate the images and save to dataframe
        for i in range(n):
            dir_fn = re.sub('/', '-', unique_id) + '_' + str(i)
            img_fn = os.path.join(dir_fn, 'gsv_0.jpg')
            if (os.path.exists(img_fn)):
                img = np.array(Image.open(img_fn))
            else:
                img = None

            data.loc[df_index] = [unique_id, samples[i][0], samples[i][1], i, img]
            df_index += 1

    data.to_pickle('chunk6.pkl') # TODO CHANGE THIS


if __name__ == '__main__':

    # get arguments from command line
    args = vars(parser.parse_args())
    split_csv = args['split_csv']
    lat_lon_csv = args['lat_lon_csv']
    split = args['split']
    n = int(args['num_samples_per_grid'])
    if (split not in ['train', 'val', 'test']):
        raise Exception('Split must be one of: train, val, test')

    # get images
    loadData(split_csv, lat_lon_csv, split, n)