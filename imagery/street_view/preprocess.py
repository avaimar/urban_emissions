import google_streetview.api
import numpy as np
import pandas as pd

def get_unique_locations(dataset, unique_filename):
    """
    Average the lat/lon for each Unique ID
    Input:
        dataset             DataFrame with 'lat', 'lon', and 'Unique_ID' columns
        unique_filename     .csv filename to save output to with rows 'lat', 'lon',
                                and 'Unique_ID'
    """
    unique_ids = {}
    for index, row in dataset.iterrows():
        if (index % 100 == 0):
            print('on row ', index)
        lat, lon = row['lat'], row['lon']
        unique_id = row['Unique_ID']
        if unique_id in unique_ids:
            unique_ids[unique_id].append([lat,lon])
        else:
            unique_ids[unique_id] = [[lat,lon]]

    unique_df = {'Unique_ID': [], 'lat': [], 'lon': []}
    for unique_id in unique_ids:
        lat_lon_array = np.array(unique_ids[unique_id])
        lat, lon = np.mean(lat_lon_array[:,0]), np.mean(lat_lon_array[:,1])
        unique_df['Unique_ID'].append(unique_id)
        unique_df['lat'].append(lat)
        unique_df['lon'].append(lon)
    
    unique_df = pd.DataFrame(unique_df)
    unique_df.to_csv(unique_filename)

def get_nonempty(unique_filename, nonempty_filename):
    locations = pd.read_csv(unique_filename)
    num_empty = 0
    locations_with_images = {'Unique_ID': [], 'lat': [], 'lon': []}
    for index, row in locations.iterrows():
        unique_id, lat, lon = row['Unique_ID'], row['lat'], row['lon']
        if (index % 10 == 0):
            print('Calling index ', index, '...', end='')
            print('(%f, %f)' %(lat,lon))
        params = [{'size': '{}x{}'.format(res,res), 'location': '%f,%f' %(lat,lon), 'key': key, 'radius': radius}]
        results = google_streetview.api.results(params)
        metadata = results.metadata[0]
        if (metadata['status'] == 'ZERO_RESULTS'):
            num_empty += 1
        else:
            locations_with_images['lat'].append(lat)
            locations_with_images['lon'].append(lon)
            locations_with_images['Unique_ID'].append(unique_id)
        if (index%100 == 0):
            print('\nNum empty = ', num_empty, '\n')
    
    locations_with_images = pd.DataFrame(locations_with_images)
    locations_with_images.to_csv(nonempty_filename)
    print('Number of unique locations = ', len(locations))
    print('Number of locations with no images within %dm is %d' %(radius, num_empty))


if __name__ == '__main__':

    # 1. Define variables
    res = 224 # pixels
    radius = 6720 # image search radius, in meters
    key = 'AIzaSyAKWO7ci4SdC4Pts-qfPYwEagsN3UabCy8' # Google Maps API key
    data_filename = '../../01_Data/01_Carbon_emissions/AirNow/World_locations_and_zipcodes_2020_avg.csv'
    unique_filename = 'unique_locations.csv'
    nonempty_filename = 'locations_with_images.csv'

    # 2. Load the Carbon Emissions dataset and get average lat/lon values for 
    #    each Unique_ID.
    dataset = pd.read_csv(data_filename)
    get_unique_locations(dataset, unique_filename)

    # 3. Check how many of the locations have an image within the radius.
    get_nonempty(unique_filename, nonempty_filename)