import google_streetview.api
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image

# 1. Define image parameters
res = 224 # image resolution
key = 'AIzaSyAKWO7ci4SdC4Pts-qfPYwEagsN3UabCy8' # Google Maps API key
radius = 6720 # same as satellite unit
fov = 120 # max zoom out
directions = {'0': 'N', '90': 'E', '180':'S', '270': 'W'}

# 2. Load unique locations with images within 6720m
#    (csv file generated using preprocess.py)
locations_filename = 'locations_without_images.csv' # 'locations_with_images.csv'
locations = pd.read_csv(locations_filename, chunksize=50)

# 3. Download images for each location in batches
iter = 0
for locations_chunk in locations:
    print('on chunk ', iter)

    # delete all images currently in directory
    for folder in os.listdir():
        if (os.path.isdir(folder)):
            for f in os.listdir(folder):
                for f_ in os.listdir(os.path.join(folder,f)):
                    os.remove(os.path.join(folder,f, f_))
                os.rmdir(os.path.join(folder,f))
            os.rmdir(folder)

    # save all the images for this chunk
    for index, row in locations_chunk.iterrows():
        unique_id, lat, lon = row['Unique_ID'], row['lat'], row['lon']
        for heading in directions:
            params = [{'location': '%f,%f' %(lat,lon),
                'key': key,
                'size': '%dx%d' %(res,res),
                'radius': radius,
                'fov': fov,
                'heading': heading,
                'source': 'outdoor'}]
            results = google_streetview.api.results(params)
            results.download_links('%s/%s_dir_%s' %(unique_id,unique_id,directions[heading]))
    
    # save all the images from the current chunk to pkl
    images = {'Unique_ID': [], 'N': [], 'E': [], 'S': [], 'W': []}
    dirs = os.listdir()
    for unique_id in dirs:
        if (os.path.isdir(unique_id)):
            images['Unique_ID'].append(unique_id)
            for direction in directions:
                img_fn = os.path.join(unique_id, unique_id+'_dir_'+directions[direction], 'gsv_0.jpg')
                if (os.path.exists(img_fn)):
                    img = np.array(Image.open(img_fn))
                    images[directions[direction]].append(img)
                else:
                    images[directions[direction]].append(np.nan)
    images = pd.DataFrame(images)
    images.to_pickle('../../01_Data/03_Street_images/street_images_missing_%d.pkl' %(iter)) # remove 'missing'
    iter += 1

# 4. Concatenate the pkl files
pkl_folder = '../../01_Data/03_Street_images'
pkls = os.listdir(pkl_folder)
num_pkls = len(pkls)

#   add column to emissions data
emissions_data = pd.read_csv('../../01_Data/01_Carbon_emissions/AirNow/World_locations_and_zipcodes_2020_avg.csv')
emissions_data['streetview'] = np.nan
emissions_data['streetview'] = emissions_data['streetview'].astype(object)

iter = 0
for pkl_file in pkls:
    print('%d/%d loading %s' %(iter,num_pkls, pkl_file))
    chunk = pickle.load(open(os.path.join(pkl_folder, pkl_file), 'rb'))
    for index, row in chunk.iterrows():
        unique_id = row['Unique_ID']
        N_image = row['N']
        E_image = row['E']
        S_image = row['S']
        W_image = row['W']
        img = np.array([N_image, E_image, S_image, W_image])
        rows = emissions_data.loc[emissions_data['Unique_ID']==unique_id].index
        for row_indx in rows:
            emissions_data['streetview'][row_indx] = img
    iter += 1

emissions_data.to_pickle('../../01_Data/03_Street_images/emissions_and_streetview.pkl')