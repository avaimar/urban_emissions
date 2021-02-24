import google_streetview.api
from google_streetview import helpers
import pandas as pd
import numpy as np
import os
import random
import imageio
import math

os.chdir(r'C:\Users\nsuar\Google Drive\Carbon_emissions\urban_emissions_git\urban_emissions\imagery\streetview')


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


def downloadImage_v2(params_no_key, name, directory, api_pos):
    """
    params_no_key: params dictionary of parameters to pass to google streetview API, without API key
    name: name of the image (string)
    directory: path where we save the images
    api_pos: number of the corresponding api key
    
    #performs the operation, and return status of the task and the image as numpy array
    """
    #adding the relevant api key before running the code
    params_no_key['key']= api['key'][api_pos]
    
    #computing the image
    results = google_streetview.api.results([params_no_key])
    
    #downloading the image if it exists
    status=0
    image=None
    if results.metadata[0]['status'] == 'OK':
        #using helper function from original code to download the image
        helpers.download(results.links[0], directory+"/"+name + '.jpg')
        #now converting it to Numpy array
        image=np.array(imageio.imread(directory+"/"+name + '.jpg'))
        status=1 #storing status of the image based on the metadata being ok
        api['uses'][api_pos]+=1 #counting the api key use if we have success
    
    return image, status


#defining dataframe with API keys
api = pd.DataFrame( index=range(2), columns=['key','uses'])
api['uses']=0
api['key'][0]='AIzaSyBqeXF0gai0zAp1zStwACF_lplAHIU8VaI'
api['key'][1]='AIzaSyCMU0cFUZF6_iuzzvlr3zd_4C7NFAmEisU'

#we import the dataset with the APIs
api=pd.read_pickle('api.pkl')

#we see which API key we are using to start running the code
limit_uses=int(500/7*1000)
if api['uses'][0]<limit_uses:
    api_pos=0
else:
    api_pos=1


#arguments and KEY
split_csv  = "../../01_Data/ozone_splits.csv"
lat_lon_csv = "../../01_Data/01_Carbon_emissions/Airnow/World_all_locations_2020_avg_clean.csv" 
split = "train"
n = 10


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
#keeping 3 variables and dropping duplicates
lat_lon=lat_lon[['Unique_ID','lat','lon']].drop_duplicates(subset=['Unique_ID'])

#merging and forming original dataset
original_data=pd.merge(unique_ids, lat_lon, how="left", on="Unique_ID")
N=len(original_data)

#output dataset
new_data = pd.DataFrame(columns=['Unique_ID', 'image','img_number','status'])
new_data['image']=np.nan
new_data['image']=new_data['image'].astype(object)


# Loop over Unique_IDs
for i in range(N):
    if (i+1) % 5 == 0: #printing message every 5 images
        print('Downloading images for unique id '+str(i+1)+' out of '+str(N))
        print()
        print('Current number of usages in API key: '+str(api['uses'][api_pos])+' out of '+ str(limit_uses) )
    
    # Get a list of n samples around that coordinate
    samples = getGridSample(original_data.loc[i,'lat'], original_data.loc[i,'lon'], n)
    
    #checking before loop
    if api['uses'][api_pos]+ n  >= limit_uses:
        api_pos=1
        print('Changing to second API key')
        
    
    #initial position in the array
    pos=i*n
    # Download each image
    for j, sample in enumerate(samples):
        #storing unique ID
        new_data.loc[pos+j,'Unique_ID']=original_data.loc[i,'Unique_ID']
        #image number
        new_data.loc[pos+j,'img_number']=j
        params_no_key = {'location': '%f,%f' %(sample[0],sample[1]),
            'size': '%dx%d' %(res,res),
            'radius': grid_size,
            'fov': fov,
            'source': 'outdoor'}
        #we are storing the status to then take its mean
        #and downloading the images to the specified folder
        new_data.loc[pos+j,'image'], new_data.loc[pos+j,'status'] = downloadImage_v2( params_no_key=params_no_key,
                                        name=original_data.loc[i,'Unique_ID']+'_'+str(j),
                                        directory='street_view_images',
                                        api_pos=api_pos)
    #updating the stored version of the api dataset
    api.to_pickle('api.pkl')

    
#average running time: 9.54953236579895 seconds per unique_id
#estimated running time: 29 hours
    
#storing final array as pickle
new_data.to_pickle('api.pkl')

    

    






