import google_streetview.api
from google_streetview import helpers
import pandas as pd
import numpy as np
import os
import random
import imageio
import math
import h5py

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
    
    #performs the operation, the image as numpy array, the status
    #and the metadata in case of error
    """
    #adding the relevant api key before running the code
    params_no_key['key']= api['key'][api_pos]
    
    #computing the image
    results = google_streetview.api.results([params_no_key])
    
    #downloading the image if it exists
    if results.metadata[0]['status'] == 'OK':
        #using helper function from original code to download the image
        helpers.download(results.links[0], directory+"/"+name + '.jpg')

        api['uses'][api_pos]+=1 #counting the api key use if we have success
        
        #now returning numpy image and status 1
        return np.array(imageio.imread(directory+"/"+name + '.jpg')), 1 , None
        
    else:        
        #if there is an error, we return no image and status 0
        return None, 0, results.metadata[0]


#we import the dataset with the APIs
api=pd.read_pickle('api.pkl')

#we see which API key we are using to start running the code
limit_uses=int(500/7*1000)
if api['uses'][0]+10<limit_uses:
    api_pos=0
else:
    api_pos=1


#arguments and KEY
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
unique_ids = pd.read_csv(r'C:/Users/nsuar/Google Drive/Carbon_emissions/urban_emissions_git/urban_emissions/01_Data/ozone_splits.csv'
                         , dtype={'Unique_ID': str, 'dataset': str})
unique_ids = unique_ids[unique_ids['dataset'] == split]

# Load the lat/lon coords for each Unique ID
lat_lon = pd.read_csv(r'C:/Users/nsuar/Google Drive/Carbon_emissions/urban_emissions_git/urban_emissions/01_Data/01_Carbon_emissions/Airnow/World_all_locations_2020_avg_clean.csv' )
#keeping observations for OZONE only
lat_lon = lat_lon[lat_lon['type']=="OZONE"]
#keeping 4 variables and dropping duplicates
lat_lon=lat_lon[['Unique_ID','lat','lon','value']].drop_duplicates(subset=['Unique_ID'])

#merging and forming original dataset
original_data=pd.merge(unique_ids, lat_lon, how="left", on="Unique_ID")
N=len(original_data)

#output dataset
new_data_labels= pd.DataFrame(columns=['Unique_ID', 'img_number'])
new_data_images= []
pos=0 #initial position in the new data array


# #loading data if code suddenly stops / re generate data if 
# os.chdir(r'C:\Users\nsuar\Google Drive')
# for i in range(N):
#     if (i+1) % 50 == 0: #printing message every 50 images
#         print('Converting to numpy images for unique id '+str(i+1)+' out of '+str(N))
#     #sampling to get the random numbers right
#     #_ = getGridSample(0, 0, n)
#     for j in range(10):
#         if os.path.exists('gsv_images'+"/"+original_data.loc[i,'Unique_ID'].replace("/","-")+'_'+str(j) + '.jpg'):
#             #adding data labels
#             new_data_labels.loc[pos]= [ original_data.loc[i,'Unique_ID'] , j] 
#             #adding the images to list
#             new_data_images.append( np.array(imageio.imread('gsv_images'+"/"+original_data.loc[i,'Unique_ID'].replace("/","-")+'_'+str(j) + '.jpg')) )
#             #last_index=i
#             pos += 1


#log of errors
pos_error=0
error_log = pd.DataFrame(columns=['Unique_ID','img_number','error_log'])
error_log['error_log']=error_log['error_log'].astype(object)

# Loop over Unique_IDs
#for i in range(N):
for i in range(10678,N):
    if (i+1) % 5 == 0: #printing message every 5 images
        print('Downloading images for unique id '+str(i+1)+' out of '+str(N))
        print('Current number of usages in API key: '+str(api['uses'][api_pos])+' out of '+ str(limit_uses) )
        print()
        
    # Get a list of n samples around that coordinate
    samples = getGridSample(original_data.loc[i,'lat'], original_data.loc[i,'lon'], n)
    
    #checking API key before loop
    if api['uses'][api_pos]+ n  >= limit_uses:
        api_pos=1
        print('Changing to second API key')
        
    # Download each image
    for j, sample in enumerate(samples):
        #parameters for Street View
        #size is {width}x{height}
        params_no_key = {'location': '%f,%f' %(sample[0],sample[1]),
        'size': '%dx%d' %(res,res),
        'radius': grid_size,
        'fov': fov,
        'source': 'outdoor'}
        
        #downloading the images        
        gsv_image, status, metadata= downloadImage_v2( params_no_key=params_no_key,
                                        name=original_data.loc[i,'Unique_ID'].replace("/","-")+'_'+str(j),
                                        directory='gsv_images',
                                        api_pos=api_pos)
        #if the image is ok, we store it in the new_data
        if status==1:
            new_data_labels.loc[pos,'Unique_ID']=original_data.loc[i,'Unique_ID']    
            new_data_labels.loc[pos,'img_number']=j
            new_data_images.append(gsv_image)
            pos += 1
        else:
            error_log.loc[pos_error, 'Unique_ID']=original_data.loc[i,'Unique_ID']    
            error_log.loc[pos_error, 'img_number']=j
            error_log.loc[pos_error, 'error_log']=[metadata]
            pos_error += 1
        

    #updating the stored version of the api dataset
    api.to_pickle('api.pkl')

    
#average running time: 9.54953236579895 seconds per unique_id
#estimated running time: 29 hours

#adding ozone values to labeled dataset
new_data_labels=pd.merge(new_data_labels,original_data[['Unique_ID','value']], how="left", on="Unique_ID")
    
#storing final array as pickle
new_data_labels[['Unique_ID','img_number']].to_csv("Carbon_emissions/urban_emissions_git/urban_emissions/01_Data/03_processed_data/OZONE/gsv_train_Unique_IDs.csv",index=False)

#storing labels as h5
gsv_train_labels=new_data_labels['value'].to_numpy()
with h5py.File("Carbon_emissions/urban_emissions_git/urban_emissions/01_Data/03_processed_data/OZONE/gsv_train_labels.h5", 'w') as hf:
    hf.create_dataset('gsv_train_labels',  data=gsv_train_labels, compression="gzip", compression_opts=7)

#converting list with images to numpy array, shape (N,H,W,C)
new_data_images= np.transpose (np.array(new_data_images), (0, 2, 1, 3))

#storing as h5
with h5py.File("Carbon_emissions/urban_emissions_git/urban_emissions/01_Data/03_processed_data/OZONE/gsv_train_images.h5", 'w') as hf:
    hf.create_dataset('gsv_train_images',  data=new_data_images, compression="gzip", compression_opts=7)
    
    
    




