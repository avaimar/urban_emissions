import numpy as np
import pandas as pd
import os
import imageio
import requests, zipfile, io
import pickle
import time
import math  
from geetools import cloud_mask
import h5py

    
def image_task(image,point,image_res,n_pixels,drive_folder,image_name):
    """
    function to download satellite images from a ee.imageCollection object.
    We first generate a bounding box of image_res*n_pixels meters around "point",
    then we clip that region from the image collection, take the mean image from the collection,
    and send that as a task to the Google Earth Engine. We can define a folder name at the root folder
    of the Google Drive account of the user, and the image will appear there as a GeoTIFF file.
    
    Inputs:
    -image= ee.ImageCollection object
    -point= ee.Geometry.Point object
    -image_res= resolution of the image in meters
    -n_pixels= number of pixels to extract on the images
    -drive_folder= string with Google drive folder to store the TIFF image
    -image_name= string with the image_name for the TIFF image

    Output:
     task= an EE task object. we can then use task.status() to check the status of the task.
     If the task is completed, we will see a TIFF image in the "drive_folder" with name "image_name.tif".
     The image has 3 dimensions, where the first 2 are n_pixels, and the 3rd is the number of bands of "image".
    """
    len=image_res*n_pixels # for landsat, 30 meters * 224 pixels
    region= point.buffer(len/2).bounds().getInfo()['coordinates']
    #defining the rectangle
    coords=np.array(region)
    #taking min and maxs of coordinates to define the rectangle
    coords=[np.min(coords[:,:,0]), np.min(coords[:,:,1]), np.max(coords[:,:,0]), np.max(coords[:,:,1])]
    rectangle=ee.Geometry.Rectangle(coords)

    #generating the export task
    task=ee.batch.Export.image.toDrive(image=image.filterBounds(rectangle).mean(), 
                        folder=drive_folder, 
                        description=image_name, 
                        region=str(region), dimensions=str(n_pixels)+"x"+str(n_pixels))
    
    #starting the task
    task.start()
    
    return task




#importing Earth Engine
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()

#reading data
dataset=pd.read_csv(r'C:\Users\nsuar\Google Drive\Carbon_emissions\urban_emissions_git\urban_emissions\01_Data\01_Carbon_emissions\AirNow\world_all_locations_2020_avg_clean.csv')

#keeping relevant data, and re-starting the indexes of the entries to loop over them later
dataset=dataset[['Unique_ID','lat','lon']].drop_duplicates(subset=['Unique_ID']).reset_index(drop=True)

#column for task status
dataset['task_status']=''


#defining image
startDate = '2020-01-01'
endDate = '2020-12-31'
landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
landsat = landsat.filterDate(startDate, endDate) # filter date
landsat = landsat.select(["B1","B2","B3","B4","B5","B6","B7"]) # select 7 first bands

#list to store the task objects
tasks=[]

#numpy matrix for the images
images=np.full([224,224,7,len(dataset)], np.nan)

#running the tasks to get the imagery
N=len(dataset)
batch_size=100

for j in range( math.ceil(N/batch_size) ):
    #determining batch lower and upper indexes, given batch size
    #lower is always fixed
    lower_i=batch_size*j
    #upper can vary at the end of the list
    if batch_size*(j+1)>N:
        upper_i=N
    else:
        upper_i=batch_size*(j+1)
        
    #generating the tasks for all the images in the batch
    for i in range(lower_i,upper_i):
        tasks.append(image_task(image=landsat,
                            point=ee.Geometry.Point(dataset['lon'][i],dataset['lat'][i] ),
                            image_res=30,n_pixels=224,
                            drive_folder='landsat_sr_no_mask',image_name='landsat_no_mask'+str(i)))
    
    #printing message:
    print('Batch '+str(j+1)+': Retrieving images '+str(lower_i+1)+' to '+str(upper_i)+' of a total of '+str(N))
    
        
    #checking status of the mentioned tasks
    batch_status=dataset.loc[lower_i:upper_i-1,'task_status'].value_counts() #counting status of the tasks
    
    while batch_status.get('COMPLETED',0) + batch_status.get('FAILED',0)< batch_size: #checking that not all tasks are done
        time.sleep(10) #running the code every 10 seconds
        for i in range(lower_i,upper_i):
            #checking status of each task
            if dataset.loc[i,'task_status']=='' or dataset.loc[i,'task_status']=='READY' or dataset.loc[i,'task_status']=='RUNNING':
                dataset.loc[i,'task_status']=tasks[i].status()['state']
                                    
        #updating batch status
        batch_status=dataset.loc[lower_i:upper_i-1,'task_status'].value_counts()
        #reporting them back
        print('Status of batch '+str(j+1)+':')
        print('completed images= '+str(batch_status.get('COMPLETED',0)))
        print('failed images= '+str(batch_status.get('FAILED',0)))
        print('pending images= '+str(batch_size-batch_status.get('COMPLETED',0)-batch_status.get('FAILED',0)))
        print('------------------')
        
    #when the batch is done, pass images to the corresponding file
    time.sleep(5) #waiting for images to sync in Google Drive
    for i in range(lower_i, upper_i):            
        if dataset.loc[i,'task_status']=='COMPLETED' :
            images[:,:,:,i]=np.array(imageio.imread(r'C:\Users\nsuar\Google Drive\landsat_sr_no_mask\landsat_no_mask'+str(i)+'.tif'))
    print('All the completed images of batch '+str(j+1)+' were successfully added to the numpy array')


#storing both arrays here

#images to h5 format
with h5py.File(r'C:\Users\nsuar\Google Drive\Carbon_emissions\data\imagery_no_mask.h5', 'w') as hf:
    hf.create_dataset("imagery_no_mask",  data=images,  data=images, compression="gzip", compression_opts=9)


#dataset as pickle
dataset.to_pickle(r'C:\Users\nsuar\Google Drive\Carbon_emissions\data\imagery_no_mask_datapoints.pkl')