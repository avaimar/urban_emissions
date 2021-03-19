import numpy as np
import pandas as pd
import pickle
import time
import math  
from geetools import cloud_mask
import apprise

    
def image_task(image,point,image_res,n_pixels,folder_name, image_name,storage="Cloud"):
    
    """
    function to download satellite images from a ee.imageCollection object.
    We first generate a bounding box of image_res*n_pixels meters around "point",
    then we clip that region from the image collection, take the mean image from the collection,
    and send that as a task to the Google Earth Engine. 
    After that, we download the image Google Cloud Storage if storage=="Cloud", 
    or to Google Drive if storage=="Drive".
    
    Inputs:
    -image= ee.ImageCollection object
    -point= ee.Geometry.Point object
    -image_res= resolution of the image in meters
    -n_pixels= number of pixels to extract on the images
    -storage= string indicating if we are storing the images in Google Cloud or Google Drive.
              Defaults to Google Cloud.
    -folder_name= string with Google Cloud bucket name if storage=="Cloud"
                  string with the name of a folder in the root of Google Drive if storage=="Drive"
    -image_name= string with the image_name for the TIFF image.

    Output:
     task= an EE task object. we can then use task.status() to check the status of the task.
     If the task is completed, we will see a TIFF image in "folder_name" with name "image_name.tif".
     The image has 3 dimensions, where the first 2 are n_pixels, and the 3rd is the number of bands of "image".
    """
    #generating the box around the point
    len=image_res*n_pixels # for landsat, 30 meters * 224 pixels
    region= point.buffer(len/2).bounds().getInfo()['coordinates']
    #defining the rectangle
    coords=np.array(region)
    #taking min and maxs of coordinates to define the rectangle
    coords=[np.min(coords[:,:,0]), np.min(coords[:,:,1]), np.max(coords[:,:,0]), np.max(coords[:,:,1])]
    rectangle=ee.Geometry.Rectangle(coords)

    #generating the export task ( dimensions is "WIDTHxHEIGHT"  )
    if storage=="Cloud":
        task=ee.batch.Export.image.toCloudStorage(image=image.filterBounds(rectangle).mean(), 
                            bucket=folder_name, 
                            description=image_name, 
                            region=str(region), dimensions=str(n_pixels)+"x"+str(n_pixels))
    if storage=="Drive":
        task=ee.batch.Export.image.toDrive(image=image.filterBounds(rectangle).mean(), 
                            folder=folder_name, 
                            description=image_name, 
                            region=str(region), dimensions=str(n_pixels)+"x"+str(n_pixels))
    
    
    #starting the task
    task.start()
    return task

def telegram_text(message):
	# create an Apprise instance
	apobj = apprise.Apprise()
	# Adding telegram id
	apobj.add('tgram://1436452271:AAH06lhnvlTMVg_TZKpw7fhZ7VjtRhZ2LLg/110473419/')
	# Sending notification
	apobj.notify(body=message)
    
    
#importing Earth Engine packages
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()


# #reading data
# dataset=pd.read_csv(r'C:\Users\nsuar\Google Drive\Carbon_emissions\urban_emissions_git\urban_emissions\01_Data\01_Carbon_emissions\AirNow\world_all_locations_2020_avg_clean.csv')

# #splits
# splits=pd.read_csv(r"C:\Users\nsuar\Google Drive\Carbon_emissions\urban_emissions_git\urban_emissions\01_Data\ozone_splits.csv")

# #keeping relevant data, and re-starting the indexes of the entries to loop over them later
# dataset = dataset[dataset['type']=="OZONE"]
# dataset=dataset[['Unique_ID','lat','lon','value']]

# #adding splits
# dataset=pd.merge(dataset,splits, how="left", on="Unique_ID")


# dataset=dataset.sort_values(by=['dataset']).reset_index(drop=True)

# #column for task status
# dataset['task_status']='_'

# #saving to CSV

# dataset.to_csv(r'C:\Users\nsuar\Google Drive\local_testing_files\dataset.csv',index=False)

#reading CSV
dataset=pd.read_csv('dataset.csv')

#dataset=pd.read_csv(r'C:\Users\nsuar\Google Drive\local_testing_files\dataset.csv')

print("dataset read succesfully")


#defining image
startDate = '2020-01-01'
endDate = '2020-12-31'
landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
# filter date
landsat = landsat.filterDate(startDate, endDate) 
#applying cloud masking
landsat_masked=landsat.map( cloud_mask.landsatSR(['cloud']) )
landsat_masked=landsat_masked.select(["B1","B2","B3","B4","B5","B6","B7"]) 

# selecting bands for unmasked version
landsat = landsat.select(["B1","B2","B3","B4","B5","B6","B7"]) 

#list to store the task objects
tasks=[]

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
        
    #every 500 iterations, send Telegram message
    if j % 5==0:
        telegram_text(str(j*100)+" images out of "+str(N)+" have been processed")
        
    #generating the tasks for all the images in the batch
    for i in range(lower_i,upper_i):
        tasks.append(image_task(image=landsat_masked,
                                point=ee.Geometry.Point(dataset['lon'][i],dataset['lat'][i] ),
                                image_res=30,
                                n_pixels=224,
                                folder_name='LS8_SR_masked',
                                image_name='image_'+str(i),
                                storage="Drive"))
        
    #printing message:
    print('Batch '+str(j+1)+': Retrieving images '+str(lower_i+1)+' to '+str(upper_i)+' of a total of '+str(N))
    
        
    #checking status of the mentioned tasks
    batch_status=dataset.loc[lower_i:upper_i-1,'task_status'].value_counts() #counting status of the tasks
    
    while batch_status.get('COMPLETED',0) + batch_status.get('FAILED',0)< upper_i - lower_i: #checking that not all tasks are done
        time.sleep(10) #running the code every 10 seconds
        for i in range(lower_i,upper_i):
            #checking status of each task
            if dataset.loc[i,'task_status']=='_' or dataset.loc[i,'task_status']=='READY' or dataset.loc[i,'task_status']=='RUNNING':
                dataset.loc[i,'task_status']=tasks[i].status()['state']
                                    
        #updating batch status
        batch_status=dataset.loc[lower_i:upper_i-1,'task_status'].value_counts()
        #reporting them back
        print('Status of batch '+str(j+1)+':')
        print('completed images= '+str(batch_status.get('COMPLETED',0)))
        print('failed images= '+str(batch_status.get('FAILED',0)))
        print('pending images= '+str(upper_i-lower_i -batch_status.get('COMPLETED',0)-batch_status.get('FAILED',0)))
        print('------------------')
        
    #updating dataset after every batch
    dataset.to_csv('dataset.csv',index=False)


#messages when the code is done
telegram_text('The Landsat download code has finished')
print('The Landsat download code has finished')
