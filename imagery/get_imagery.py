import numpy as np
import pandas as pd
import os
import datetime



#importing Earth Engine
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()

#setting working directory
os.chdir(r'C:\Users\nsuar\Google Drive\Carbon_emissions')

#reading afrobarometer data to experiment
fake_data=pd.read_csv('data/afb_full_r6.csv')[['uniqueea','longitude','latitude']]
#dropping duplicates and keeping first 2,000 observations
fake_data=fake_data.drop_duplicates(subset='uniqueea')[:2000]

#starting to take images

# for i in range(10): #we are only taking 
#     #generating a point for every longitude/latitude pair
#     point=ee.Geometry.Point( fake_data.iloc[i, 1:3].tolist() ).buffer(500).bounds()
#     #getting the images. Filter bounds pass the geographic points to the satellite image. Limit puts a limit to the number of images per pixel
#     imagery=ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").filterBounds(point).filterDate('2020-01-01', '2020-12-31').select(['B3','B4','B2']).mean()
#     #locally saving the image
#     task=ee.batch.Export.image.toDrive(image=imagery, folder="Carbon_emissions\\data\\fake_images", 
#                             description='LANDSTAT_img_id'+str(int(fake_data.iloc[i,0])), 
#                             region=point,fileFormat='TFRecord')
#     task.start()
#     print("iteration"+str(i)+"done")

i=0
#converting first lon/lat pair to a point (with 500 meters augmented to each side)
point=ee.Geometry.Point( fake_data.iloc[i, 1:3].tolist() ).buffer(500).bounds()
#getting the images. Filter bounds pass the geographic points to the satellite image. We also filter the bands and the date, and then we take an average of the images
imagery=ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").filterBounds(point).filterDate('2020-01-01', '2020-12-31').select(['B3','B4','B2']).mean()
#saving the image to Drive (the only option)
task=ee.batch.Export.image.toDrive(image=imagery, folder="Carbon_emissions_fake_images", 
                        description='LANDSTAT_img_id'+str(int(fake_data.iloc[i,0])), 
                        region=point,fileFormat='TFRecord')
task.start()
#check status with task.status()



# #starting to take images
# #converting points to EE geometry (multipoints)
# lista=fake_data[:10][['longitude','latitude']].values.tolist() #taking only 10 to test

# points=ee.Geometry.MultiPoint(lista).buffer(500) #buffer sets a radio, in meters, in which every pixel is expanded

# #getting the images. Filter bounds pass the geographic points to the satellite image. Limit puts a limit to the number of images per pixel

# imagery=ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").filterBounds(points).filterDate('2019-01-01', '2019-05-01').limit(20)

# #collecting the images to Google Drive
# tasks = geetools.batch.Export.imagecollection.toDrive(
#             collection=imagery,
#             folder='Carbon_emissions_images',
#             region=points,
#             namePattern='{sat}_{system_date}_{WRS_PATH:%d}-{WRS_ROW:%d}',
#             dataType='uint32',
#             datePattern='ddMMMy',
#             extra=dict(sat='L8SR'),
#             verbose=True,
#             maxPixels=int(1e13)
#         )
