## problems scrapping satellite data

import numpy as np
import pandas as pd


#importing Earth Engine
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()


#converting a lon/lat pair from Menlo Park to an ee.point object
point=ee.Geometry.Point( [-122.2036486, 37.4237011] )
# converting the point to a patch: we define a circle with a 500m radius, and then we put a box around it
patch=point.buffer(500).bounds() 

#checking that the patch has approximately the right area 1000^2 m^2
patch.area(1).getInfo()

#defining the image: Landsat 8 collection of images from 2020, with RGB bands. We take the mean of them
imagery=ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate('2020-01-01', '2020-12-31').select(['B3','B4','B2']).mean()


#the problem here is that we lost the dimensions of the image. A quick and temporary fix is calling the fist image
#with imagery.getInfo() you should see a dimensions parameter in each band that should be 'dimensions': [9161, 9161], but it is not here

imagery2=ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate('2020-01-01', '2020-12-31').select(['B3','B4','B2']).first()
#here the dimensions are correct

#getting a matrix from the image for our patch of interest
rect_image = imagery2.sampleRectangle(region=patch)

#in theory, since this is 1000m*1000m and the resolution of the satellite is 30m,
#we should obtain a 33*33 matrix for each one of the 3 bands

#extracting 1 band to check
band_b4 = rect_image.get('B4')
#checking the shape of the resulting matrix
np.array(band_b4.getInfo()).shape

#we get an error. I tried unmasking (applying .unmask(-9999) to the imagery2 object), but it doesn't work.
#we still need to solve the issue with the dimensions of the images

#another suggested solution was to change the projection to UTM in our geometry
patch2=ee.Geometry.Polygon(patch.getInfo().get('coordinates'), proj='UTM')
rect_image2 = imagery2.sampleRectangle(region=patch2)

np.array(rect_image2.get('B4').getInfo()).shape

#but we have problems with the CRS this time
np_arr_b4.T


# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 22:05:26 2021

@author: nsuar
"""

## problems scrapping satellite data

import numpy as np
import pandas as pd


#importing Earth Engine
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()


#converting a lon/lat pair from Menlo Park to an ee.point object
point=ee.Geometry.Point( [-122.2036486, 37.4237011] )
# converting the point to a patch: we define a circle with a 500m radius, and then we put a box around it
patch=point.buffer(500).bounds() 

#checking that the patch has approximately the right area 1000^2 m^2
patch.area(1).getInfo()

#defining the image: Landsat 8 collection of images from 2020, with RGB bands. We take the mean of them
imagery=ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate('2020-01-01', '2020-12-31').select(['B3','B4','B2']).mean()
imagery=ee.Image(imagery)

imagery.getInfo()['bands'][1]['dimensions']

bounded_image=imagery.clip(patch)

bounded_image.getThumbURL({format: 'jpg'})

task=ee.batch.Export.image.toDrive(image=bounded_image, folder="Carbon_emissions_fake_images", 
                        description='LANDSTAT_img_test', 
                        region=patch)
#run the following to start exporting the file
task.start()

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 03:34:04 2021

@author: nsuar
"""

startDate = '2020-01-01'
endDate = '2020-12-01'

# London (n=325)
region = '[[48.5379,2.8416], [54.4768,2.8416], [48.5379,-3.0972], [54.4768,-3.0972]]'
rectangle = [48.5379,-3.0972,54.4768,2.8416]



# boston (n=325)
# region = '[[32.9542,-68.0490], [51.2126,-68.0490], [32.9542,-73.9878], [51.2126,-73.9878]]'
# rectangle = [32.9542,-73.9878,51.2126,-68.0490]

bands = ["B2","B3","B4"]

rectangle1 = ee.Geometry.Rectangle(rectangle)




dataset = ee.ImageCollection("LANDSAT/LC08/C01/T2")
dataset = dataset.filterBounds(rectangle1) # Filter region
dataset = dataset.filterDate(startDate, endDate).sort('system:time_start', True) # filter date
dataset = dataset.select(bands) # select RGB channels

count = dataset.size().getInfo()
print('num_images = ', count)
data = dataset.toList(count)

if (count == 0):
    print('No data found.')

else:
    print('Data found')
    dataset_avg = dataset.mean() # get the average over all images in the grid cell
    image = ee.Image(dataset_avg) # convert to an Image
    
    # Option 1: Get clickable download link
    print(image.getDownloadURL({'region': region})) 
    
    
    print(image.getThumbURL({'region': region, format: 'jpg', 'min':0, 'max':0.5, 'gamma': [0.95, 1.1, 1]}))