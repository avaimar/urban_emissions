# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 04:31:43 2021

@author: nsuar
"""

import numpy as np
import os
import imageio
import requests, zipfile, io

#importing Earth Engine
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()

#testing point in Menlo Park
point=ee.Geometry.Point( [-122.2036486, 37.4237011] )

#function to get a rectangle around a point
def point_box(point,len):
    #point is a ee.Geometry.Point object
    #len is the length of the box in meters
    #defining the square
    patch=point.buffer(len/2).bounds() 
    #retrieving all pixels for regions
    region= point.buffer(500).bounds().getInfo()['coordinates']
    #defining the rectangle
    coords=np.array(region)
    #taking min and maxs of coordinates to define the rectangle
    coords=[np.min(coords[:,:,0]), np.min(coords[:,:,1]), np.max(coords[:,:,0]), np.max(coords[:,:,1])]
    rectangle=ee.Geometry.Rectangle(coords)
    return str(region), rectangle
    
#testing the new function
region, rectangle = point_box(point,1000)

#img dimensions
def img_dim(image):
    #dimensions of the first band available
    try:
        print( 'image dimensions: '+str(image.getInfo()['bands'][0]['dimensions']) )
    except KeyError:
        print('image size not displaying, probably 1*1')

#testing
imagery=ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate('2020-01-01', '2020-12-31').select(['B3','B4','B2'])

#should not be 1*1
img_dim(imagery.first())
#should be 1*1
img_dim(imagery.mean())


def image_to_np(image,rectangle,region,directory):
    """
    function to download a clip of "region" from "image". We download a ZIP with all the bands,
    then we convert each band to a numpy array, and the we stack them togheter into a final
    numpy array.
    
    Inputs:
    -image= ee.Image object
    -rectangle= rectangle around our point from point_box function
    -region= string with a list of points defining a rectangle, obtained from the point_box function
    -directory= directory to store TIFF files
    
    Output:
    -np_image= numpy array with 3 dimensions, where the first 2 are the size of the patches, and the 3rd is the number of bands
    """    
    #deleting files in the target directory
    for file in os.listdir(directory):
        os.remove(directory+"\\"+file)
    
    
    #download the zip file containing the images, and clipping the image
    image = image.filterBounds(rectangle).first()
    r = requests.get(image.getDownloadURL({'region': region}))
    
    #unzip it to the selected directory
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(directory)
    
    #empty list to store the bands
    bandas=[]
    
    #iterating through all the bands we extracted
    for band in os.listdir(directory):
        #path of band TIFF file
        file_path=directory+"\\"+band
        #adding a numpy array with the band to a list
        bandas.append(np.array(imageio.imread(file_path)))
    
    #stacking list with bands to a 3-d array
    np_image=np.stack(bandas,axis=2)
    
    #printing dimensions
    print('output image size: '+str(np_image.shape))
    #returning output
    return np_image


#example of previous function
startDate = '2020-01-01'
endDate = '2020-12-01'
dataset = ee.ImageCollection("LANDSAT/LC08/C01/T2")
dataset = dataset.filterDate(startDate, endDate).sort('system:time_start', True) # filter date
dataset = dataset.select(["B2","B3","B4"]) # select RGB channels

directory=r'C:\Users\nsuar\Google Drive\Carbon_emissions\data\fake_images'

menlo_park_image=image_to_np(dataset,rectangle,region,directory)


#now testing with point in London
london = ee.Geometry.Point( [-0.1277583, 51.5073509] )
region, rectangle = point_box(london,1000)

london_image=image_to_np(dataset,rectangle,region,directory)