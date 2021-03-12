#importing packages
import requests
import numpy as np
from geetools import cloud_mask
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()

#defining parameters for the function
image_res=30
n_pixels=224

#getting satellite image collections
startDate = '2020-01-01'
endDate = '2020-12-31'
landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
# filter date
landsat = landsat.filterDate(startDate, endDate) 
#applying cloud masking
landsat_masked=landsat.map( cloud_mask.landsatSR(['cloud']) )
landsat_masked=landsat_masked.select(["B2","B3","B4"]).mean()
landsat=landsat.select(["B2","B3","B4"]).mean()

#visualization parameters
visParams={'min': 0, 'max': 3000, 'gamma': 1.4,  
           'bands' : ['B4', 'B3', 'B2'], 'dimensions' : str(n_pixels)+"x"+str(n_pixels),
           'format' : 'jpg'}


def visualization(lat,lon,name,mask=True):
    '''
    Function to visualize the images for our ML application.
    Inputs:
        -lat, lon: latitude and longitude coordinates
        -name: name that is going to be given to the jpg file
        -mask: True to get masked image, False to get unmasked image
    Outputs:
        The function doesn't produce an output, but generates a file called
        "name.jpg" in the current directory
    '''
    #computing point and bounding box
    point=ee.Geometry.Point(lon, lat)
    len=image_res*n_pixels # for landsat, 30 meters * 224 pixels
    region= point.buffer(len/2).bounds().getInfo()['coordinates']
    coords=np.array(region)
    coords=[np.min(coords[:,:,0]), np.min(coords[:,:,1]), np.max(coords[:,:,0]), np.max(coords[:,:,1])]
    rectangle=ee.Geometry.Rectangle(coords)
    
    #clipping the area from satellite image
    if mask==True:
        clipped_image= landsat_masked.clip(rectangle)
    else:
        clipped_image= landsat.clip(rectangle)
        
    #getting the image
    requests.get(clipped_image.getThumbUrl(visParams))
    open(name+'.jpg', 'wb').write(requests.get(clipped_image.getThumbUrl(visParams)).content)


#example    
visualization(16.8256,96.1445,'unmasked',mask=True)

    
    

