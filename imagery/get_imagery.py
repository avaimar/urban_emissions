#importing Earth Engine
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()

#importing Google Earth Engine Tools
import geetools 

#setting working directory
os.chdir(r'C:\Users\nsuar\Google Drive\Carbon_emissions')

#reading afrobarometer data to experiment
fake_data=pd.read_csv('data/afb_full_r6.csv')[['uniqueea','longitude','latitude']]
#dropping duplicates and keeping first 2,000 observations
fake_data=fake_data.drop_duplicates(subset='uniqueea')[:2000]

#starting to take images
#converting points to EE geometry (multipoints)

for i in range(10): #we are only taking 
    #generating a point for every longitude/latitude pair
    point=ee.Geometry.Point( fake_data.iloc[i, 1:3].tolist() ).buffer(500)
    #getting the images. Filter bounds pass the geographic points to the satellite image. Limit puts a limit to the number of images per pixel
    imagery=ee.Image("LANDSAT/LC08/C01/T1_SR").filter(ee.Filter.date('2019-01-01', '2019-05-01')).filter(ee.Filter.bounds(point))
    #locally saving the image
    tasks = geetools.batch.image.toLocal(
            image=imagery,
            path="data\fake_images",
            region=point,
            name='{sat}_{system_date}_{WRS_PATH:%d}-{WRS_ROW:%d}'
            )



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

