import pickle
import h5py
import pandas as pd
import numpy as np

# Get the Labels data and filter the ozone measurements
emissions_data = pd.read_csv('World_all_locations_2020_avg_clean.csv')
emissions_data = emissions_data[emissions_data['type'] == 'OZONE']

# Set up label output (for .pkl)
labels = pd.DataFrame(columns=['Unique_ID', 'lat', 'lon', 'image_index'])
label_indx = 0

# Set up image output (for .h5)
imgs = []
img_index = 0

# Loop over each chunk
for i in range(10):
    print('loading chunk ', i)
    filename = 'test/test_chunk%d.pkl' %(i)
    data = pickle.load(open(filename, 'rb'))
    print('\tfinished loading pkl file')
    n = len(data)
    
    # loop over each row in chunk
    for indx, row in data.iterrows():
        if (indx % 100 == 0):
            print('\ton row ', indx, 'of ', n-1)
        unique_id = row['Unique_ID']
        lat, lon = row['lat'], row['lon']
        img = row['img']
        
        if (img is not None):
            
            # add image to imgs
            imgs.append(img)
        
            # add label data to labels
            labels.loc[label_indx] = [unique_id, lat, lon, img_index]
            img_index += 1
            label_indx += 1
    print('\tdone with chunk ', i, '. len of images is ', len(imgs), ' and len of labels is ', len(labels))

labels.to_pickle('street_test.pkl')
h5f = h5py.File('street_test.h5', 'w')
h5f.create_dataset('X', data=imgs, compression='gzip')
h5f.close()

# Check that the files loaded correctly
h5f = h5py.File('street_dev.h5','r')
print(h5f.keys())
print(len(h5f['X']))
h5f.close()

a = pickle.load(open('street_dev.pkl', 'rb'))
print(a)