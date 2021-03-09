import h5py
import numpy as np
import pickle


# Load the image and label data
data = h5py.File('../../01_Data/03_Processed_Data/OZONE/street_train.h5', 'r')

# Load the Unique IDs
labels = pickle.load(open('../../01_Data/03_Processed_Data/OZONE/gsv_train_labels.pkl', 'rb'))

print('loaded datasets')

# Set up an array to save the subset of images
img_subset = []

# Set up an array to save the subset of labels
label_subset = []

# Set up an array to save the corresponding Unique_IDs
id_subset = []

# Get image_index
labels['image_index'] = labels.index

a = labels.groupby(['Unique_ID']).head(3)
num_rows = len(a)
count = 0
for i, row in a.iterrows():
    if (count % 100 == 0):
        print('on count ', count, ' of ', num_rows)
    img_indx = row['image_index']
    img = data['gsv_train_images'][:, :, :, img_indx]
    label = data['Y'][img_indx]
    uid = row['Unique_ID']

    img_subset.append(img)
    label_subset.append(label)
    id_subset.append(uid)

    count += 1

print('writing datasets')
out_file = h5py.File('street_train_subset.h5', 'a')
out_file.create_dataset('X', data=img_subset, compression='gzip')
out_file.create_dataset('Y', data=label_subset, compression='gzip')
out_file.create_dataset('Unique_ID', data=id_subset, compression='gzip')
out_file.close()
