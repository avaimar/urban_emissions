import h5py
import matplotlib.pyplot as plt
import pickle


# Load Unique_ID table
file = '01_Data/02_Sat_Images/imagery_no_mask_datapoints.pkl'
with open(file, 'rb') as f:
    UID_table = pickle.load(f)

# Load images
image_file = h5py.File('01_Data/02_Sat_images/imagery_no_mask_comp.h5', 'r')
images = image_file['imagery_no_mask']

# Get indexes for Rangoon and Mojave
rangoon_UID = 'S_104MMR010001'
mojave_UID = 'S_060711001'
rangoon_idx = UID_table.index[UID_table['Unique_ID'] == rangoon_UID].tolist()[0]
mojave_idx = UID_table.index[UID_table['Unique_ID'] == mojave_UID].tolist()[0]

# Grab images
rangoon_img = images[:, :, :, rangoon_idx]
mojave_img = images[:, :, :, mojave_idx]

# Limit images to first three channels
rangoon_img = rangoon_img[:, :, 0:3]
mojave_img = mojave_img[:, :, 0:3]

# Clip to range [0, 10000]
rangoon_img /= 10000
mojave_img /= 10000

# Render images
plt.imshow(rangoon_img)
plt.show()

plt.imshow(mojave_img)
plt.show()

# Close image database
image_file.close()
