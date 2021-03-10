import h5py
import pandas as pd
import pickle

# Save street dev and test .csv keys
for split in ['dev', 'test']:
    path = '../../01_Data/03_Processed_data/OZONE/street_{}.pkl'
    save_path = '../../01_Data/03_Processed_data/OZONE/street_{}_unique_IDs.csv'

    # Read pkl file
    dat = pd.read_pickle(path.format(split))

    # Grab UID and label
    dat = dat[['Unique_ID', 'label']]

    # Rename label col to value
    dat.rename(columns={'label': 'value'}, inplace=True)

    # Save
    dat.to_csv(save_path.format(split), index=False)


# Save street train keys
str_train_db = h5py.File('../../01_Data/03_Processed_data/OZONE/street_train.hdf5')
str_labs = str_train_db['Y']

str_key = pd.read_csv('../../01_Data/03_Processed_data/OZONE/street_train_unique_IDs_raw_keys.csv')

# Check length
str_key.shape[0] == str_labs.shape[0]

# Concatenate columns and save
str_key['value'] = str_labs
str_key.drop(columns='img_number', inplace=True)
str_key.to_csv('../../01_Data/03_Processed_data/OZONE/street_train_unique_IDs.csv',
               index=False)
str_train_db.close()
