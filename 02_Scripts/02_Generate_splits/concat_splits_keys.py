import numpy as np
import pandas as pd
import pickle
import os


def load_split_info(split_dir_):
    """
    Reads in the csv files for each split of the sat and street datasets.
    :param split_dir_: (str) directory containing the split key files
    :return: a dictionary with the dataframes for each split type
    """
    split_dict_ = {}
    for dat_type in ['sat', 'street']:
        for cur_split in ['train', 'dev', 'test']:
            cur_split_file = os.path.join(
                split_dir_, '{}_{}_unique_IDs.csv'.format(dat_type, cur_split))
            cur_key = '{}_{}'.format(dat_type, cur_split)
            try:
                split_dict_[cur_key] = pd.read_csv(cur_split_file)
            except FileNotFoundError:
                print('[ERROR] Key file not found for {} {}'.format(
                    dat_type, cur_split))

    return split_dict_


# Setup
sat_model_file = '03_Trained_Models/SatCNN/Dropout_0.5/best.pth.tar'
sat_params_file = '03_Trained_Models/SatCNN/Dropout_0.5/params.json'
street_model_file = '03_Trained_Models/StreetCNN/Optimizer_Adam/best.pth.tar'
street_params_file = '03_Trained_Models/StreetCNN/Optimizer_Adam/params.json'
feature_dir = '01_Data/03_Processed_data/OZONE/Extracted_Features'
split_dir = '01_Data/03_Processed_data/OZONE'
key_path = '01_Data/03_Processed_data/OZONE/Extracted_Features/concat_key_{}.csv'

# Load split information
split_key_dict = load_split_info(split_dir)

# Create datasets
train_subset_percent_ = 0.5
split_sizes = {'train': split_key_dict['street_train'].shape[0],
               'dev': split_key_dict['street_dev'].shape[0],
               'test': split_key_dict['street_test'].shape[0]}
database_dict = {'train': pd.DataFrame(columns=['Unique_ID']),
                 'dev': pd.DataFrame(columns=['Unique_ID']),
                 'test': pd.DataFrame(columns=['Unique_ID'])}

# Extract features for each location
for split in ['train', 'dev', 'test']:
    # Grab key
    str_key = split_key_dict['street_{}'.format(split)]

    # Helper counter
    processed_counter = 0

    # Grab indexes to loop over
    indexes = range(str_key.shape[0])
    real_length = len(indexes)
    if split == 'train':
        np.random.seed(42)
        indexes = np.random.randint(
            0, high=str_key.shape[0],
            size=int(str_key.shape[0] * train_subset_percent_))
        real_length = indexes.shape[0]

    # Loop over each (sat image, str image) pair in indexes
    for i in indexes:
        # Get uid and save
        uid = str_key['Unique_ID'][i]
        database_dict[split] = database_dict[split].append(
            {'Unique_ID': uid}, ignore_index=True)

        # Print information on progress
        if processed_counter % 1000 == 0:
            print('[INFO] Processing {} split; image {}/{}'.format(
                split, processed_counter, real_length))
        processed_counter += 1

    # Save dataset
    database_dict[split].to_csv(key_path.format(split))
