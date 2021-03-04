import argparse
import h5py
import pandas as pd
import pickle
import os

import Models.CNNs
import utils

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-sat', '--sat_model', required=True,
                    help='Path to satellite model weights')
parser.add_argument('-satp', '--sat_model_params', required=True,
                    help='Path to satellite json params dict')
parser.add_argument('-str', '--street_model', required=True,
                    help='Path to street model weights')
parser.add_argument('-strp', '--street_model_params', required=True,
                    help='Path to street json params dict')
parser.add_argument('-d', '--feature_dir', required=True,
                    help='Directory to save extracted features')
parser.add_argument('-s', '--split_file', required=True,
                    help='Path to csv indicating T/D/T splits')
parser.add_argument('-i', '--image_dir', required=True,
                    help='Directory containing sat and street image splits')


def generate_split_keys(split_file_path):
    """
    Generates 3 data frames with the unique IDs pertaining to train, dev and
    test sets.
    :param split_file_path: (str) the path to the split .csv file
    :return: tuple of 3 DataFrames containing the Unique_ID, dataset columns
    """
    # Load splits
    try:
        split_info = pd.read_csv(split_file_path)
    except FileNotFoundError:
        print('[ERROR] Split file not found.')

    # Partition into train/dev/test
    train_key = split_info[split_info['dataset'] == 'train'].copy()
    train_key['img_idx'] = range(train_key.shape[0])

    dev_key = split_info[split_info['dataset'] == 'val'].copy()
    dev_key['img_idx'] = range(dev_key.shape[0])

    test_key = split_info[split_info['dataset'] == 'test'].copy()
    test_key['img_idx'] = range(test_key.shape[0])

    return train_key, dev_key, test_key


def load_street_split_info(street_split_dir):
    """
    Reads in the pickle files for each split of the street level dataset.
    :param street_split_dir: (str) directory containing these pickle files
    :return: a tuple with the data frames for each split
    """
    str_split_dict = {}
    for cur_split in ['train', 'dev', 'test']:
        cur_split_file = os.path.join(
            street_split_dir, 'street_{}.pkl'.format(cur_split))
        try:
            with open(cur_split_file, 'rb') as file:
                str_split_dict[cur_split] = pickle.load(file)
        except FileNotFoundError:
            print('[ERROR] Street pkl file not found for {}'.format(cur_split))

    return str_split_dict['train'], str_split_dict['dev'], str_split_dict['test']


def create_hdf5_feature_datasets(path, split_sizes_dict, feat_size_num):
    """
    Creates the HDF5 datasets to store the exctrated features.
    :param path: (str) Directory where the datasets should be stored
    :param split_sizes_dict: (dict) The number of data points in each split
    :param feat_size_num: (int) The length of the extracted feature vectors
    :return: a dict of the 3 databases, each containing an X and Y dataset
    """
    train_db = h5py.File(path.format('train'), "w")
    train_db.create_dataset(
        name='X', shape=(split_sizes_dict['train'], feat_size_num), dtype='f')
    train_db.create_dataset(
        name='Y', shape=(split_sizes_dict['train'], 1), dtype='f')

    val_db = h5py.File(path.format('dev'), "w")
    val_db.create_dataset(
        name='X', shape=(split_sizes_dict['val'], feat_size_num), dtype='f')
    val_db.create_dataset(
        name='Y', shape=(split_sizes_dict['val'], 1), dtype='f')

    test_db = h5py.File(path.format('test'), "w")
    test_db.create_dataset(
        name='X', shape=(split_sizes_dict['test'], feat_size_num), dtype='f')
    test_db.create_dataset(name='Y', shape=(split_sizes_dict['test'], 1), dtype='f')
    return {'train': train_db, 'dev': val_db, 'test': test_db}


def get_linear_features(model, input, output):
    return output # TODO


if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    sat_model_file = args['sat_model']
    sat_params = args['sat_model_params']
    street_model_file = args['street_model']
    street_params = args['street_model_params']
    feature_dir = args['feature_dir']
    split_file = args['split_file']
    image_dir = args['image_dir']

    # Load Sat and Street Models
    assert (sat_params['output_variable'] == street_params['output_variable'])
    if 'AQI' not in sat_params['output_variable']:
        SatModel = Models.CNNs.ResNetRegression(
            no_channels=7, p=sat_params['p_dropout'],
            add_block=sat_params['extra_DO_layer'],
            num_frozen=sat_params['num_frozen'])
        StreetModel = Models.CNNs.ResNetRegression(
            no_channels=3, p=street_params['p_dropout'],
            add_block=street_params['extra_DO_layer'],
            num_frozen=street_params['num_frozen'])
    else:
        raise Exception('[ERROR] AQI not yet implemented.')

    # Load trained weights
    print('[INFO] Loading pretrained weights for satellite and street models.')
    utils.load_checkpoint(sat_model_file, SatModel, optimizer=None)
    utils.load_checkpoint(street_model_file, StreetModel, optimizer=None)

    # Load split information
    sat_train_key, sat_dev_key, sat_test_key = generate_split_keys(split_file)
    street_train_key, street_dev_key, street_test_key = load_street_split_info(
        image_dir)
    split_key_dict = {'sat_train': sat_train_key, 'sat_dev': sat_dev_key,
                      'sat_test': sat_test_key, 'str_train': street_train_key,
                      'str_dev': street_dev_key, 'str_test': street_test_key}

    # Create datasets
    datasets_path = os.path.join(feature_dir, 'concat_{}.hdf5')
    split_sizes = {'train': street_train_key.shape[0],
                   'dev': street_dev_key.shape[0],
                   'test': street_test_key.shape[0]}
    # * Note: We use the number of features in the next to last linear layer
    feat_size = SatModel.model.final_layers[0].out_features + \
                StreetModel.model.final_layers[0].out_features
    print('[INFO] Extracting {} features per image'.format(feat_size))
    feat_db_dict = create_hdf5_feature_datasets(datasets_path, split_sizes, feat_size)

    # Extract features for each location
    for split in ['test', 'dev', 'test']:
        print('[INFO] Extracting features for {} split'.format(split))

        # Load image datasets for the split
        sat_db = h5py.File(os.path.join(image_dir, 'sat_{}.hdf5'.format(split)), 'r')
        str_db = h5py.File(os.path.join(image_dir, 'street_{}.h5'.format(split)), 'r')

        # Grab images and labels
        sat_X, sat_Y = sat_db['X'], sat_db['Y']
        if split == 'train':
            str_X, str_Y = str_db['gsv_train_images'], str_db['Y']
        else:
            str_X, str_Y = str_db['X'], str_db['Y']

        # Grab keys
        sat_key = split_key_dict['sat_{}'.format(split)]
        str_key = split_key_dict['str_{}'.format(split)]

        # Helper counters
        cur_sat_idx = -1

        # Loop over each (sat image, str image) pair
        for i in range(str_key.shape[0]):
            # Get sat image, label (we only update if the index has changed as
            # we will mostly grab the same sat image ~10 times)
            uid = str_key['Unique_ID'][i]
            sat_idx = sat_key.loc[sat_key['Unique_ID'] == uid, 'img_idx'].iloc[0]
            if sat_idx != cur_sat_idx:
                sat_img = sat_X[sat_idx, :, :, :]
                sat_y = sat_Y[sat_idx].item()
                cur_sat_idx = sat_idx

            # Get str image, label
            if split == 'train':
                str_img = str_X[:, :, :, i]
            else:
                str_img = str_X[i, :, :, :]
            str_y = str_Y[i]

            # Verify that sat and street target values y are close
            if abs(sat_y - str_y) > 1e-4:
                print('[WARNING] Target values for UID {} differ. Sat: {}; Street {}'.format(
                    uid, sat_y, str_y))

            # Get model output (up to next to last linear layer)
            sat_feat = None # TODO
            str_feat = None # TODO
            concat_feat = None # TODO

            # Save features and labels to feature split file
            feat_db_dict[split]['X'][i] = concat_feat
            feat_db_dict[split]['Y'][i] = sat_y

            # Print information on progress
            if i % 1000 == 0:
                print('[INFO] Processing {} split; image {}/{}'.format(
                    split, i, str_key.shape[0]))

        # Close datasets
        sat_db.close()
        str_db.close()

        # Close feature dataset
        feat_db_dict[split].close()
