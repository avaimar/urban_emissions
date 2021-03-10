import argparse
import h5py
import numpy as np
import pandas as pd
import pickle
import os

import Models.data_loaders as data_loaders
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
parser.add_argument('-s', '--split_dir', required=True,
                    help='Directory containing sat and street image splits and keys')


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
        name='X', shape=(split_sizes_dict['dev'], feat_size_num), dtype='f')
    val_db.create_dataset(
        name='Y', shape=(split_sizes_dict['dev'], 1), dtype='f')

    test_db = h5py.File(path.format('test'), "w")
    test_db.create_dataset(
        name='X', shape=(split_sizes_dict['test'], feat_size_num), dtype='f')
    test_db.create_dataset(name='Y', shape=(split_sizes_dict['test'], 1), dtype='f')
    return {'train': train_db, 'dev': val_db, 'test': test_db}


feat_dict = {}


def sat_linear_hook(model, input, output):
    feat_dict['sat'] = output.detach()


def str_linear_hook(model, input, output):
    feat_dict['str'] = output.detach()


def process_sat_image(sat_image, transform):
    """
    Prepare satellite image to be fed to model.
    :param sat_image: (np.array) in the format (H, W, C)
    :param transform: (torchvisions.transform)
    :return: image (torch.tensor)
    """
    # Images are (H, W, C) as expected by Torch

    # Normalize image (valid ranges for bands are [0, 10,000])
    X_item = sat_image / 10000.

    # Apply transforms
    X_item = transform(X_item)

    return X_item


def process_str_image(str_image, transform):
    """
    Prepare street image to be fed to model.
    :param str_image: (np.array) in the format (H, W, C)
    :param transform: (torchvisions.transform)
    :return: image (torch.tensor)
    """
    return transform(str_image)


if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    sat_model_file = args['sat_model']
    sat_params_file = args['sat_model_params']
    street_model_file = args['street_model']
    street_params_file = args['street_model_params']
    feature_dir = args['feature_dir']
    split_dir = args['split_dir']

    # Load params dictionaries
    try:
        sat_params = utils.load_dict(sat_params_file)
        street_params = utils.load_dict(street_params_file)
    except FileNotFoundError:
        print("[ERROR] Parameter files not found.")

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
    split_key_dict = load_split_info(split_dir)

    # Create datasets
    datasets_path = os.path.join(feature_dir, 'concat_{}.hdf5')
    split_sizes = {'train': split_key_dict['street_train'].shape[0],
                   'dev': split_key_dict['street_dev'].shape[0],
                   'test': split_key_dict['street_test'].shape[0]}
    # * Note: We use the number of features in the next to last linear layer
    feat_size = SatModel.model.final_layers[0].out_features + \
                StreetModel.model.final_layers[0].out_features
    print('[INFO] Extracting {} features per image'.format(feat_size))
    feat_db_dict = create_hdf5_feature_datasets(datasets_path, split_sizes, feat_size)

    # Gather transforms
    sat_band_means = utils.load_dict(os.path.join(split_dir, 'band_means.json'))
    sat_band_sds = utils.load_dict(os.path.join(split_dir, 'band_sds.json'))
    sat_transforms = data_loaders.define_data_transforms(
        'sat', sat_band_means, sat_band_sds)
    str_transforms = data_loaders.define_data_transforms('street', None, None)

    # Extract features for each location
    for split in ['train', 'dev', 'test']:
        print('[INFO] Extracting features for {} split'.format(split))

        # Load image datasets for the split
        sat_db = h5py.File(os.path.join(split_dir, 'sat_{}.hdf5'.format(split)), 'r')
        str_db = h5py.File(os.path.join(split_dir, 'street_{}.hdf5'.format(split)), 'r')

        # Grab images and labels
        sat_X, sat_Y = sat_db['X'], sat_db['Y']
        str_X, str_Y = str_db['X'], str_db['Y']

        # Grab keys
        sat_key = split_key_dict['sat_{}'.format(split)]
        str_key = split_key_dict['street_{}'.format(split)]

        # Helper counters
        cur_sat_idx = -1

        # Grab indexes to loop over
        indexes = range(str_key.shape[0])
        if split == 'train':
            np.random.seed(42)
            indexes = np.random.randint(
                0, high=str_key.shape[0],
                size=int(str_key.shape[0] * 0.5))

        # Loop over each (sat image, str image) pair in indexes
        for i in indexes:
            # Get sat image, label (we only update if the index has changed as
            # we will mostly grab the same sat image ~10 times)
            uid = str_key['Unique_ID'][i]
            sat_idx = sat_key.loc[sat_key['Unique_ID'] == uid, 'value'].index.tolist()[0]
            if sat_idx != cur_sat_idx:
                sat_img = sat_X[sat_idx, :, :, :]
                sat_y = sat_Y[sat_idx]
                cur_sat_idx = sat_idx

            # Get str image, label
            str_img = str_X[i, :, :, :]
            str_y = str_Y[i]

            # Verify that sat and street target values y are close
            if abs(sat_y - str_y) > 1e-4:
                print('[WARNING] Target values for UID {} differ. Sat: {}; Street {}'.format(
                    uid, sat_y, str_y))

            # Process images to be input to model (we use 'dev' transforms as
            # we don't want any modifications to the input image, just the
            # processing to feed it to the model)
            proc_sat_img = process_sat_image(sat_img, sat_transforms['dev'])
            proc_str_img = process_str_image(str_img, str_transforms['dev'])

            # Get model output (up to next to last linear layer)
            SatModel.eval()
            SatModel = SatModel.double()
            SatModel.model.final_layers[0].register_forward_hook(sat_linear_hook)
            sat_yhat = SatModel(proc_sat_img[None, ...])
            sat_feat = feat_dict['sat']

            StreetModel.eval()
            SatModel = SatModel.double()
            StreetModel.model.final_layers[0].register_forward_hook(str_linear_hook)
            str_yhat = StreetModel(proc_str_img[None, ...])
            str_feat = feat_dict['str']

            # Concatenate and save features and labels to feature split file
            concat_feat = np.concatenate((sat_feat, str_feat), axis=1).reshape(feat_size, )
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
