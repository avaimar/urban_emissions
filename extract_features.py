import argparse
import h5py
import pandas as pd
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
    train_key = split_info[split_info['dataset'] == 'train']
    dev_key = split_info[split_info['dataset'] == 'val']
    test_key = split_info[split_info['dataset'] == 'test']

    return train_key, dev_key, test_key


def load_street_split_info(street_split_dir):
    """

    :param street_split_dir: (str)
    :return:
    """
    street_train_pkl, street_dev_pkl, street_test_pkl = None, None, None
    for split in ['train', 'dev', 'test']:
        split_file = os.path.join(street_split_dir, '{}')
        try:
            with open(split_file, 'rb') as file:
                pass # TODO
        except FileNotFoundError:
            print('[ERROR] Street pkl file not found for {}'.format(split))
    return street_train_pkl, street_dev_pkl, street_test_pkl


def create_hdf5_feature_datasets(path, split_sizes, feat_size):
    """

    :param path:
    :param split_sizes:
    :param feat_size:
    :return:
    """
    train_db = h5py.File(path.format('train'), "w")
    train_db.create_dataset(
        name='X', shape=(split_sizes['train'], feat_size), dtype='f')
    train_db.create_dataset(
        name='Y', shape=(split_sizes['train'], 1), dtype='f')

    val_db = h5py.File(path.format('dev'), "w")
    val_db.create_dataset(
        name='X', shape=(split_sizes['val'], feat_size), dtype='f')
    val_db.create_dataset(
        name='Y', shape=(split_sizes['val'], 1), dtype='f')

    test_db = h5py.File(path.format('test'), "w")
    test_db.create_dataset(
        name='X', shape=(split_sizes['test'], feat_size), dtype='f')
    test_db.create_dataset(name='Y', shape=(split_sizes['test'], 1), dtype='f')
    return train_db, val_db, test_db




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
    assert(sat_params['output_variable'] == street_params['output_variable'])
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

    # Load images
    #sat_train, sat_dev, sat_test = load_sat_imagery(image_dir)
    #street_train, street_dev, street_test = load_street_imagery(image_dir)

    # Create datasets
    datasets_path = os.path.join(feature_dir, 'concat_{}.hdf5')
    split_sizes = {'train': street_train_key.shape[0],
                   'dev': street_dev_key.shape[0],
                   'test': street_test_key.shape[0]}
    # * Note: We use the number of features in the next to last linear layer
    feat_size = SatModel.model.final_layers[0].out_features + \
                StreetModel.model.final_layers[0].out_features
    print('[INFO] Extracting {} features per image'.format(feat_size))
    train_db, val_db, test_db = create_hdf5_feature_datasets(
        datasets_path, split_sizes, feat_size)

    # Extract features for each location
    for split in ['test', 'dev', 'test']:
        # Load image datasets
        sat_db = None # TODO
        street_db = None

        #for location in locations:
            # Get model output (up to next to last linear layer)
            # TODO
            # Save to t/d/t file

        # Close datasets

