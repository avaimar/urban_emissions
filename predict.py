import argparse
import h5py
import numpy as np
import pandas as pd
import os
import torch

import Models.data_loaders as data_loaders
import Models.NNs
import utils


# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--concat_model', required=True,
                    help='Path to concat model weights')
parser.add_argument('-cp', '--concat_model_params', required=True,
                    help='Path to concat json params dict')
parser.add_argument('-f', '--feature_dir', required=True,
                    help='Directory containing extracted features')


def create_prediction_datasets():
    """
    Creates the datasets to store the predictions and true labels.
    :return: a dict of the 3 databases
    """
    train_db = pd.DataFrame(columns=['Unique_ID', 'label', 'prediction'])
    dev_db = pd.DataFrame(columns=['Unique_ID', 'label', 'prediction'])
    test_db = pd.DataFrame(columns=['Unique_ID', 'label', 'prediction'])
    return {'train': train_db, 'dev': dev_db, 'test': test_db}


def load_concat_key(feature_dir_):
    """
    Load the concat key files containing the Unique IDs in order
    :param feature_dir_: (str) directory where the files are located
    :return: (dict) of 3 databases, one for each split
    """
    split_dict_ = {}
    for cur_split in ['train', 'dev', 'test']:
        cur_split_file = os.path.join(
            feature_dir_, 'concat_key_{}.csv'.format(cur_split))
        try:
            split_dict_[cur_split] = pd.read_csv(cur_split_file)
        except FileNotFoundError:
            print('[ERROR] Concat Key file not found for {}'.format(cur_split))
    return split_dict_


def process_feature(x_feat):
    """
    Preprocess a feature vector to pass to ConcatNN.
    :return: (torch.tensor)
    """
    X_item = np.asarray(x_feat)

    # Normalize X
    X_item = (X_item - data_loaders.CONCAT_MEAN) / data_loaders.CONCAT_STD

    # Transform to torch tensor
    X_item = torch.from_numpy(X_item)

    return X_item


if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    concat_model_file = args['concat_model']
    concat_params_file = args['concat_model_params']
    feature_dir = args['feature_dir']

    # Load params dictionaries
    try:
        concat_params = utils.load_dict(concat_params_file)
    except FileNotFoundError:
        print("[ERROR] Parameter file not found.")

    # Load ConcatNN
    if 'AQI' not in concat_params['output_variable']:
        print('[INFO] Loading ConcatNet with 1,024 features')
        ConcatModel = Models.NNs.ConcatNet(feat_size=1024, out_size=1)
    else:
        raise Exception('[ERROR] AQI not yet implemented.')

    # Load trained weights
    print('[INFO] Loading pretrained weights for concat model.')
    utils.load_checkpoint(concat_model_file, ConcatModel, optimizer=None)

    # Load split information
    split_key_dict = load_concat_key(feature_dir)

    # Create datasets
    pred_db_dict = create_prediction_datasets()

    # Define percentage of training examples to predict
    train_subset_percent_ = 1

    # Predict
    for split in ['dev', 'test', 'train']:
        print('[INFO] Predicting for {} split'.format(split))

        # Load features for the split
        concat_db = h5py.File(os.path.join(
            feature_dir, 'concat_{}.hdf5'.format(split)), 'r')

        # Grab features and labels
        concat_X, concat_Y = concat_db['X'], concat_db['Y']

        # Grab keys
        concat_key = split_key_dict[split]

        # Helper counters
        processed_counter = 0

        # Grab indexes to loop over
        indexes = range(concat_X.shape[0])
        real_length = len(indexes)
        if split == 'train' and train_subset_percent_ < 1:
            np.random.seed(42)
            indexes = np.random.randint(
                0, high=concat_X.shape[0],
                size=int(concat_X.shape[0] * train_subset_percent_))
            real_length = indexes.shape[0]

        # Loop over each data point in indexes
        for i in indexes:
            # Get uid
            uid = concat_key['Unique_ID'][i]

            # Get feature vector and label
            feat_x = concat_X[i, :]
            feat_y = concat_Y[i].item()

            # Process features to be input to ConcatNN
            proc_feat_x = process_feature(feat_x)

            # Get model output (up to next to last linear layer)
            ConcatModel.eval()
            ConcatModel = ConcatModel.float()
            feat_yhat = ConcatModel(proc_feat_x[None, ...]).item()

            # Save uid, label and prediction
            pred_db_dict[split] = pred_db_dict[split].append(
                {'Unique_ID': uid, 'label': feat_y, 'prediction': feat_yhat})

            # Print information on progress
            if processed_counter % 1000 == 0:
                print('[INFO] Processing {} split; image {}/{}'.format(
                    split, processed_counter, real_length))
            processed_counter += 1

        # Close dataset
        concat_db.close()

        # Save feature dataset
        pred_db_dict[split].to_csv(
            os.path.join(feature_dir, 'predict_{}.csv'.format(split)),
            index=False)
