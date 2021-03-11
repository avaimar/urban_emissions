import argparse
import h5py
import numpy as np
import pandas as pd
import pickle
import os

import extract_features
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
    split_key_dict = None # TODO need to save and load this

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
        concat_key = split_key_dict['concat_{}'.format(split)]

        # Helper counters
        processed_counter = 0

        # Grab indexes to loop over
        indexes = range(concat_X.shape[0])
        real_length = len(indexes)
        if split == 'train':
            np.random.seed(42)
            indexes = np.random.randint(
                0, high=concat_X.shape[0],
                size=int(concat_X.shape[0] * train_subset_percent_))
            real_length = indexes.shape[0]

        # Loop over each data point in indexes
        for i in indexes:
            # Get uid
            # TODO BELOW HERE

            # Get feature vector and label
            uid = str_key['Unique_ID'][i]
            sat_idx = sat_key.loc[sat_key['Unique_ID'] == uid, 'value'].index.tolist()[0]

            # Get str image, label
            str_img = str_X[i, :, :, :]
            str_y = str_Y[i]

            # Process images to be input to model
            proc_sat_img = extract_features.process_sat_image(sat_img, sat_transforms['dev'])
            proc_str_img = extract_features.process_str_image(str_img, str_transforms['dev'])

            # Get model output (up to next to last linear layer)
            SatModel.eval()
            SatModel = SatModel.double()
            SatModel.model.final_layers[0].register_forward_hook(sat_linear_hook)
            sat_yhat = SatModel(proc_sat_img[None, ...])
            sat_feat = feat_dict['sat']

            # Concatenate and save features and labels to feature split file
            concat_feat = np.concatenate((sat_feat, str_feat), axis=1).reshape(feat_size, )
            feat_db_dict[split]['X'][processed_counter] = concat_feat
            feat_db_dict[split]['Y'][processed_counter] = sat_y

            # Print information on progress
            if processed_counter % 1000 == 0:
                print('[INFO] Processing {} split; image {}/{}'.format(
                    split, processed_counter, real_length))
            processed_counter += 1

        # Close dataset
        concat_db.close()

        # Close feature dataset
        pred_db_dict[split].close()
