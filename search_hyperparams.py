"""
Conduct hyperparameter searches over selected ranges.
"""
import argparse
import numpy as np
import os
import random
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', default='01_Data/02_Imagery',
                    required=True, help='Directory containing the dataset')
parser.add_argument('-o', '--model_output', default='03_Trained_Models/SatCnn',
                    required=True,
                    help='Directory to output training files for each model')
parser.add_argument('-t', '--model_tag',
                    required=True, help='Name of model directory')
parser.add_argument('-p', '--parameter', required=True,
                    help='Name of parameter to be searched over')

# Set up parameter defaults
params = {
    'model_type': "sat",
    'learning_rate': 1e-3,
    'batch_size': 64,
    'num_epochs': 500,
    'save_summary_steps': 100,
    'num_workers': 0,
    'output_variable': 'OZONE',
    'base_sat_image_file': "01_data/02_sat_images/imagery_no_mask_comp.h5",
    'base_sat_id_file': "01_data/02_sat_images/imagery_no_mask_datapoints.pkl",
    'base_sat_labels_file': "01_data/01_carbon_emissions/AirNow/World_all_locations_2020_avg_clean.csv",
    'sat_data_split': [0.85, 0.075, 0.075],
    'optimizer': 'SGD',
    'validation_metric': 'RMSE',
    'p_dropout': 0.5,
    'extra_DO_layer': 0,
    'num_frozen': 62,
    'restore_file': "",
    'subset_percent': 1
}

# Set up parameter searches
# Learning rate
random.seed(42)
r = -4 * np.random.rand(5)
alpha = 10**r
learning_rate = list(alpha)

# Batch size
batch_size = [32, 64, 128]

# Optimizers
optimizer = ['SGD', 'Adam', 'RMSprop', 'Adagrad']


if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    data_directory = args['data_directory']
    model_output = args['model_output']
    model_tag = args['model_tag']
    model_param = args['parameter']

    # Grab values for selected hyperparameter
    try:
        hp_list = globals()[model_param]
    except KeyError:
        print('No parameter list defined for selected parameter.')

    # Loop over each hyperparameter value
    print('[INFO] Running models for {} search'.format(model_param))
    for hp in hp_list:
        print('[INFO] Running model for {}: {}'.format(model_param, str(hp)))

        # Set up directory
        model_path = os.path.join(model_output, model_tag + '_' + str(hp))
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Modify params dictionary and save
        dict_path = os.path.join(model_path, 'params.json')
        params_dict = params
        params_dict[model_param] = hp
        utils.save_dict(params_dict, dict_path)

        # Execute for each model
        main_command = '{python} train.py -d {data_directory} -o {model_output} -m {model_parameters}'
        cmd = main_command.format(
            python=PYTHON, data_directory= data_directory,
            model_output=model_path, model_parameters=dict_path)
        print(cmd)
        check_call(cmd, shell=True)

print('[INFO] Hyperparameter search completed.')
