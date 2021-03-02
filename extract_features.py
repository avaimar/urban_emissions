import argparse
import h5py

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

if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    sat_model_file = args['sat_model']
    sat_params = args['sat_model_params']
    street_model_file = args['street_model']
    street_params = args['street_model_params']
    feature_dir = args['feature_dir']

    # Load Sat and Street Models
    assert(sat_params['output_variable'] == street_params['output_variable'])
    if 'AQI' not in sat_params['output_variable']:
        SatModel = Models.CNNs.ResNetRegression(
            no_channels=no_channels, p=sat_params['p_dropout'],
            add_block=sat_params['extra_DO_layer'],
            num_frozen=sat_params['num_frozen'])
        StreetModel = Models.CNNs.ResNetRegression(
            no_channels=no_channels, p=street_params['p_dropout'],
            add_block=street_params['extra_DO_layer'],
            num_frozen=street_params['num_frozen'])
    else:
        raise Exception('[ERROR] AQI not yet implemented.')

    # Load trained weights
    utils.load_checkpoint(sat_model_file, SatModel, optimizer=None)
    utils.load_checkpoint(sat_model_file, StreetModel, optimizer=None)

    # Extract features for each location


    # Save features
