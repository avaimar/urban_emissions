import argparse
import matplotlib.pyplot as plt
import os

import utils
import Models.CNNs


# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--model_weights', required=True,
                    help='Path to model weights')
parser.add_argument('-p', '--model_params', required=True,
                    help='Path to model parameter dictionary')
parser.add_argument('-o', '--vis_output', required=True,
                    help='Directory to output visualizations')


if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    model_weights_file = args['model_weights']
    params_file = args['model_params']
    output_dir = args['vis_output']

    # Load params dictionary
    try:
        params = utils.load_dict(params_file)
    except FileNotFoundError:
        print("[ERROR] Parameter files not found.")

    # Define number of image channels
    if params['model_type'] == 'sat':
        num_channels = 7
    elif params['model_type'] == 'street':
        num_channels = 3
    else:
        raise Exception('[ERROR] Visualization only available for sat or street.')

    # Load selected model
    if 'AQI' not in params['output_variable']:
        Model = Models.CNNs.ResNetRegression(
            no_channels=num_channels, p=params['p_dropout'],
            add_block=params['extra_DO_layer'], num_frozen=params['num_frozen'])
    else:
        raise Exception('[ERROR] AQI not yet implemented.')

    # Load trained weights
    print('[INFO] Loading pretrained weights for selected models.')
    utils.load_checkpoint(model_weights_file, Model, optimizer=None)

    # Visualize filters
    # TODO

    # Save output
    # TODO
