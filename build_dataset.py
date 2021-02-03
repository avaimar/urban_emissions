import numpy as np
import os
import pickle


def process_sat_data(base_data_file, output_files, output_variable, data_split):
    """
    Pre-processes Satellite Data and creates train/val/test splits.
    :param base_data_file: (str)
    :param output_files: (str)
    :param output_variable: (str)
    :param data_split: (list)
    :return:
    """
    # Load file
    try:
        data = pickle.load(open(base_data_file, 'rb'))
    except FileNotFoundError:
        print('[ERROR] Dataset not found.')

    # Filter for selected output variable
    data = data[data['type'] == output_variable]

    # Get dataset size
    m = len(data)

    # Get features and labels
    X = np.array(data['imagery'].to_list())

    # Distinguish between preprocessing for classification and regression
    if "AQI" in output_variable:
        Y = data['AQI_level'].to_numpy().reshape(m, 1)
    else:
        Y = data['value'].to_numpy().reshape(m, 1)

    # Create train/test/split
    set.seed(42)
    assert sum(data_split) == m
    # TODO train/test/split

    # Save each split
    for split in ['train', 'val', 'test']:
        path = os.path.join(output_files, '{}_{}_split'.format(
            output_variable, split))
        # TODO write to file
