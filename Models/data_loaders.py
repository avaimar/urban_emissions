"""
Define
"""

import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader


class SatelliteData:
    """
    Define the Satellite Dataset.
    """

    def __init__(self, base_data_file, output_variable, split):
        """
        Store the data filtered by the selected output variable.
        :param base_data_file: (str) file path to dataset
        :param output_variable: (str) output variable
        """
        # Load file
        data_path = os.path.join(base_data_file, '{}_{}_split'.format(
            output_variable, split))
        try:
            with open(data_path, 'r') as file:
                data = file.read()
        except FileNotFoundError:
            print('[ERROR] Dataset not found.')

        # Get dataset size
        self.m = len(data)

        # Get features and labels
        self.X = np.array(data['imagery'].to_list())
        self.Y = np.array(data['label']).reshape(self.m, 1)

        # Get image information
        self.res = self.X.shape[1]
        self.num_channels = self.X.shape[3]

    def __len__(self):
        """
        Returns the size of the dataset
        :return: (int)
        """
        return self.m

    def __getitem__(self, item):
        """
        Returns a single datapoint given an index
        :param item: (int)
        :return: a tuple containing the image (res, res, num_bands) and label
        """
        return self.X[item, :], self.Y[item, :].item()


def fetch_dataloader(dataset_types, base_data_file, output_variable,  params):
    """
    Fetches the DataLoader object for each type of data.
    :param output_variable: (str) selected output variable
    :param base_data_file: (str) path to the dataset directory
    :param dataset_types: (list) list including ['train', 'val', 'test']
    :param params: (dict) a dictionary containing the model specifications
    :return: dataloaders (dict) a dictionary containing the DataLoader object
        for each type of data
    """

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in dataset_types:
            data = SatelliteData(base_data_file, output_variable, split)
            dl = 1  # TODO define dataloader
            dataloaders[split] = dl

    return dataloaders
