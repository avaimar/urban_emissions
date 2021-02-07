import glob
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import build_dataset
import utils


# Define data transforms
def define_data_transforms(training_band_means, training_band_sds):
    """
    Define the transforms to be applied to the data.
    :param training_band_sds: (dict) the sds for each band computed from the training
        set
    :param training_band_means: (dict) the means for each band computed from the training
        set
    :return: (dict) of transforms for each type of dataset
    """
    # ImageNet means and sds
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    image_net_means = [0.485, 0.456, 0.406]
    image_net_sds = [0.229, 0.224, 0.225]

    # Get means and sds for our training set
    training_band_means = [training_band_means['band_{}'.format(i)] for i in range(7)]
    training_band_sds = [training_band_sds['band_{}'.format(i)] for i in range(7)]

    # Use means and sds for channels 4 to 7 from our training set
    image_net_means.extend(training_band_means[3:])
    image_net_sds.extend(training_band_sds[3:])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ])
    }
    return data_transforms


# Satellite Data class for dataloaders
class SatelliteData(Dataset):
    """
    Define the Satellite Dataset.
    """

    def __init__(self, data_dir, output_variable, split, transform=None):
        """
        Store the data filtered by the selected output variable.
        :param data_dir: (str) path to split dataset locations
        :param output_variable: (str) output variable
        :param split: (str) one of ['train', 'val', 'test']
        :param transform (torchvision.transforms)
        """
        # Load files in path
        data_path = os.path.join(data_dir, output_variable, '{}/*'.format(split))
        data_points = glob.glob(data_path)
        self.data_points = data_points
        self.m = len(data_points)

        # Get transforms
        self.transform = transform

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
        # Grab data point and load
        try:
            data = np.load(self.data_points[item])
        except FileNotFoundError:
            print('[ERROR] Data point not found.')

        # Get features and labels
        X_item = data['X']
        Y_item = data['y']

        # Apply transforms
        if self.transform:
            # We must reorder the dimensions as torch assumes numpy images
            # are in the format (H,W,C)
            X_item = np.transpose(X_item, (1, 2, 0))
            X_item = self.transform(X_item)
        return X_item, Y_item


def fetch_dataloader(dataset_types, data_dir, output_variable, params,
                     base_data_file, data_split):
    """
    Fetches the DataLoader object for each type of data.
    :param dataset_types: (list) list including ['train', 'val', 'test']
    :param data_dir: (str) path to the split dataset directory
    :param output_variable: (str) selected output variable
    :param params: (dict) a dictionary containing the model specifications
    :param base_data_file: (str) Path to the file generated by the GGEarth
        script
    :param data_split: (list) containing the % of each split in the order
        [size_train, size_val, size_test]
    :return: dataloaders (dict) a dictionary containing the DataLoader object
        for each type of data
    """

    # Build datasets for selected output variable if they do not exist
    file_path = os.path.join(data_dir, output_variable)
    if not os.path.exists(file_path):
        print('[INFO] Building dataset...')
        os.mkdir(file_path)
        build_dataset.process_sat_data(
            base_data_file, data_dir, output_variable, data_split)

    # Use GPU if available
    use_cuda = torch.cuda.is_available()

    # Get mean and sd dictionaries from our training set and define transforms
    training_band_means = utils.load_dict(
        os.path.join(data_dir, output_variable, 'band_means.json'))
    training_band_sds = utils.load_dict(
        os.path.join(data_dir, output_variable, 'band_sds.json'))
    transforms_dict = define_data_transforms(training_band_means, training_band_sds)

    # Get data loaders
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in dataset_types:
            data = SatelliteData(
                data_dir, output_variable, split, transform=transforms_dict[split])
            dl = DataLoader(
                dataset=data, batch_size=params['batch_size'], shuffle=True,
                num_workers=params['num_workers'], pin_memory=use_cuda)
            dataloaders[split] = dl

    return dataloaders
