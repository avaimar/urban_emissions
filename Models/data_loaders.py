import glob
import h5py
import numpy as np
import pickle
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import build_dataset
import utils


# Define data transforms
def define_data_transforms(model_type, training_band_means, training_band_sds):
    """
    Define the transforms to be applied to the data.
    :param model_type: (str) one of {sat, street}
    :param training_band_sds: (dict) the sds for each band computed from the
       satellite image training set
    :param training_band_means: (dict) the means for each band computed from
       the satellite image training set
    :return: (dict) of transforms for each type of dataset
    """
    # ImageNet means and sds
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    image_net_means = [0.485, 0.456, 0.406]
    image_net_sds = [0.229, 0.224, 0.225]

    # Get means and sds for our training set if we're training satellite data
    if model_type == 'sat':
        training_band_means = [training_band_means['band_{}'.format(i)] for i in range(7)]
        training_band_sds = [training_band_sds['band_{}'.format(i)] for i in range(7)]

        # Use means and sds for channels 4 to 7 from our training set
        image_net_means.extend(training_band_means[3:])
        image_net_sds.extend(training_band_sds[3:])

    # Define transforms for each split
    data_transforms = {
        'dev': transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) to (C, H, W)
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) to (C, H, W)
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ])
    }

    # Select train transforms according to the dataset
    if model_type == 'sat':
        # This indicates we're training the satellite imagery
        data_transforms['train'] = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) to (C, H, W)
            transforms.RandomHorizontalFlip(),  # Requires (.., H, W)
            transforms.RandomVerticalFlip(),  # Requires (.., H, W)
            # transforms.RandomRotation(20), # Requires (.., H, W)
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds)),
        ])
    elif model_type == 'street':
        # This indicates we're training the street imagery
        data_transforms['train'] = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) to (C, H, W)
            transforms.RandomHorizontalFlip(),  # Requires (.., H, W)
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds)),
        ])
    else:
        raise Exception('[ERROR] Model type should be one of {sat, street}')

    return data_transforms


# Satellite Data class for DataLoaders
class SatelliteData(Dataset):
    """
    Define the Satellite Dataset.
    """

    def __init__(self, data_dir, output_variable, split, transform):
        """
        Store the data filtered by the selected output variable.
        :param data_dir: (str) path to split dataset locations
        :param output_variable: (str) output variable
        :param split: (str) one of ['train', 'dev', 'test']
        :param transform (torchvision.transforms)
        """
        # Open HDF5 dataset
        data_path = os.path.join(
            data_dir, output_variable, 'sat_{}.hdf5'.format(split))
        try:
            self.db = h5py.File(data_path, 'r')
        except FileNotFoundError:
            print('[ERROR] Dataset not found.')

        # Save image and label data
        self.image_data = self.db['X']
        self.label_data = self.db['Y']
        self.m = self.image_data.shape[0]
        self.output_variable = output_variable

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
        # Grab image and label
        X_item = np.asarray(self.image_data[item, :, :, :])
        Y_item = self.label_data[item]

        # Transpose image from (W, H, C) to (H, W, C) as expected by Torch
        X_item = np.transpose(X_item, (1, 0, 2))

        # Normalize image (valid ranges for bands are [0, 10,000])
        X_item = X_item / 10000.

        # Convert label to int in case of classification task
        if 'AQI' in self.output_variable:
            Y_item = int(Y_item)

        # Apply transforms
        if self.transform:
            X_item = self.transform(X_item)
        return X_item, Y_item

    def __del__(self):
        self.db.close()


# Street Data class for DataLoaders
class StreetData(Dataset):
    """
    Define the Street Dataset.
    """

    def __init__(self, data_dir, output_variable, split, transform):
        """
        Store the data filtered by the selected output variable.
        :param data_dir: (str) path to split dataset locations
        :param output_variable: (str) output variable
        :param split: (str) one of ['train', 'dev', 'test']
        :param transform (torchvision.transforms)
        """
        # Open HDF5 dataset
        data_path = os.path.join(data_dir, output_variable, 'street_{}.{}')
        try:
            self.db = h5py.File(data_path.format(split, 'h5'), 'r')
        except FileNotFoundError:
            print('[ERROR] Street {} image dataset not found.'.format(split))

        # Open label data
        try:
            with open(data_path.format(split, 'pkl'), 'rb') as labels_file:
                labels = pickle.load(labels_file)
        except FileNotFoundError:
            print('[ERROR] Street {} labels file not found.'.format(split))

        # Get split names
        if split == 'train':
            image_database_name = 'gsv_train_images'
            label_column_name = 'value'
        elif split == 'dev' or split == 'test':
            image_database_name = 'X'
            label_column_name = 'label'
        else:
            raise Exception('[ERROR] split should be in {train, dev, test}')

        # Save image and label data
        self.image_data = self.db[image_database_name]
        if 'AQI' in output_variable:
            self.label_data = np.array(labels[label_column_name])
        else:
            raise Exception('[ERROR] AQI not yet implemented for street.')

        # Save dimensions and output variable
        self.m = self.image_data.shape[0]
        self.output_variable = output_variable
        self.split = split

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
        # Grab image and label
        if self.split == 'train':
            X_item = np.asarray(self.image_data[:, :, :, item])
        else:
            X_item = np.asarray(self.image_data[item, :, :, :])
        Y_item = self.label_data[item]

        # Note: images are in (H, W, C) format, [0, 255] uint8. ToTensor()
        # expects (H, W, C) and converts to (C, H, W) in range [0, 1]. So
        # no transpose nor normalization is required.

        # Convert label to int in case of classification task
        if 'AQI' in self.output_variable:
            Y_item = int(Y_item)

        # Apply transforms
        if self.transform:
            X_item = self.transform(X_item)
        return X_item, Y_item

    def __del__(self):
        self.db.close()


def fetch_dataloader(dataset_types, data_dir, output_variable, params,
                     base_sat_image_file, base_sat_id_file, base_sat_labels_file,
                     sat_data_split):
    """
    Fetches the DataLoader object for each type of data.
    :param dataset_types: (list) list including ['train', 'dev', 'test']
    :param output_variable: (str) selected output variable
    :param params: (dict) a dictionary containing the model specifications
    :param base_sat_labels_file: (str) Path to the satellite images
    :param base_sat_id_file: (str) Path to the image identifiers and status
    :param base_sat_image_file: (str) Path to the unique_ID labels
    :param sat_data_split: (list) containing the % of each split in the order
        [size_train, size_dev, size_test]
    :return: dataloaders (dict) a dictionary containing the DataLoader object
        for each type of data
    """

    # Build datasets for selected output variable if they do not exist
    file_path = os.path.join(data_dir, output_variable)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    if len(glob.glob(os.path.join(
            file_path, '{}*'.format(params['model_type'])))) == 0:
        # If Sat model, build dataset. If Street, alert to missing data.
        if params['model_type'] == 'sat':
            print('[INFO] Building satellite dataset...')
            build_dataset.process_sat_data(
                base_sat_image_file, base_sat_id_file, base_sat_labels_file,
                data_dir, output_variable, sat_data_split)
        elif params['model_type'] == 'street':
            raise Exception('[ERROR] Could not find street data.')
        else:
            raise Exception('[ERROR] Model Type should be one of {sat, street}')

    # Use GPU if available
    use_cuda = torch.cuda.is_available()

    # Get mean and sd dictionaries from our training set and define transforms
    if params['model_type'] == 'sat':
        training_band_means = utils.load_dict(
            os.path.join(data_dir, output_variable, 'band_means.json'))
        training_band_sds = utils.load_dict(
            os.path.join(data_dir, output_variable, 'band_sds.json'))
    elif params['model_type'] == 'street':
        # Means and SDs are not required as images have only 3 channels
        training_band_means = None
        training_band_sds = None
    else:
        raise Exception('[ERROR] Model Type should be one of {sat, street}')
    transforms_dict = define_data_transforms(
        params['model_type'], training_band_means, training_band_sds)

    # Get data loaders
    dataloaders = {}

    for split in ['train', 'dev', 'test']:
        if split in dataset_types:
            # Get the correct Data class
            if params['model_type'] == 'sat':
                data = SatelliteData(
                    data_dir, output_variable, split, transforms_dict[split])
            elif params['model_type'] == 'street':
                data = StreetData(
                    data_dir, output_variable, split, transforms_dict[split])
            else:
                raise Exception('[ERROR] Model Type should be one of {sat, street}')

            # Grab dataloader for the data class and split
            dl = DataLoader(
                dataset=data, batch_size=params['batch_size'], shuffle=True,
                num_workers=params['num_workers'], pin_memory=use_cuda)
            dataloaders[split] = dl

    return dataloaders
