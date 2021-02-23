import glob
import h5py
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
            # Note: We first convert to tensor to get the appropriate channel ordering
            #transforms.ToTensor(), # Converts (H, W, C) to (C, H, W)
            #transforms.ToPILImage(), # Preserves (C, H, W)
            #transforms.RandomHorizontalFlip(), # Requires (.., H, W)
            #transforms.RandomVerticalFlip(), # Requires (.., H, W)
            #transforms.RandomRotation(20), # Requires (.., H, W)
            transforms.ToTensor(),
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds)),
        ]),
        'dev': transforms.Compose([
            transforms.ToTensor(), # Converts (H, W, C) to (C, H, W)
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(), # Converts (H, W, C) to (C, H, W)
            transforms.Normalize(tuple(image_net_means), tuple(image_net_sds))
        ])
    }
    return data_transforms


# Satellite Data class for dataloaders
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
        data_path = os.path.join(data_dir, output_variable, 'sat_{}.hdf5'.format(split))
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


def fetch_dataloader(dataset_types, data_dir, output_variable, params,
                     base_image_file, base_id_file, base_labels_file,
                     data_split):
    """
    Fetches the DataLoader object for each type of data.
    :param dataset_types: (list) list including ['train', 'dev', 'test']
    :param output_variable: (str) selected output variable
    :param params: (dict) a dictionary containing the model specifications
    :param base_labels_file: (str) Path to the satellite images
    :param base_id_file: (str) Path to the image identifiers and status
    :param base_image_file: (str) Path to the unique_ID labels
    :param data_split: (list) containing the % of each split in the order
        [size_train, size_dev, size_test]
    :return: dataloaders (dict) a dictionary containing the DataLoader object
        for each type of data
    """

    # Build datasets for selected output variable if they do not exist
    file_path = os.path.join(data_dir, output_variable)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    if len(glob.glob(os.path.join(file_path, 'sat*'))) == 0:
        print('[INFO] Building dataset...')
        build_dataset.process_sat_data(
            base_image_file, base_id_file, base_labels_file, data_dir,
            output_variable, data_split)

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

    for split in ['train', 'dev', 'test']:
        if split in dataset_types:
            data = SatelliteData(
                data_dir, output_variable, split, transform=transforms_dict[split])
            dl = DataLoader(
                dataset=data, batch_size=params['batch_size'], shuffle=True,
                num_workers=params['num_workers'], pin_memory=use_cuda)
            dataloaders[split] = dl

    return dataloaders
