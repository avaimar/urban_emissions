import h5py
import numpy as np
import os
import pandas as pd
import pickle
import random
import utils


# Define global means over each channel of the training set. These are
# estimated means in order to compute channel standard deviations when
# building the dataset
GLOBAL_MEANS = np.array([0.26334649324417114, 0.25160321593284607,
                         0.2347201257944107,  0.24427558481693268,
                         0.33065736293792725, 0.23822002112865448,
                         0.1870376616716385])


def create_data_split(valid_rows_bool, data_split):
    """
    Creates a train/val/test split given a list of sizes for each split.
    :param valid_rows: (np.array) Boolean array indicating images that should
        be included in the train/dev/test split
    :param data_split: (list) containing the % of each split in the order
        [size_train, size_val, size_test]
    :return: a tuple (split_IDs, split_sizes) containing an array indicating
        to which split a row belongs and a dictionary of sizes for each split
    """
    # Set seed and verify sat_data_split is appropriate given dataset size
    np.random.seed(42)
    assert abs(sum(data_split) - 1) < 1e-10

    # Get valid image IDs
    valid_imgs_IDs = np.arange(0, valid_rows_bool.shape[0])[valid_rows_bool]

    # Get sizes for each split
    m = valid_rows_bool.sum()
    train_size = int(m * data_split[0])
    val_size = int(m * data_split[1])

    # Create permutation and indexes for each split
    perm = np.random.permutation(valid_imgs_IDs)
    train_id, val_id, test_id = perm[: train_size], \
                                perm[train_size: train_size + val_size], \
                                perm[train_size + val_size:]

    # Reorder
    split_IDs = np.zeros(valid_rows_bool.shape, dtype='object')
    split_IDs[train_id] = 'train'
    split_IDs[val_id] = 'val'
    split_IDs[test_id] = 'test'

    # Get split sizes
    split_sizes = {'train': train_size, 'val': val_size,
                   'test': m - train_size- val_size}

    return split_IDs, split_sizes


def preprocess_label_data(base_labels_file, output_variable):
    """
    Prepares the labels data frame for processing
    :param base_labels_file: (str) location of labels data
    :param output_variable: (str) selected model output variable
    :return: (DataFrame)
    """
    # Load Labels file
    try:
        with open(base_labels_file, 'r') as labels_file:
            label_data = pd.read_csv(
                labels_file, dtype={
                    'Unique_ID': 'string', 'Location_type': 'string',
                    'Zipcode': 'string', 'County': 'string', 'type': 'string',
                    'measurement': 'string', 'value': float, 'lat': float,
                    'lon': float, 'AQI_level': 'string'})
    except FileNotFoundError:
        print('[ERROR] Labels file not found.')

    # Filter label data for selected variable
    label_data = label_data[label_data['type'] == output_variable]

    # Define label and convert to int for AQI (hdf5 requires this)
    if 'AQI' in output_variable:
        label_data['label'] = label_data['AQI_level']
        aqi_dict = {'good': 0, 'moderate': 1, 'unhealthy_sensitive_groups': 2,
                    'unhealthy': 3, 'very_unhealthy': 4, 'hazardous': 5}
        label_data = label_data.replace({'label': aqi_dict})
    else:
        label_data['label'] = label_data['value']

    return label_data[['Unique_ID', 'label']]


def process_sat_data(base_image_file, base_id_file, base_labels_file,
                     data_dir, output_variable, data_split):
    """
    Pre-processes Satellite Data and creates train/val/test splits.
    :param base_labels_file: (str) Path to the satellite images
    :param base_id_file: (str) Path to the image identifiers and status
    :param base_image_file: (str) Path to the unique_ID labels
    :param data_dir: (str) Path to where train, val, test files will be
        exported
    :param output_variable: (str) Selected output variable
    :param data_split: (list) containing the % of each split in the order
        [size_train, size_val, size_test]
    :return: void
    """
    # Load image file
    try:
        db = h5py.File(base_image_file, 'r')
        image_data = db['imagery_no_mask']
    except FileNotFoundError:
        print('[ERROR] Dataset not found.')

    # Load Identifier file
    try:
        with open(base_id_file, 'rb') as id_file:
            id_data = pickle.load(id_file)
    except FileNotFoundError:
        print('[ERROR] Image IDs file not found.')

    # Join id_data and label_data
    label_data = preprocess_label_data(base_labels_file, output_variable)
    id_data = id_data.merge(
        label_data, on='Unique_ID', how='left', validate='one_to_one')
    id_data = id_data[['Unique_ID', 'task_status', 'label']]

    # Get rows whose task was completed and have a label
    valid_imgs = np.array((id_data['task_status'] == 'COMPLETED') &
                          (id_data['label'].notna()))

    # Create train/dev/test splits
    split_IDs, split_sizes = create_data_split(valid_imgs, data_split)

    # Gather image dimensions and ensure dimension ordering is (n_W, n_H, n_C)
    n_H = image_data.shape[1]
    n_C = image_data.shape[2]

    # Ensure dimension ordering is in line with PyTorch
    if n_C > n_H:
        print('[ERROR] Channel ordering is incorrect.')
        assert 0

    # Create path for data directories
    main_path = os.path.join(data_dir, output_variable)
    if not os.path.exists(main_path):
        os.mkdir(main_path)

    # Open datasets
    path = os.path.join(main_path, 'sat_{}.hdf5')
    train_db = h5py.File(path.format('train'), "w")
    train_db.create_dataset(name='X', shape=(split_sizes['train'], n_H, n_H, n_C), dtype='f')
    train_db.create_dataset(name='Y', shape=(split_sizes['train'], 1), dtype='f')

    val_db = h5py.File(path.format('dev'), "w")
    val_db.create_dataset(name='X', shape=(split_sizes['val'], n_H, n_H, n_C), dtype='f')
    val_db.create_dataset(name='Y', shape=(split_sizes['val'], 1), dtype='f')

    test_db = h5py.File(path.format('test'), "w")
    test_db.create_dataset(name='X', shape=(split_sizes['test'], n_H, n_H, n_C), dtype='f')
    test_db.create_dataset(name='Y', shape=(split_sizes['test'], 1), dtype='f')

    # Set up arrays to compute normalization metrics
    band_means = np.zeros((n_C,))
    band_sds = np.zeros((n_C,))

    # Loop over each image, identify valid images and create train/dev/test
    train_counter, val_counter, test_counter = 0, 0, 0
    for i in range(image_data.shape[3]):
        # Identify if image is valid
        if valid_imgs[i]:
            if i % 500 == 0:
                print('[INFO] Processing image {}/{}'.format(i, image_data.shape[3]))

            # Grab image
            img = np.array(image_data[:, :, :, i])
            label = id_data.iloc[i]['label']
            assert(img.shape == (n_H, n_H, n_C))

            # Identify which dataset the image belongs to and write
            if split_IDs[i] == 'train':
                train_db['X'][train_counter] = img
                train_db['Y'][train_counter] = label
                train_counter += 1

                # Add to normalization metrics
                img_means = np.mean(img, axis=(0, 1))
                band_means += img_means / split_sizes['train']

                img_sds = (img_means - GLOBAL_MEANS)**2
                band_sds += img_sds / split_sizes['train']

            elif split_IDs[i] == 'val':
                val_db['X'][val_counter] = img
                val_db['Y'][val_counter] = label
                val_counter += 1
            elif split_IDs[i] == 'test':
                test_db['X'][test_counter] = img
                test_db['Y'][test_counter] = label
                test_counter += 1
            else:
                assert 0

    # Close datasets
    train_db.close()
    val_db.close()
    test_db.close()
    db.close()

    # Verify that all images were added to databases
    print('Number of train images loaded: ', train_counter)
    print('Number of val images loaded: ', val_counter)
    print('Number of test images loaded: ', test_counter)

    # Obtain means and sds for each band in the train set (for normalization)
    band_means = {'band_{}'.format(i): band_mean.item() for (i, band_mean) in
                  enumerate(np.nditer(band_means))}
    band_sds = np.sqrt(band_sds)
    band_sds = {'band_{}'.format(i): band_sd.item() for (i, band_sd) in
                enumerate(np.nditer(band_sds))}
    utils.save_dict(band_means, os.path.join(main_path, 'band_means.json'))
    utils.save_dict(band_sds, os.path.join(main_path, 'band_sds.json'))
