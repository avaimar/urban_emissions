import argparse
import os

import torch
import numpy as np

import utils
import Models.CNNs
from Models.data_loaders import fetch_dataloader


# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', default='01_Data/02_Imagery',
                    required=True, help='Directory containing the dataset')
parser.add_argument('-e', '--eval_output', default='03_Trained_Models',
                    required=True, help='Directory to output model results')
parser.add_argument('-m', '--model_parameters',
                    required=True, help='Path to model parameters')
parser.add_argument('-r', '--restore_file',
                    required=True, help='Path to model weights')


# Define training function
def evaluate(model, loss_fn, dataloader, metrics, params, logger):
    """
    Evaluate the model using the parameters specified in the params file path
    for a single epoch
    :param model: (torch.nn.Module)
    :param loss_fn: a function to compute the loss based on outputs and labels
    :param dataloader:
    :param metrics: (dict) a dictionary including relevant metrics
    :param params: a dictionary of hyper-parameters
    :param logger: (utils.Logger) file to output training information
    :return: metrics_mean: (dict)
    """

    # Set model to evaluation mode
    model.train(mode=False)

    # Detect GPU
    use_cuda = torch.cuda.is_available()

    # Set summary lists
    metrics_summary = []
    losses = []

    for i, (eval_batch, labels_batch) in enumerate(dataloader):
        # Check for GPU and send variables
        if use_cuda:
            model.cuda()
            eval_batch, labels_batch = eval_batch.cuda(), labels_batch.cuda()

        # Prepare data
        if not 'AQI' in params['output_variable']:
            labels_batch = labels_batch.float()

        # Forward propagation and loss computation
        output_batch = model(eval_batch)
        loss = loss_fn(output_batch, labels_batch)

        # Convert output_batch and labels_batch to np
        if use_cuda:
            output_batch, labels_batch = output_batch.cpu(), labels_batch.cpu()
        output_batch, labels_batch = output_batch.detach().numpy(), \
                                     labels_batch.detach().numpy()

        # Compute metrics
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        metrics_summary.append(summary_batch)
        losses.append(loss.item())

    # Compute mean of metrics
    metrics_mean = {metric: np.mean([x[metric] for x in metrics_summary]) for metric in metrics}
    logger.write('[MODEL INFO] Evaluation metrics mean')
    logger.write_dict(metrics_mean)

    # Compute average loss
    avg_loss = np.mean(losses)
    logger.write("[MODEL INFO] Running average evaluation loss: {:2f}".format(avg_loss))

    return metrics_mean, avg_loss


if __name__ == '__main__':
    """
    Evaluates model performance on a test set.
    """

    # Capture parameters from the command line
    args = vars(parser.parse_args())
    data_directory = args['data_directory']
    eval_output = args['eval_output']
    params_file = args['model_parameters']
    restore_file = args['restore_file']

    # Verify parameter file
    try:
        params = utils.load_dict(params_file)
    except FileNotFoundError:
        print("[ERROR] Parameter file not found.")

    # Use GPU if available
    use_cuda = torch.cuda.is_available()

    # Set random seed
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed(42)

    # Set up logger
    logger = utils.Logger(os.path.join(eval_output, 'logger.txt'))

    # Fetch dataloaders
    logger.write('[INFO] Loading the test set...')
    dataloaders = fetch_dataloader(
        ['test'], data_directory, params['output_variable'], params,
        params['base_data_file'], params['data_split'])
    test_dl = dataloaders['test']
    logger.write('[INFO] Test set loaded successfully...')

    # Get number of channels
    no_channels = next(iter(test_dl))[0].shape[1]

    # Define model, and fetch loss function and metrics
    if not 'AQI' in params['output_variable']:
        model = Models.CNNs.ResNetRegression(no_channels=no_channels)
        loss_fn = Models.CNNs.loss_fn_regression
        metrics = Models.CNNs.metrics_regression
    else:
        model = Models.CNNs.ResNetClassifier(
            no_channels=no_channels, num_classes=params['num_classes'])
        loss_fn = Models.CNNs.loss_fn_classification
        metrics = Models.CNNs.metrics_classification
    if use_cuda:
        model = model.cuda()

    # Reload weights
    logger.write('[INFO] Loading weights from file ' + restore_file)
    try:
        utils.load_checkpoint(restore_file, model)
    except FileNotFoundError:
        print('[ERROR] Model weights path not found.')
    logger.write('[INFO] Weights loaded successfully...')

    # Evaluate
    logger.write('[INFO] Starting evaluation...')
    test_metrics, _ = evaluate(model, loss_fn, test_dl, metrics, params, logger)

    # Save performance metrics
    save_metrics_path = os.path.join(eval_output, 'metrics_test.json')
    try:
        utils.save_dict(test_metrics, save_metrics_path)
    except FileNotFoundError:
        print("[ERROR] Output path cannot be accessed.")
    logger.write('[INFO] Evaluation completed.')
