import argparse
import os
import time

import torch
import torch.optim as optim
import numpy as np

import utils
import Models.CNNs
from evaluate import evaluate
from Models.data_loaders import fetch_dataloader


# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', default='01_Data/02_Imagery',
                    required=True, help='Directory containing the dataset')
parser.add_argument('-o', '--model_output', default='03_Trained_Models',
                    required=True, help='Directory to output model results')
parser.add_argument('-m', '--model_parameters',
                    required=True, help='Path to model parameters')


# Define training function
def train(model, optimizer, loss_fn, dataloader, metrics, params, logger):
    """
    Trains model on training data using the parameters specified in the
    params file path for a single epoch
    :param model: (torch.nn.Module)
    :param optimizer: (torch.optim)
    :param loss_fn: a function to compute the loss based on outputs and labels
    :param dataloader:
    :param metrics: (dict) a dictionary including relevant metrics
    :param params: a dictionary of parameters
    :param logger: (utils.Logger) file to output training information
    :return: (float) average training loss for the epoch
    """

    # Set model to train mode
    model.train(mode=True)

    # Set summary lists
    metrics_summary = []
    losses = []

    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # Check for GPU and send variables
        if use_cuda:
            model.cuda()
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

        # Prepare data
        if not 'AQI' in params['output_variable']:
            labels_batch = labels_batch.float()

        # Forward propagation, loss computation and backpropagation
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        # Log progress and compute statistics
        if i % params['save_summary_steps'] == 0:
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

        # Append loss
        losses.append(loss.item())

    # Compute metrics mean and add to logger
    metrics_mean = {metric: np.mean([x[metric] for x in metrics_summary]) for metric in metrics}
    logger.write('[MODEL INFO] Training metrics mean:')
    logger.write_dict(metrics_mean)

    # Compute average loss
    avg_loss = np.mean(losses)
    logger.write("[MODEL INFO] Running average training loss: {:2f}".format(avg_loss))

    return avg_loss


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader,
                       val_dataloader, metrics, params, model_dir, logger,
                       restore_file=None):
    """
    Train the model and evaluate on a validation dataset using the parameters
    specified in the params file path.
    :param model: (torch.nn.Module) the model to be trained
    :param optimizer: (torch.optim)
    :param loss_fn: (nn.MSEloss or nn.CrossEntropyLoss)
    :param train_dataloader: (torch.utils.data.Dataloader)
    :param val_dataloader: (torch.utils.data.Dataloader)
    :param metrics: (dict) metrics to be computed
    :param params: (dict) model parameters
    :param model_dir: (str) directory to output model performance
    :param restore_file: (str) path to model reload model weights
    :return: void
    """

    train_losses = []
    eval_losses = []

    # Reload weights if specified
    if restore_file is not None:
        try:
            utils.load_checkpoint(restore_file, model, optimizer)
        except FileNotFoundError:
            print('[ERROR] Model weights file not found.')
        logger.write('[INFO] Restoring weights from file ' + restore_file)

    # Initiate best validation accuracy
    if params['validation_metric'] == 'RMSE':
        best_val_metric = np.Inf
    else:
        best_val_metric = 0.0

    for epoch in range(params['num_epochs']):
        # Train single epoch on the training set
        logger.write('[INFO] Training Epoch {}/{}'.format(epoch + 1, params['num_epochs']))
        train_loss = train(
            model, optimizer, loss_fn, train_dataloader, metrics, params, logger)
        train_losses.append(train_loss)

        # Evaluate single epoch on the validation set
        val_metrics, eval_loss = evaluate(
            model, loss_fn, val_dataloader, metrics, params, logger)
        eval_losses.append(eval_loss)
        val_metric = val_metrics[params['validation_metric']]

        # Determine if model is superior
        if params['validation_metric'] == 'RMSE':
            is_best = val_metric <= best_val_metric
        else:
            is_best = val_metric >= best_val_metric

        # Save weights
        utils.save_checkpoint(
            state={'epoch': epoch + 1,
                   'state_dict': model.state_dict(),
                   'optim_dict': optimizer.state_dict()},
            is_best=is_best, checkpoint=model_output)

        # Save superior models
        if is_best:
            logger.write('[INFO] New best {}: {}'.format(
                params['validation_metric'], val_metric))
            best_val_metric = val_metric

            # Save best val metrics
            best_json_path = os.path.join(
                model_dir, 'metrics_val_best_weights.json')
            utils.save_dict(
                {params['validation_metric']: str(val_metric)},
                best_json_path)

        # Save metrics
        last_json_path = os.path.join(
            model_dir, 'metrics_val_last_weights.json')
        utils.save_dict(
            {params['validation_metric']: str(val_metric)}, last_json_path)

    # Save learning plot
    utils.plot_learning(train_losses, eval_losses, model_dir)


if __name__ == '__main__':
    # Capture parameters from the command line
    args = vars(parser.parse_args())
    data_directory = args['data_directory']
    model_output = args['model_output']
    params_file = args['model_parameters']

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
    logger = utils.Logger(os.path.join(model_output, 'logger.txt'))

    # Fetch dataloaders
    logger.write('[INFO] Loading the datasets...')
    dataloaders = fetch_dataloader(
        dataset_types=['train', 'dev'], data_dir=data_directory,
        output_variable=params['output_variable'], params=params,
        base_image_file=params['base_image_file'],
        base_id_file=params['base_id_file'],
        base_labels_file=params['base_labels_file'],
        data_split=params['data_split'])
    train_dl = dataloaders['train']
    val_dl = dataloaders['dev']
    logger.write('[INFO] Datasets loaded successfully...')

    # Get number of channels
    no_channels = next(iter(train_dl))[0].shape[1]

    # Define model, and fetch loss function and metrics
    if not 'AQI' in params['output_variable']:
        model = Models.CNNs.ResNetRegression(
            no_channels=no_channels, p=params['p_dropout'])
        loss_fn = Models.CNNs.loss_fn_regression
        metrics = Models.CNNs.metrics_regression
    else:
        model = Models.CNNs.ResNetClassifier(
            no_channels=no_channels, num_classes=params['num_classes'],
            p=params['p_dropout'])
        loss_fn = Models.CNNs.loss_fn_classification
        metrics = Models.CNNs.metrics_classification
    if use_cuda:
        model = model.cuda()

    # Define optimizer
    optimizer = getattr(optim, params['optimizer'])(
        params=model.parameters(), lr=params['learning_rate'])

    # Train
    logger.write('[INFO] Starting training for {} epoch(s)'.format(
        params['num_epochs']))
    t0 = time.time()
    train_and_evaluate(model, optimizer, loss_fn, train_dl,
                       val_dl, metrics, params, model_output, logger,
                       restore_file=None)
    logger.write('[INFO] Training completed in {:2f} minute(s)'.format(
        (time.time() - t0) / 60))
