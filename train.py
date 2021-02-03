import argparse
import os
import time

import torch
import torch.optim as optim
import numpy as np

import utils
import Models.CNNs
from evaluate import evaluate
from Models.data_loaders import data_loader


# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', default='01_Data/02_Imagery',
                    required=True, help='Directory containing the dataset')
parser.add_argument('-o', '--model_output', default='03_Trained_Models',
                    required=True, help='Directory to output model results')
parser.add_argument('-m', '--model_parameters',
                    required=True, help='Path to model parameters')


# Define training function
def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """
    Trains model on training data using the parameters specified in the
    params file path for a single epoch
    :param model: (torch.nn.Module)
    :param optimizer: (torch.optim)
    :param loss_fn: a function to compute the loss based on outputs and labels
    :param dataloader:
    :param metrics: (dict) a dictionary including relevant metrics
    :param params: a dictionary of parameters ['learning_rate', 'batch_size',
    'num_epochs', 'num_channels', 'save_summary_steps', 'num_workers', 'cuda']
    :param num_steps: (int)
    :return: void
    """

    # Set model to train mode
    model.train()

    # Set summary lists
    metrics_summary = []
    losses = []
    loss_avg=0

    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # Check for GPU and send variables
        if use_cuda:
            model.cuda()
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

        # Convert data to torch variables
        train_batch = torch.tensor(train_batch.astype(float))
        labels_batch = torch.tensor(labels_batch.astype(float)) ## TODO Is this okay for classification?

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
            output_batch, labels_batch = output_batch.numpy(), labels_batch.numpy()

            # Compute metrics
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            metrics_summary.append(summary_batch)
            summary_batch['loss'] = loss.item()
            losses.append(loss.item())
            loss_avg = np.mean(losses)


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader,
                       val_dataloader, metrics, params, model_dir,
                       restore_file=None):
    """
    Train the model and evaluate on a validation dataset using the parameters
    specified in the params file path.
    :param validation_metric:
    :param model: (torch.nn.Module) the model to be trained
    :param optimizer: (torch.optim)
    :param loss_fn:
    :param train_dataloader:
    :param val_dataloader:
    :param metrics:
    :param params:
    :param model_dir:
    :param restore_file:
    :return: void
    """

    # Reload weights if specified
    if restore_file is not None:
        try:
            utils.load_checkpoint(restore_file, model, optimizer)
        except FileNotFoundError:
            print('[ERROR] Model weights file not found.')

    # Initiate best validation accuracy
    best_val_metric = 0.0

    for epoch in range(params['num_epochs']):
        # Train single epoch on the training set
        print('[INFO] Training Epoch {}/{}'.format(
            epoch + 1, params['num_epochs']))
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate single epoch on the validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        val_metric = val_metrics[params['validation_metric']]
        is_best = val_metric >= best_val_metric

        # Save weights
        utils.save_checkpoint(
           {'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict()},
           is_best=is_best, checkpoint=model_output)

        # Save superior models
        if is_best:
            print('[INFO] New best {}: {}'.format(
                params['validation_metric'], val_metric))
            best_val_metric = val_metric

            # Save best val metrics
            best_json_path = os.path.join(
                model_dir, 'metrics_val_best_weights.json')
            utils.save_dict(val_metric, best_json_path)

        # Save metrics
        last_json_path = os.path.join(
            model_dir, 'metrics_val_last_weights.json')
        utils.save_dict(val_metric, last_json_path)


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

    # TODO create logger ?

    # Fetch dataloaders
    print('[INFO] Loading the datasets...')
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], data_directory, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    # Define model, and fetch loss function and metrics
    if params['model_type'] == 'regression':
        model = Models.CNNs.ResNetRegression(
            no_channels=3, out_features=1000) # TODO change num channels to depend on train
        loss_fn = Models.CNNs.loss_fn_regression
        metrics = Models.CNNs.metrics_regression
    else:
        model = Models.CNNs.ResNetClassifier(
            no_channels=3, out_features=1000, # TODO change num channels to depend on train
            num_classes=params['num_classes']
        )
        loss_fn = Models.CNNs.loss_fn_classification
        metrics = Models.CNNs.metrics_classification
    if use_cuda:
        model = model.cuda()

    # Define optimizer
    optimizer = getattr(optim, params['optimizer'])(
        params=model.parameters(), lr=params['learning_rate'])

    # Train
    print('[INFO] Starting training for {} epoch(s)'.format(
        params['num_epochs']))
    t0 = time.time()
    train_and_evaluate(model, optimizer, loss_fn, train_dl,
                       val_dl, metrics, params, model_output,
                       restore_file=None)
    print('[INFO] Training completed in {:2f} minute(s)'.format(
        (time.time() - t0) / 60))
