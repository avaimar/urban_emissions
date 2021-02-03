import argparse
import os

import torch
import numpy as np

import utils
import Models.CNNs
import Models.data_loaders as data_loader


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
def evaluate(model, loss_fn, dataloader, metrics, params):
    """
    Evaluate the model using the parameters specified in the params file path
    for a single epoch
    :param model: (torch.nn.Module)
    :param loss_fn: a function to compute the loss based on outputs and labels
    :param dataloader:
    :param metrics: (dict) a dictionary including relevant metrics
    :param params: a dictionary of hyper-parameters ['learning_rate',
     'batch_size', 'num_epochs', 'num_channels', 'save_summary_steps',
     'num_workers', 'cuda']
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

        # Convert data to torch variables
        eval_batch = torch.tensor(eval_batch.astype(float))
        labels_batch = torch.tensor(labels_batch.astype(float)) ## Is this okay for classification?

        # Forward propagation and loss computation
        output_batch = model(eval_batch)
        loss = loss_fn(output_batch, labels_batch)

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

    # Compute mean of metrics
    metrics_mean = {metric: np.mean([x[metric] for x in metrics_summary]) for metric in metrics}

    return metrics_mean


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

    # Fetch dataloaders
    print('[INFO] Loading the test set...')
    dataloaders = data_loader.fetch_dataloader(['test'], data_directory, params)
    test_dl = dataloaders['test']

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

    # Reload weights
    try:
        utils.load_checkpoint(restore_file, model)
    except FileNotFoundError:
        print('[ERROR] Model weights path not found.')

    # Evaluate
    print('[INFO] Starting evaluation...')
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)

    # Save performance metrics
    save_metrics_path = os.path.join(eval_output, 'metrics_test.json')
    try:
        utils.save_dict(test_metrics, save_metrics_path)
    except FileNotFoundError:
        print("[ERROR] Output path cannot be accessed.")
    print('[INFO] Evaluation completed.')
