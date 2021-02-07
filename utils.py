import json
import matplotlib.pyplot as plt
import os
import shutil
import torch


def save_dict(dictionary, path):
    """
    Saves a dictionary as a json file to the specified path
    :param dictionary: (dict)
    :param path: (str)
    :return: void
    """
    with open(path, 'w') as file:
        json.dump(dictionary, file)


def load_dict(path):
    """
    Loads a dictionary from a json file
    :param path: (str)
    :return: (dict)
    """
    with open(path, 'r') as file:
        return json.load(file)


def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("[INFO] Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class Logger:
    def __init__(self, path):
        self.path = path

        # Create text file


def plot_learning(train_loss, test_loss, model_tag):
    # TODO implement
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['Train loss', 'Test loss'])
    path = os.path.join('03_Trained_Models', 'NN', 'images', 'model_%s_%s.png' %(date.today(), model_tag.replace('.', '-')))
    plt.savefig(path)
    plt.show()
    plt.clf()

