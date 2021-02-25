from datetime import date, time
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


# Note: The following function was sourced from Stanford CS 230's Computer
# Vision project code examples, located at:
# https://github.com/cs230-stanford/cs230-code-examples
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


# Note: The following function was sourced from Stanford CS 230's Computer
# Vision project code examples, located at:
# https://github.com/cs230-stanford/cs230-code-examples
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
        """
        Instantiates the logger as a .txt file at the specified path
        :param path: (str) path to model outputs
        """
        self.path = path

        # Create text file
        with open(self.path, 'w') as file:
            file.write('Logger initiated: {} \n \n'.format(date.today()))

    def write(self, text):
        """
        Writes text to logger and prints text.
        :param text: (str)
        :return: void
        """
        with open(self.path, 'a+') as file:
            file.write(text + '\n')
        print(text)

    def write_dict(self, dict):
        """
        Writes a dictionary to the logger and prints its contents
        :param dict: (dict) to be written to logger
        :return: void
        """
        text = ""
        for key, value in dict.items():
            text = text + "{}: {} \n".format(key, value)

        with open(self.path, 'a+') as file:
            file.write(text)
        print(text)


def plot_learning(train_loss, eval_loss, path_to_save):
    """
    Generates and saves a plot of training and development set loss curves
    to file.
    :param train_loss: (list) training losses
    :param eval_loss: (list) dev set losses
    :param path_to_save: (str) output directory
    :return: void
    """
    plt.plot(train_loss)
    plt.plot(eval_loss)
    plt.legend(['Train loss', 'Dev loss'])
    path = os.path.join(path_to_save, 'learning_plot.png')
    plt.savefig(path)
    plt.clf()

