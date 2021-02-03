import json
import matplotlib.pyplot as plt


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


def load_checkpoint():
    pass


def save_checkpoint():
    pass


def plot_learning(train_loss, test_loss, model_tag):
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['Train loss', 'Test loss'])
    path = os.path.join('03_Trained_Models', 'NN', 'images', 'model_%s_%s.png' %(date.today(), model_tag.replace('.', '-')))
    plt.savefig(path)
    plt.show()
    plt.clf()

