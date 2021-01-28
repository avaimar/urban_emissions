import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

selected_output_variable = "OZONE"
data_file = "../01_Data/02_Imagery/data_and_imagery_test.pkl"

def visualize(data_file, selected_output_variable, image_index):
    data = pickle.load(open(data_file, 'rb'))
        
    # filter for output variable
    data = data[data['type'] == selected_output_variable]

    # get X data as np array and check dims
    image = np.array(data['imagery'].to_list())[image_index]
    print(image.shape)

    imgplot = plt.imshow(image)
    plt.show()

visualize(data_file, selected_output_variable, 3)