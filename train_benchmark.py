# Proposal benchmark model

# Data
#   Input: data/satellite image
#   Output: emission value classification

import torch # install using $ pip install torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# Temporary values
res = 64
num_channels = 3 # will be 7 later
num_images = 10
learning_rate = 0.05
num_categories = 6
classes = ['good', 'moderate', 'unhealthy_sensitive_groups',
           'unealthy', 'very_unhealthy', 'hazardous']

class Net(nn.Module):
    """
    Define the neural network
    """
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(res*res_num_channels, num_categories) # One linear layer

    def forward(self, x):
        """
        Define forward pass
        """
        x = nn.Softmax(self.layer1(x))  # Softmax activation
        return x

# may need dataloader if too much data
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

def process_images(image_files):
    """
    EDIT LATER
    Load and reshape images into an array where each column represents one image
    Input:
        filepath to folder with images or image filename if already aggregated
    Output:
        tensor of size (resolution * resolution * num_channelss, num_images)
    """
    data = np.random((res*res*num_channels, num_images)) # fake data for now
    x_train = torch.tensor(data)
    return x_train

def process_emissions(emissions_file):
    """
    EDIT LATER IF NEEDED
    Load emission measurements into one column
    Input:
        path to local folder that contains emission file
    Output:
        tensor of size (num_images, 1) each labeled with 1 of 6 categories
    """
    # emissions_data = pd.read_csv(emissions_file)
    # do things here
    emissions_data = np.array([np.randint(1,6) for i in range(num_images)])
    return emissions_data

def train(x_train, y_train):
    """
    Train model on training data
    Input:
        x data and labels
    Output:
        trained model
    """
    # model = torchvision.models.resnet18(pretrained=True)
    loss_sum = 0
    for i in range(num_images):
        inputs, labels = x_train[i], y_train[i]
        optimizer.zero_grad() # zero gradients
        output = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    # save trained model
    PATH = ('./models/benchmark_model.pth')
    torch.save(net.state_dict(), PATH)
    return net

def test(x_test, y_test, model):
    prediction = model(x_test)
    loss = (prediction - y_test).sum()

if __name__ == '__main__':

    # initialize network, loss function, and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # load data
    x_train = process_images("foo.image_files")
    print(x_train)
    # y_train = process_emissions('01_Data/filename')