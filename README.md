# Using satellite and street level images to predict urban emissions

## Description
This project examines the relationship between the level of ozone concentration
in urban locations and their physical features through the use of Convolutional
Neural Networks (CNNs). We train two models, including one trained on satellite
imagery to capture higher-level features such as the location's geography, and 
the other trained on street-level imagery to learn ground-level features such as
motor vehicle activity. These features are then concatenated to form a shared
representation from which to predict the location's level of ozone as measured
in parts per billion. 

## Code structure
* The `02_Scripts/` directory comprises the code to scrape and preprocess
the ozone concentration data, which is sourced from the 
[AirNow API](https://docs.airnowapi.org/). It also contains a 
`01_Data_Exploration` directory which includes code to visualize elements
of the dataset such as particular data points and a geographical distribution
of the locations with ozone readings.

* In "imagery" we find the scripts to retrieve satellite imagery from Google 
Earth Engine (`imagery/getting_imagery_no_mask.py`) and the street level images 
from Google Street View (training set is build with 
`imagery/get_street_imagery_train_set.py` and .... )

* The `Models` directory comprises `CNNs.py`, which implements the CNNs
used for training. We use a ResNet-18 model for both the satellite and the
street-level imagery, pretrained on the ImageNet dataset. Adjustments include
modification of the input layer to accommodate for higher image channels in the
satellite dataset (7), additional regularization through the use of
`Fully Connected -> Batchnormalization -> Dropout` blocks in the highest layers,
and modification of the final layer's number of units to fit our regression and
classification tasks. The `data_loaders.py` script includes the `SatelliteData`
and `StreetData` classes implemented to user Torch's DataLoaders. The
DataLoaders call upon the functions in `build_dataset.py` to build the
train/dev/test splits for the satellite data if not already generated.

* `train.py`, `evaluate.py` and `search_hyperparams.py` implement the code
to train our models. It is important to note that this training code has been 
largely adapted from Stanford CS 230's Computer Vision project code examples 
found at [this link](https://github.com/cs230-stanford/cs230-code-examples).

* `utils.py` implements helper classes and functions used to log our model
training process, load and write dictionaries, and plot learning loss curves.
