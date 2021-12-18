# CS294-082 Final Project: Estimating the complexity of CIFAR-10 and training a compact and accurate mode
### *By Philipp Hohlfeld, Stefan Bielmeier, and Bob Ni*

## Contents and Descriptions

The purpose of this repository is to serve as the supporting documentation for reproducing the results of our paper. With this repository, readers should have everything needed to understand and regenerate the analysis in the paper. 

### Data

**Found in the data/ folder:**

gray_cat_dog_q_20_test.zip: Contains the training set used for most of the paper; this is a subset of CIFAR10 corresponding to all of the dog and cat images in the training set; we also apply JPEG compression (Q20) and grayscaling.

gray_cat_dog_q_20_test.zip: Contains the test set used for most of the paper; this is a subset of CIFAR10 corresponding to all of the dog and cat images in the test set; we also apply JPEG compression (Q20) and grayscaling.


### Scripts

**Found in the src/ foulder:**

nntailoring-binary.py: This is an adapted form of the nntailoring repo's capacityreq.py file, and has the same purpose of producing the number of thresholds, MEC, and max capacity requirements for a binary class set. 

nntailoring-multi.py: This is an adapted form of the nntailoring repo's capacityreq.py file, and has the same purpose of producing the number of thresholds, MEC, and max capacity requirements for a multi-class set. 

CIFAR10 2 class to memorization.ipynb: This is the notebooke used to generate the model trained to 100% memorization, using the data found in the data folder. This script includes:
- Compression and normalization of data 
- Splitting training, test, and validation sets
- Training the model across 13 neurons
- Examples of hyperparameter tuning
- Visualizing the capacity/accuracy curve

Optional CNN Training to 100 Percent Memorization.ipynb: This is the notebook corresponding to the CNN built and described in the appendix of the paper. It is optional, but provides promising results on another form of model which performs better in terms of accuracy. This notebook includes:
- Pulling and subsetting data from PyTorch's built in CIFAR10 data
- Splitting training, test, and validation sets
- A prebuilt CNN structure, with convolutional layers followed by pooling and fully connected layers
- Training script with parameters described in the paper
- Scoring accuracy across training, test, and validation sets
- Exporting the output of the convolutional layer to a CSV to be analyzed by nntailoring