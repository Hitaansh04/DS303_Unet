# DS303_Unet
## Overview
Repo containing our implementation of the U-net Paper as the project for the course DS303.
The script U-net.py contains the model architecture of our implementation. The major differences are usage of Batch Normalization in the double convolution layer since it allows us to use much faster learning rates.

The other major difference being the manner of concatenation where we've used resizing instead of centre crop so that dimensions are not disturbed during upsampling.

Next, is the script Carvana.py which contains the class of the dataset that we are going to use.We replaced the 'gif' extension of the masks to 'jpg' for better compatibility. We also scaled the pixels from 0-255 to fit between 0-1 for ease of mathematical operations.

The utils.py contains all the utilities that we shall be accessing throughout the project, that includes saving and loading the checkpoints,
creating dataloaders for training and validation data and masks. The validation data is a subset of training data that is used to measure the model accuracy during training. It also contains the functions for checking accuracy and to save_predictions as images. 

Lastly , the train.py script is where the magic happens. We include basic transformations before we can put the image as an input tensor to the model. The accuracy metric that was used is dice score which is a standard practice in segmentation of images.

## Possible Additions / Improvements : 
These changes could improve the implementations had we got more time.
1. During training , we could add optional parser arguements for hyperparameters such as epochs, learning rate , batch size . 
2. Write a script to predict segmentation images . 

## Architecture



The model was trained for nearly 5 hours and here is the accuracy and dice score of our model.

![Screenshot 2023-04-29 at 11 44 32 AM](https://user-images.githubusercontent.com/63915396/235287339-b5d0c6ff-2a0a-4ea0-bc0f-c420856dd8e5.png)
