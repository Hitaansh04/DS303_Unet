# DS303_Unet
Repo containing our implementation of the U-net Paper as the project for the course DS303.
The script U-net.py contains the model architecture of our implementation. The major differences are usage of Batch Normalization in the double convolution layer since it allows us to use much faster learning rates.

The other major difference being the manner of concatenation where we've used resizing instead of centre crop so that dimensions are not disturbed during upsampling.

