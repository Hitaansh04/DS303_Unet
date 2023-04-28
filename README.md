# DS303_Unet
Repo containing our implementation of the U-net Paper as the project for the course DS303.
The script U-net.py contains the model architecture of our implementation. The major differences are usage of Batch Normalization in the double convolution layer since it allows us to use much faster learning rates.

The other major difference being the manner of concatenation where we've used resizing instead of centre crop so that dimensions are not disturbed during upsampling.

Next, is the script Carvana.py which contains the class of the dataset that we are going to use.We replaced the 'gif' extension of the masks to 'jpg' for better compatibility. We also scaled the pixels from 0-255 to fit between 0-1 for ease of mathematical operations.

