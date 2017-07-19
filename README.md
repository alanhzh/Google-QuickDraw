# Google-QuickDraw

Given Gaussian-noised sketch images from the Google QuickDraw dataset, we classify the object found inside the noisy image by first denoising it then classify the denoised image. The result of our trained noisy classifier on 10 example noisy images is

<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" height="100%" align="center">

The code provided here only provides the trained model and some example query images to test the trained model with (we did not include the data set as it is provided online at https://github.com/googlecreativelab/quickdraw-dataset, and our training data was > 500Mb). Also, for reasons of simplicity we only incorporate 5 sketch object classes in the training of the model (specifically: pineapples, cats, fish, cups, and jackets) -- the Google QuickDraw dataset contains a couple hundred more class labels that one can experiment with, but the code's purpose here is to demonstrate the ability to denoise sketch images using autoencoders and later classify them using convolutional neural networks.

Below we provide the algorithm steps, usage instructions of the pre-existing trained model, and how instructions for how to newly train the model using your own custom-selected QuickDraw data.


## The algorithm: 

### 1.a) Create/extract query images from Google QuickDraw
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png" width="100%" align="center">

### 1.b) Make noisy copies of the dataset by adding random Gaussian noise to them

### 2.a) Using the clean and noisy versions of the dataset, we train a convolutional autoencoder (convAE) to learn to denoise the noisy images
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png" width="100%" align="center">

### 2.b) Using the clean dataset, we train a classifier using convolutional neural networks (convNN) to learn how to classify the sketches
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png" width="100%" align="center">

### 3) We now classify our noisy query images by applying the denoising convAE, then classify the object using the convNN classifier
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" height="100%" align="center">


## Usage:
To classify the noise query images found in the `query` directory, use the command:

> python QuickDraw_noisy_classifier.py

The images resulting of this run will be found in the `answer` directory.


## How to newly train the model using custom-selected Google QuickDraw data:
To freshly train the model using your own custom-selected Google QuickDraw data set, delete the existing `.h5` models from the `models` directory (this will force it to train and save the models). Afterwards, download some Google QuickDraw object datasets from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1 and place them in a folder of your liking. This folder will contain the training, validation, and test (query) data for the run.

The last step is to edit two variables in `QuickDraw_noisy_classifier.py` to account for your custom data: 

`data_dir` which is the directory where your Google QuickDraw `.npy` data files with `full%2Fnumpy_bitmap%2F` header are located

and

`categories` which is a list of the category name strings you downloaded 

After this, the code `QuickDraw_noisy_classifier.py` should be ready to run. The training, validation, and query images will be created automatically.
