# Classifying noisy Google-QuickDraw images (keras)

Given Gaussian-noised sketch images from the Google QuickDraw dataset, we classify the object found inside the noisy image by first denoising it then classify the denoised image. The result of our trained noisy classifier on 10 example noisy images is

<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" height="100%" align="center">

The code provided here only provides the trained model and some example query images to test the trained model with (we did not include the data set as it is provided online at https://github.com/googlecreativelab/quickdraw-dataset, and our training data was > 500Mb). Also, for reasons of simplicity we only incorporate 5 sketch object classes in the training of the model (specifically: pineapples, cats, fish, cups, and jackets) -- the Google QuickDraw dataset contains a couple hundred more class labels that one can experiment with, but the code's purpose here is to demonstrate the ability to denoise sketch images using autoencoders and later classify them using convolutional neural networks.

Below we provide the algorithm steps, usage instructions, and how instructions for how to newly train the model using your own custom-selected QuickDraw data.


## The algorithm: 

#### 1) Extract training/validation/query images from Google QuickDraw, and make noisy copies by adding random Gaussian noise to them. Here we show the query images with added Gaussian noise.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png" width="100%" align="center">

#### 2.a) Using the clean and noisy training/validation dataset, we train a convolutional autoencoder (convAE) to learn to denoise. We then apply this to the noisy query images.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png" width="100%" align="center">

#### 2.b) Using the clean training/validation dataset, we train an sketch object classifier using convolutional neural networks (convNN). We then apply this to the denoised query images.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png" width="100%" align="center">

#### 3) Having denoised the noisy query images and classified the sketch object in the image, we have our final result of a classification for noisy sketch images.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" height="100%" align="center">


## Usage:
To use the pre-existing trained model for classify the noise query images that can already be found in the `query` directory, use the command:

> python QuickDraw_noisy_classifier.py

The result of this run will be saved into the `answer` directory.


## How to freshly train the model using custom-selected Google QuickDraw data:
If you are feeling ambitious, you can freshly train the model using your own custom-selected Google QuickDraw data set, delete the existing `.h5` models from the `models` directory (this will force it to train and save the models). Afterwards, download some Google QuickDraw object datasets from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1 and place them in a folder of your liking. This folder will contain the training, validation, and test (query) data for the run.

The last step is to edit two variables: `data_dir` and `categories` in `QuickDraw_noisy_classifier.py` to account for your custom data: 

- `data_dir` is the directory where your Google QuickDraw `.npy` data files (with `full%2Fnumpy_bitmap%2F` header) were placed in

- `categories` is a list of the object category name strings you downloaded 

After this, the previous usage instructions will be applicable where you can train the model and classify the noisy query images at the same time by running:

> python QuickDraw_noisy_classifier.py

The training, validation, and query images will be handlded and created automatically.
