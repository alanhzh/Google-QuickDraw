# Classifying noisy Google-QuickDraw images (keras)

Given Gaussian-noised sketch images from the Google QuickDraw dataset, we classify the object found inside the noisy image by first denoising it, then classifying the denoised image. An example result of our trained noisy classifier on 10 randomly selected noisy images

<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" height="100%" align="center">

The files provided here consists of a trained model and some example query images to test it with (we did not include the data set as open-sourced at https://github.com/googlecreativelab/quickdraw-dataset, and our minimal training data was already > 500Mb). We also incorporateed 5 sketch object classes in the training of the model (specifically: pineapples, cats, fish, cups, and jackets) for simplicity -- the Google QuickDraw dataset contains a couple hundred more class labels that you can experiment with, but our purpose here was to demonstrate the ability to denoise sketch images using autoencoders and later classify them using convolutional neural networks.

Below we provide the algorithm steps with visualizations, usage instructions, and how you can also newly train the model using your own custom-selected QuickDraw data.


## The algorithm: 

#### 1) Extract training/validation/query images from Google QuickDraw, and make noisy copies by adding random Gaussian noise to them. Here we demonstrate our query images with added Gaussian noise.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png" width="100%" align="center">

#### 2.a) Using the clean and noisy training/validation dataset, we train a convolutional autoencoder (convAE) to learn to denoise. We then apply this to the noisy query images.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png" width="100%" align="center">

#### 2.b) Using the clean training/validation dataset, we train a sketch object classifier using convolutional neural networks (convNN). We then apply this to the denoised query images.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png" width="100%" align="center">

#### 3) Having our trained denoiser and classifier models, we have finished building a procedure for classifying noisy sketch images.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" height="100%" align="center">


## Usage:
To use the pre-existing trained model to classify our noisy query images (found in the `query` directory), run the command:

> python QuickDraw_noisy_classifier.py

The result of this run will be saved to the `answer` directory.


## How to freshly train the model using custom-selected Google QuickDraw data:
If you are feeling ambitious, you can freshly train the model using your own custom-selected Google QuickDraw data set downloaded from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1 and placed in a folder of your liking. Make sure to delete the existing `.h5` models from the `models` directory (this will force the run to newly train and save the models). 

The last step before you run is to edit two important variables: `data_dir` and `categories` in `QuickDraw_noisy_classifier.py` to account for your custom data: 

- `data_dir` is the directory where your Google QuickDraw `.npy` data files (with `full%2Fnumpy_bitmap%2F` header) were placed in

- `categories` is a list of the object category name strings you downloaded 

After this, the previous usage instructions will be applicable where you can train the model and classify the noisy query images at the same time by running:

> python QuickDraw_noisy_classifier.py
