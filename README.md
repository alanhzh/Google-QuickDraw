# Classifying noisy Google-QuickDraw images (keras)

Given noisy versions of the Google QuickDraw sketch image data set, we classify the object sketched inside the noisy image by denoising the image, then classifying the 'denoised' object. We select 5 object classes: pineapples, cats, fish, cups, and jackets, for the simple demonstration of how to denoise sketch images using autoencoders, then classify them using convolutional neural networks. An example result of performing our noisy classifier on 10 randomly selected noisy images:

<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="150%" align="center">

The files provide here include the training code, an already trained noisy classifier model, and some example query images to apply the model to (we did not include the QuickDraw data set as it exceeded 500Mb in size and are open-sourced at https://github.com/googlecreativelab/quickdraw-dataset). 

Below is the algorithm we used to train our model (visualizations included), alongside the usage instructions for how to use the provided model as is, or freshly train the model yourself using your own custom-selected QuickDraw object classes.

## Algorithm: 

#### 1. We extract the training/validation/query images from provided Google QuickDraw dataset and add random Gaussian noise to them.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png" width="100%" align="center">

#### 2a.  Using clean and noisy training/validation sketches, we train a convolutional autoencoder to learn how to denoise the images. 
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png" width="100%" align="center">

#### 2b. Using clean training/validation sketches, we train a classifier using convolutional neural networks.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png" width="100%" align="center">

#### 3. Having trained our denoiser and classifier, we can now classify noisy sketch images by applying to them the denoiser then the classifier.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" align="center">


## Usage:

### Usage 1 (apply model to query images)
To use the provided trained model (in `models`) to classify our provided noisy query images (in `query`), run the command:

> python QuickDraw_noisy_classifier.py

The result of this run will be saved into `answer`.

### Usage 2 (freshly train model, then apply model to query images)
You can freshly train a model using your own custom-selected Google QuickDraw data as follows:

1. Download your desired sketch classes from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1 and place them all into a folder of your choice. 

2. Delete all existing `.h5` models from the `models` directory (their absence will force the code to newly train/save the models). 

3. Edit the two variables: `data_dir` and `categories` in `QuickDraw_noisy_classifier.py` accordingly. `data_dir` is the directory where you saved your new QuickDraw `.npy` data files (which should have a `full%2Fnumpy_bitmap%2F` header, make sure of this!). `categories` is the list of the object category name strings you saved.

After following the above instructions, simply run:
``python
python QuickDraw_noisy_classifier.py
``
This will train/save the model with your QuickDraw data set, classify new noisy query images, and place the results into `answer`.

## Libraries required:
* keras 