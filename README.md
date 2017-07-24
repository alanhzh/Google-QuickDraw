# Classifying noisy Google-QuickDraw images (keras)

Given noisy versions of the Google QuickDraw sketch image data set, we classify the object sketched inside the noisy image by denoising the image, then classifying the 'denoised' object. Although the QuickDraw data set contains 200+ sketch class labels, we select only 5 (specifically: pineapples, cats, fish, cups, and jackets) for the simple demonstration of how to denoise sketch images using autoencoders, then classify them using convolutional neural networks. An example result of performing our noisy classifier on 10 randomly selected noisy images

<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="200" height="200" align="center">

The provided files include the training code, an already trained model, and some example query images to apply the model to (we omitted > 500Mb worth of QuickDraw training data; they are open-sourced at https://github.com/googlecreativelab/quickdraw-dataset). 

Below are the algorithmic steps we used to train our model (visualizations included), along with usage instructions for how to use the model or even freshly train the model to your own custom-selected QuickDraw data set.

## Algorithm (visualizations are on query images): 

#### 1. We extract the training/validation/query images from provided Google QuickDraw dataset and add random Gaussian noise to them.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png" width="100%" align="center">

#### 2.  Using clean and noisy training/validation sketches, we train a convolutional autoencoder to learn how to denoise the images. 
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png" width="100%" align="center">

#### 3. Using clean training/validation sketches, we train a classifier using convolutional neural networks.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png" width="100%" align="center">

#### 4. Having trained our denoiser and classifier, we can now classify noisy sketch images by applying them sequentially.
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="100%" align="center">


## Usage:

### Usage 1 (apply model to query images)
To use the provided trained model (in `models`) to classify our provided noisy query images (in `query`), run the command:
``python
python QuickDraw_noisy_classifier.py
``
The result of this run will be saved to `answer`.

### Usage 2 (freshly train model, then apply model to query images)
You can freshly train a model using your own custom-selected Google QuickDraw data by:

1. Download your desired sketch classes from from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1 and placed them all into a folder of your liking. 

2. Delete all existing `.h5` models from the `models` directory (their absense will force the code to newly train/save the models). 

3. Edit two important variables in `QuickDraw_noisy_classifier.py`: `data_dir` and `categories`. `data_dir` is the directory where you stored your new QuickDraw `.npy` data files (which should have a `full%2Fnumpy_bitmap%2F` header, make sure of this!). `categories` is the list of the object category name strings you downloaded.

After following these instructions, running:
``python
python QuickDraw_noisy_classifier.py
``
will train/save the model with your QuickDraw data set, classify new noisy query images, and place the results into `answer`.

## Libraries required:
* keras 