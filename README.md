# Classifying noisy Google QuickDraw images (keras)

Given noisy versions of the Google QuickDraw sketch image data set, we aim to classify the object sketched inside the noisy image. The procedure performed here is to first denoise the noisy image using autoencoders, then classify the image object using convolutional neural networks. For our training, we selected 5 object classes to train on: pineapples, cats, fish, cups, and jackets (~ 500Mb of data). An example result of performing our noisy classifier model on 10 noisy query images:

<p align="center">
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="90%">
</p>

The files provided here include the training code, an already trained noisy classifier model, and some example query images to apply the model to (we omitted including the training data as it was too large in size, and is publicly available for download at https://github.com/googlecreativelab/quickdraw-dataset). 

Below is the algorithm we used to train our model (visualizations included), alongside some usage instructions for how to use the provided model as is, or freshly train the model yourself using your own custom-selected classes from the QuickDraw data set.

## Algorithm: 

1. We extract clean training/validation/query images from our selected Google QuickDraw classes and add random Gaussian noise to them.

<p align="center">
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png" width="90%">
</p>

2. Using both the clean and noisy training/validation sketch images, we train a convolutional autoencoder to learn how to denoise the noisy images. 

<p align="center">
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png" width="90%">
</p>

3. Using only the clean training/validation sketch images, we train a clean classifier using convolutional neural networks.

<p align="center">
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png" width="90%">
</p>

4. Now with a trained denoiser and classifier in hand, we can classify noisy sketch images by applying the denoiser first then the classifier

<p align="center">
<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="90%">
</p>

## Usage:

### To apply model to query images in `query` folder:
To use the provided trained model (in `models`) to classify our provided noisy query images (in `query`), run the command:

> python QuickDraw_noisy_classifier.py

The result of this run will be placed into the `answer` folder.

### To train the model from scratch, then apply model to query images)
You can freshly train a model using your own custom-selected Google QuickDraw data as follows:

1. Download your desired sketch class `.npy` data sets from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1 and place them all into a folder of your choice. Make sure the filename headers are of the format `full%2Fnumpy_bitmap%2F` as the code assumes this!

2. Delete all existing `.h5` models from the `models` directory (their absence will force the code to newly train/save the models). 

3. Edit the two variables: `data_dir` and `categories` in `QuickDraw_noisy_classifier.py` accordingly. `data_dir` is the directory where you saved your new QuickDraw `.npy` data files. `categories` is the list of the object category name strings you saved.

After following the above instructions, follow the above run command:

> python QuickDraw_noisy_classifier.py

to train and save your new model with your QuickDraw data set, and to classify your noisy query images in the `query` folder. The results will be placed in the `answer` folder.

## Libraries required:
* keras, numpy, scipy, pylab