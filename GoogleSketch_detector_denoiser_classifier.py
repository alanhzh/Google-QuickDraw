"""

 GoogleSketchRNN_OPENCV_CAE_CNN.py (Anson Wong)
 
 Given a set of task images that are large 112 x 112 images with a noisy object image in it,
 we use Google SketchRNN data set of labeled clean animals/objects as training data to classify
 what the noisy object is in the task images. The tools use (in chronological order) is opencv for
 edge/object detection, convolutional autoencoders for denoising, and convolutional neural networks 
 for classification.

 The main steps in this code are:
  1) Create/collect task images for classification, and clean images for training   (create_task_images.py)
  2) We use edge detection to find the noisy object that needs classification   (object_detect_28x28.py)
  3) We then train a CAT to learn how to denoise the image   (GoogleSketchRNN_OPENCV_CAE_CNN.py)
  4) We then train a CNN to classify the denoised image   (GoogleSketchRNN_OPENCV_CAE_CNN.py)
  
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import argparse

from skimage import exposure
from PIL import Image
import scipy.misc

import cv2

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D  # CAE
from keras.models import Model  # CAE
from keras.callbacks import TensorBoard  # CAE
from keras.models import Sequential  # CNN
from keras.layers import Dense, Dropout, Flatten  # CNN
from keras.layers import Conv2D, MaxPooling2D  # CNN
from keras import backend as K  # CNN
from keras.models import load_model  # save keras model

def main():

    # =======================================================
    #
    # Set parameters for our run
    #
    # =======================================================
    categories = ['car', 'cat', 'coffee cup', 'cookie', 'fish']  # sets item/animal categories
    xpixels = 28  # set x pixel numbers for clean training/test examples
    ypixels = 28  # set y pixel numbers for clean training/test examples
    xpixels_task = 4 * 28  # set x pixel numbers for raw task examples
    ypixels_task = 4 * 28  # set y pixel numbers for raw task examples
    n_take_train = 8000  # number of training images to take from each category
    n_take_test = 1600  # number of test images to take from each category
    obj_noise_factor = 0.5  # how much gaussian noise to add to our noisy images [0,1]
    bg_noise_factor = 0.0  # how much gaussian noise to add to task images background [0,1]

    use_CAE = True  # use convolutional autoencoder?
    use_saved_CAE_model = True  # used previously saved CAE model? only relevant if use_CAE == True
    n_epochs_CAE = 30  # number of epochs for CAE training (30)
    batch_size_CAE = 128  # batch size of CAE training

    use_CNN = True  # use convolutional neural network?
    use_saved_CNN_model = True  # used previously saved CNN model? only relevant if use_CNN == True
    n_epochs_CNN = 5  # number of epochs for CNN training (5)
    batch_size_CNN = 128  # batch size of CNN training

    check_noisydata = False  # check noisy data before run?
    check_original_training = False  # check original training images before run?
    check_original_test = False  # check original test images before run?

    # =======================================================
    #
    # Create task images
    # Noisy up some clean Google SketchRNN images and embed them into a noisy large background.
    #
    # =======================================================
    n_take_task = 4  # number of test images to take from each category
    x_task_temp = []
    for index_category, category in enumerate(categories):
        input_file = "data/full%2Fnumpy_bitmap%2F" + category + ".npy"
        data = np.load(input_file)
        print("[%d/%d] Reading category index %d :'%s'" %
              (index_category, len(categories), index_category, category))
        for j, data_j in enumerate(data):
            img = np.array(data_j).reshape((ypixels, xpixels))
            if j < n_take_task:
                x_task_temp.append(img)
            else:
                break
    x_task_temp = np.array(x_task_temp)

    # Embed our image into a noisy background
    x_task_temp_final = embed_randomly_in_largebackground(x_task_temp, obj_noise_factor, bg_noise_factor)

    if 0:
        plot_img(x_task_temp_final[0], "1) Object Image", 1)
        sys.exit()

    for i in range(20):
        print("Printing task image %d to file..." % (i+1))
        scipy.misc.imsave("task_images/task_image_%d.jpg" % (i + 1), x_task_temp_final[i])


    # =======================================================
    #
    # Data I/O and Preprocessing
    # Read Google SketchRNN data of sketches and append them to the training/test sets.
    # We take original copies of it, and also create noisy copies of them too.
    #
    # =======================================================
    #
    # Read clean training/test images
    #
    n_categories = len(categories)  # number of classes
    x_train = []; y_train = []  # holds training images/labels
    x_test = []; y_test = []  # holds test images/labels
    for index_category, category in enumerate(categories):
        input_file = "data/full%2Fnumpy_bitmap%2F" + category + ".npy"
        data = np.load(input_file)
        n_data = len(data)
        print("[%d/%d] Reading category index %d :'%s' (%d images: take %d training, take %d test)" %
              (index_category, len(categories), index_category, category, n_data, n_take_train, n_take_test))
        for j, data_j in enumerate(data):
            img = np.array(data_j).reshape((ypixels, xpixels))
            if j < n_take_train:
                x_train.append(img); y_train.append(index_category)  # append to training set
            elif j - n_take_train < n_take_test:
                x_test.append(img); y_test.append(index_category)  # append to test set
            else:
                break
    x_train = np.array(x_train); y_train = np.array(y_train)  # convert to numpy arrays
    x_test = np.array(x_test); y_test = np.array(y_test)  # convert to numpy arrays
    x_train_original = x_train.copy(); y_train_original = y_train.copy()  # make original untouched copies
    x_test_original = x_test.copy(); y_test_original = y_test.copy()  # make original untouched copies

    # Create noisy copies of our image data sets by adding gaussian noise
    x_train_noisy = x_train + obj_noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) + 1) / 2
    x_test_noisy = x_test + obj_noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) + 1) / 2
    x_train_noisy = np.clip(x_train_noisy, 0., 255.)
    x_test_noisy = np.clip(x_test_noisy, 0., 255.)

    # Convert our greyscaled image data sets to have values [0,1] and reshape to form (n, ypixels, xpixels, 1)
    x_train = convert_img2norm(x_train, ypixels, xpixels)
    x_test = convert_img2norm(x_test, ypixels, xpixels)
    x_train_noisy = convert_img2norm(x_train_noisy, ypixels, xpixels)
    x_test_noisy = convert_img2norm(x_test_noisy, ypixels, xpixels)


    #
    # Read task images and use opencv edge/contour detection
    # to extract a 28x28 cropped image of the main object
    #
    x_task_raw = []
    x_task_extracted = []
    for i in range(20):

        # Read the task images
        image_filename = "task_images/task_image_%d.jpg" % (i+1)
        x_task_large_i = cv2.imread(image_filename)
        x_task_large_i = cv2.cvtColor(x_task_large_i, cv2.COLOR_BGR2GRAY)  # -> grayscale
        x_task_raw.append(x_task_large_i)
        if 0:
            plot_img(x_task_large_i, "Task image", 1)

        # Use opencv edge/contour detection to extract relevant 28x28 crops of main object
        # The method is to find max/min of contour points, and find the mid point of this for the box center.
        # From there we crop the 28 x 28 image
        extracted_img_i = extract_obj_28by28(image_filename)
        x_task_extracted.append(extracted_img_i)
        if 0:
            plot_img(extracted_img_i, "Extracted task image", 1)

    x_task = np.array(x_task_extracted.copy())  # convert extracted version to numpy array (should already by numpy)
    x_task_raw = np.array(x_task_raw)
    x_task_original = x_task.copy()  # copy original just in case

    x_task = convert_img2norm(x_task, ypixels, xpixels)  # normalize and reshape

    if 0:
        # Print an example to familiarize with extracting from the raw task image
        fignum = 1
        fignum = plot_img(x_task_raw[0], "Task Image Raw", fignum)
        fignum = plot_img(x_task_extracted[0], "Task Image Extracted", fignum)
        sys.exit()

    #
    # Convert class vectors to binary class matrices (categorical encoding)
    # This is for CNN (not CAE as that is unsupervised)
    #
    y_train = keras.utils.to_categorical(y_train, n_categories)
    y_test = keras.utils.to_categorical(y_test, n_categories)

    # Visualize the noisy data set for debugging
    # For debugging purposes, we check 10 noisy test images
    if check_noisydata:
        plot_unlabeled_images_random(x_test_noisy, 10, "Noisy test images", ypixels, xpixels)
    # For debugging purposes, we check 10 original training images
    if check_original_training:
        plot_labeled_images_random(x_train_original, y_train_original, categories, 10, "Original training images", ypixels, xpixels)
    # For debugging purposes, we check 10 original test images
    if check_original_test:
        plot_labeled_images_random(x_test_original, y_test_original, categories, 10, "Original test images", ypixels, xpixels)


    # =================================================
    #
    # Training (and setting up neural network layers, optimizer, loss function)
    # 1) We train the convolutional autoencoder to denoise images
    # 2) Using the denoised images, we predict the class of the denoised image by
    #    training a convolutional neural network
    #
    # =================================================
    input_shape = (ypixels, xpixels, 1)  # our data format for the input layer of our NNs

    # ==================================================
    # Train the CAE to denoise the images
    # ==================================================
    if use_CAE and use_saved_CAE_model:
        # Load previously saved trained autoencoder model
        autoencoder = load_model('models/CAE_trained.h5')

    elif use_CAE:

        n_epochs = n_epochs_CAE
        batch_size = batch_size_CAE

        # Build convolutional auto encoder layers
        input_img = Input(shape=input_shape)
        encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)  # this is the final encoding layer -> repr (7, 7, 32)

        decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

        # Set autoencoder, choose optimizer, and choose loss function
        # We use:
        #  adadelta optimization -> fast convergence properties in general NN
        #  binary cross entropy loss -> penalize when estimated class prob hits wrong target classes
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        # Train the convolutional autoencoder to denoise samples
        # Takes noisy data as input and clean data output
        autoencoder.fit(x_train_noisy, x_train,
                        epochs = n_epochs,
                        batch_size = batch_size,
                        shuffle = True,
                        validation_data = (x_test_noisy, x_test),
                        callbacks = [TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

        # Save trained autoencoder model
        autoencoder.save('models/CAE_trained.h5')  # creates a HDF5 file

    #
    # Visualization: plot example test reconstructions of the trained encoding/decoding
    #
    decoded_noisy_test_imgs = autoencoder.predict(x_test_noisy)
    plot_compare(x_test_noisy, decoded_noisy_test_imgs)

    #
    # Denoise noisy task images and plot
    #
    if 1:
        denoised_task_imgs = autoencoder.predict(x_task)
        plot_unlabeled_images_random(denoised_task_imgs, 5, "Denoising extracted task images", ypixels, xpixels)
        x_task = denoised_task_imgs

    # ==================================================
    # Train the CNN to classify the denoised images
    # ==================================================
    if use_CNN and use_saved_CNN_model:
        # Load previously saved trained CNN model
        model = load_model('models/CNN_trained.h5')

    elif use_CNN:

        batch_size = batch_size_CNN
        n_epochs = n_epochs_CNN

        # Build our CNN mode layer-by-layer
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_categories, activation='softmax'))

        # Set our optimizer and loss function (similar settings to our CAE approach)
        model.compile(loss = keras.losses.categorical_crossentropy,
                      optimizer = keras.optimizers.Adadelta(),
                      metrics = ['accuracy'])

        # Train our CNN
        model.fit(x_train, y_train,
                  batch_size = batch_size,
                  epochs = n_epochs,
                  verbose = 1,
                  validation_data = (x_test, y_test))

        # Save trained CNN model
        model.save('models/CNN_trained.h5')  # creates a HDF5 file

        # Evaluate our model test loss/accuracy
        score = model.evaluate(x_test, y_test, verbose=0)
        print('CNN classifier test loss:', score[0])
        print('CNN classifier test accuracy:', score[1])

    #
    # Visualization: print 10 randomly selected test images and their classifications
    # Note that we kept original test data sets for the purpose of printing here
    #
    if 1:
        x_test_plot = x_test_original.copy()
        x_test_plot = np.array(x_test_plot).reshape((len(x_test_plot), 28, 28, 1))  # reshape
        y_test_plot_pred = model.predict_classes(x_test_plot)  # predict the class index (integer)
        print("Plotting test predictions")
        plot_labeled_images_random(x_test_original, y_test_plot_pred, categories, 5,
                                   "Classifying test images", ypixels, xpixels)

    #
    # Classify denoised task images and plot it
    #
    if 1:
        x_task_plot = x_task.copy()
        x_task_plot = np.array(x_task_plot).reshape((len(x_task_plot), 28, 28, 1))  # reshape
        y_task_plot_pred = model.predict_classes(x_task_plot)  # predict the class index (integer)
        print("Plotting extracted task predictions")
        plot_labeled_images_random(x_task_original, y_task_plot_pred, categories, 5,
                                   "Classifying extracted task images", ypixels, xpixels)
        print("Plotting raw task predictions")
        plot_labeled_images_random(x_task_raw, y_task_plot_pred, categories, 5,
                                   "Classifying raw task images", ypixels_task, xpixels_task)



# ===============================================
#
# Side functions
#
# ===============================================
# extract_obj_28by28: extract the main 28 x 28 object square (highest number of contour points) from given image
def extract_obj_28by28(image_filename):
    ypixels = 4 * 28
    xpixels = 4 * 28
    ypixels_extract = 28
    xpixels_extract = 28

    image = cv2.imread(image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # -> grayscale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # blur image
    edged = cv2.Canny(gray, 30, 200)  # find edges using Canny edge detection algorithm
    im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)  # find edged image contours

    y = []
    x = []
    n_contourpts = -1
    objects = contours
    for i, obj in enumerate(objects):
        # print("%d) Contour points = %d" % (i,len(obj)))
        if len(obj) > n_contourpts:
            n_contourpts = len(obj)
            x = []
            y = []
            for pt in obj:
                point = pt[0]
                y.append(point[1])
                x.append(point[0])

    if 0:
        plt.plot(x, [ypixels - alpha for alpha in y], 'ro')
        plt.axis([0, xpixels, 0, ypixels])
        plt.show()

    #
    # Extract 28 x 28 image corners and centers
    #
    x_mean = min(x) + (max(x) - min(x)) / 2  # find the max-min center
    y_mean = min(y) + (max(y) - min(y)) / 2  # find the max-min center
    x_mean_int = int(x_mean)
    y_mean_int = int(y_mean)

    if 0:
        print("mean point = [%f, %f]" % (y_mean, x_mean))
        print("mean point (int) = [%f, %f]" % (y_mean_int, x_mean_int))

    extract_x = [int(x_mean_int - xpixels_extract / 2),
                 int(x_mean_int + xpixels_extract / 2)]  # left right boundaries of x extraction
    extract_y = [int(y_mean_int - ypixels_extract / 2),
                 int(y_mean_int + ypixels_extract / 2)]  # left right boundaries of y extraction

    min_extract_x = min(extract_x)
    min_extract_y = min(extract_y)
    max_extract_x = max(extract_x)
    max_extract_y = max(extract_y)
    if min_extract_x < 0:
        for i, alpha in enumerate(extract_x):
            extract_x[i] = extract_x[i] + abs(min_extract_x - 0)
    if min_extract_y < 0:
        for i, alpha in enumerate(extract_x):
            extract_y[i] = extract_y[i] + abs(min_extract_y - 0)
    if xpixels - max_extract_x < 0:
        for i, alpha in enumerate(extract_x):
            extract_x[i] = extract_x[i] - abs(max_extract_x - xpixels)
    if ypixels - max_extract_y < 0:
        for i, alpha in enumerate(extract_y):
            extract_y[i] = extract_y[i] - abs(max_extract_y - ypixels)

    if 0:
        print("Extracting x:", extract_x)
        print("Extracting y:", extract_y)

    xplot = [extract_x[0], extract_x[0], extract_x[1], extract_x[1]]
    yplot = [extract_y[0], extract_y[1], extract_y[0], extract_y[1]]
    if 0:
        plt.plot(xplot, [ypixels - alpha for alpha in yplot], 'ro')
        plt.axis([0, xpixels, 0, ypixels])
        plt.show()

    #
    # Extract the image
    #
    extract_img = gray
    extract_img = extract_img[extract_y[0]:extract_y[1], extract_x[0]:extract_x[1]]

    return extract_img

# embed_randomly_in_largebackground: used in creating task images, makes a clean image noisy and embeds it in a noisy large background
def embed_randomly_in_largebackground(x_task, noise_factor, bg_noise_factor):
    object_embedded = x_task.copy()

    # We make no background noise (simplification)
    background_shape = (x_task.shape[0], 4 * x_task.shape[1], 4 * x_task.shape[2])
    background_noise = bg_noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=background_shape) + 1) / 2

    object = object_embedded
    object_shape = object.shape
    object = object + noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=object_shape) + 1) / 2
    object_noisy = np.clip(object, 0., 255.)

    dy = background_shape[1] - object_shape[1]
    dx = background_shape[2] - object_shape[2]

    for i in range(object_noisy.shape[0]):
        x_offset = np.random.choice(dx)  # positive offset from left of image
        y_offset = np.random.choice(dy)  # negative offset from top of image
        background_noise[i, y_offset:y_offset + object_noisy.shape[1], x_offset:x_offset + object_noisy.shape[2]] = object_noisy[i]

    return background_noise

# convert_img2norm: converts image list to a normed image list (used as input for NN)
def convert_img2norm(img_list, ypixels, xpixels):
    norm_list = img_list.copy()
    norm_list = norm_list.astype('float32') / 255
    norm_list = np.reshape(norm_list, (len(norm_list), ypixels, xpixels, 1))
    return norm_list

# convert_norm2img: convers normed image list to image list (used for plotting visualization)
def convert_norm2img(norm_list, ypixels, xpixels):
    img_list = norm_list.copy()
    img_list = np.reshape(img_list, (len(img_list), ypixels, xpixels))
    img_list = (img_list * 255.).astype('float32')
    return img_list

# plot_labeled_images_random: plots labeled images at random
def plot_labeled_images_random(image_list, label_list, categories, n, title_str, ypixels, xpixels):
        index_sample = np.random.choice(len(image_list), n)
        plt.figure(figsize=(2*n, 2))
        #plt.suptitle(title_str)
        for i, ind in enumerate(index_sample):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(image_list[ind].reshape(ypixels, xpixels))
            plt.gray()
            ax.set_title("pred: " + categories[label_list[ind]])
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        plt.show()

# plot_unlabeled_images_random: plots unlabeled images at random
def plot_unlabeled_images_random(image_list, n, title_str, ypixels, xpixels):
        index_sample = np.random.choice(len(image_list), n)
        plt.figure(figsize=(2*n, 2))
        plt.suptitle(title_str)
        for i, ind in enumerate(index_sample):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(image_list[ind].reshape(ypixels, xpixels))
            plt.gray()
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        plt.show()

# plot_compare: given test images and their reconstruction, we plot them for visual comparison
def plot_compare(x_test, decoded_imgs):
    n = 10
    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# plot_img: plots greyscale image
def plot_img(img, title_str, fignum):
    plt.plot(fignum), plt.imshow(img, cmap='gray')
    plt.title(title_str), plt.xticks([]), plt.yticks([])
    fignum += 1  # move onto next figure number
    plt.show()
    return fignum

#
# Driver file
#
if __name__ == '__main__':
    main()
