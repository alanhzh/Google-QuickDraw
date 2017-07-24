"""

 QuickDraw_noisy_classifier.py (author: Anson Wong / git: ankonzoid)
 
 Given a set of Gaussian-noisy sketch greyscale images of size 28x28 pixels taken
 from Google's QuickDraw data set:

 https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

 We create a model that denoise the image, then classifies the image object.

 The main steps:
  1) Create/collect query images for classification, and clean images for training
  2) Train a convolutional autoencoder to learn how to denoise the image   
  3) Train a convolutional neural network to classify the denoised image
  
"""
import os, scipy.misc, pylab, random
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.layers import Input, UpSampling2D  # CAE
from keras.models import Model  # CAE
from keras.callbacks import TensorBoard  # CAE
from keras.models import Sequential  # CNN
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # CNN
from keras.models import load_model  # save keras model

def main():

    # =======================================================
    #
    # Manually set user parameters
    #
    # =======================================================
    data_dir = "/Users/ansonwong/Desktop/training_heaven/QuickDraw"
    categories = ['cat', 'cup', 'fish', 'jacket', 'pineapple']  # sets item/animal categories


    # =======================================================
    # Automatically set remainder parameters
    # =======================================================
    xpixels = 28  # set x pixel numbers for query/training/test examples
    ypixels = 28  # set y pixel numbers for query/training/test examples
    noise_factor = 0.5  # how much gaussian noise to add to our noisy images [0,1]

    CAE_model_filename = 'models/convAE_trained.h5'  # save/load conv AE model filename
    n_epochs_CAE = 20  # number of epochs for CAE training (20)
    batch_size_CAE = 512  # batch size of CAE training

    CNN_model_filename = 'models/convNN_trained.h5'  # save/load conv NN model filename
    n_epochs_CNN = 5  # number of epochs for CNN training (5)
    batch_size_CNN = 512  # batch size of CNN training

    n_query_category = 2  # number of query images to take from each category
    seed = 101


    # =======================================================
    #
    # Create query images
    # Noise up some clean Google QuickDraw images
    #
    # =======================================================
    check_noisydata = False  # check noisy data before run?
    check_original_training = False  # check original training images before run?
    check_original_test = False  # check original test images before run?

    category_filenames = []
    for catname in categories:
        filename = os.path.join(data_dir, "full%2Fnumpy_bitmap%2F" + catname + ".npy")
        category_filenames.append(filename)
    
    # Read data and extract some for query data
    x_query_create = []
    n_remaining_category = []
    for i_category, category in enumerate(categories):
        data = np.load(category_filenames[i_category])
        n_total = len(data)
        print("Reading data for category index {0}/{1}: '{2}' (shape = {3})".format(
            i_category+1, len(categories), category, data.shape))
        for j in range(n_query_category):
            img = np.array(data[j]).reshape((ypixels, xpixels))
            x_query_create.append(img)
        n_remaining_category.append(n_total-n_query_category)
    x_query_create = np.array(x_query_create)


    # Set number of training and test data
    n_take_train = min([8000, min(n_remaining_category)])  # number of training images to take from each category
    n_take_test = min([1600, min(n_remaining_category)])  # number of test images to take from each category
    print("n_take_train = {0}".format(n_take_train))
    print("n_take_test = {0}".format(n_take_test))

    # Add noise to our greyscale image
    x_query_create_final = add_noise(x_query_create, noise_factor)

    for i in range(n_query_category*len(categories)):
        print("Printing query image %d to file..." % (i+1))
        scipy.misc.imsave("query/query_image_%d.jpg" % (i + 1), x_query_create_final[i])


    # =======================================================
    #
    # Data I/O and Preprocessing
    # Read QuickDraw image data and append them to the training/test sets.
    # We take original copies of it, and also create noisy copies of them too.
    #
    # =======================================================

    #
    # Read clean training/test images
    #
    n_categories = len(categories)  # number of classes
    x_train = []; y_train = []  # holds training images/labels
    x_test = []; y_test = []  # holds test images/labels
    for i_category, category in enumerate(categories):
        
        data = np.load(category_filenames[i_category])
        data = data[n_query_category:]  # omit the query images extracted earlier

        n_data = len(data)
        print("[%d/%d] Reading category index %d: '%s' (%d images: take %d training, take %d test)" %
              (i_category+1, len(categories), i_category, category, n_data, n_take_train, n_take_test))
        
        for j, data_j in enumerate(data):
            img = np.array(data_j).reshape((ypixels, xpixels))
            if j < n_take_train:
                x_train.append(img); y_train.append(i_category)  # append to training set
            elif j - n_take_train < n_take_test:
                x_test.append(img); y_test.append(i_category)  # append to test set
            else:
                break

    # Convert to numpy
    x_train = np.array(x_train); y_train = np.array(y_train)  # convert to numpy arrays
    x_test = np.array(x_test); y_test = np.array(y_test)  # convert to numpy arrays
    x_train_original = x_train.copy(); y_train_original = y_train.copy()  # make original untouched copies
    x_test_original = x_test.copy(); y_test_original = y_test.copy()  # make original untouched copies

    # Create noisy copies of our image data sets by adding gaussian noise
    x_train_noisy = x_train + noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) + 1) / 2
    x_test_noisy = x_test + noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) + 1) / 2
    x_train_noisy = np.clip(x_train_noisy, 0., 255.)
    x_test_noisy = np.clip(x_test_noisy, 0., 255.)

    # Convert our greyscaled image data sets to have values [0,1] and reshape to form (n, ypixels, xpixels, 1)
    x_train = convert_img2norm(x_train, ypixels, xpixels)
    x_test = convert_img2norm(x_test, ypixels, xpixels)
    x_train_noisy = convert_img2norm(x_train_noisy, ypixels, xpixels)
    x_test_noisy = convert_img2norm(x_test_noisy, ypixels, xpixels)


    #
    # Read query images
    #
    x_query_extracted = []
    for i in range(n_query_category*len(categories)):

        # Read the query images
        query_image_filename = "query/query_image_%d.jpg" % (i+1)
        query_img_i = read_img(query_image_filename, gray_scale=True)
        x_query_extracted.append(query_img_i)

    x_query = np.array(x_query_extracted.copy())  # convert extracted version to numpy array (should already by numpy)
    x_query_original = x_query.copy()  # copy original just in case

    x_query = convert_img2norm(x_query, ypixels, xpixels)  # normalize and reshape

    #
    # Convert class vectors to binary class matrices (categorical encoding)
    # This is for CNN (not CAE as that is unsupervised)
    #
    y_train = keras.utils.to_categorical(y_train, n_categories)
    y_test = keras.utils.to_categorical(y_test, n_categories)

    # Visualize the noisy data set for debugging
    # For debugging purposes, we check 10 noisy test images
    if check_noisydata:
        plot_unlabeled_images_random(x_test_noisy, 10, "Noisy test images",
                                     ypixels, xpixels, seed, "answer/noisy_test_imgs.png")
    # For debugging purposes, we check 10 original training images
    if check_original_training:
        plot_labeled_images_random(x_train_original, y_train_original, categories, 10, "Original training images",
                                   ypixels, xpixels, seed, "answer/original_train_imgs.png")
    # For debugging purposes, we check 10 original test images
    if check_original_test:
        plot_labeled_images_random(x_test_original, y_test_original, categories, 10, "Original test images",
                                   ypixels, xpixels, seed, "answer/original_test_imgs.png")


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
    if os.path.isfile(CAE_model_filename):

        autoencoder = load_model(CAE_model_filename)  # load saved model

    else:

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
        autoencoder.compile(optimizer='adadelta',
                            loss='binary_crossentropy')

        # Train the convolutional autoencoder to denoise samples
        # Takes noisy data as input and clean data output
        autoencoder.fit(x_train_noisy, x_train,
                        epochs = n_epochs_CAE,
                        batch_size = batch_size_CAE,
                        shuffle = True,
                        validation_data = (x_test_noisy, x_test),
                        callbacks = [TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

        # Save trained autoencoder model
        autoencoder.save(CAE_model_filename)  # creates a HDF5 file

    #
    # Denoise noisy query images and plot
    #
    denoised_query_imgs = autoencoder.predict(x_query)
    x_query = denoised_query_imgs  # set as the query images now


    # ==================================================
    # Train the CNN to classify the denoised images
    # ==================================================
    if os.path.isfile(CNN_model_filename):

        cnn = load_model(CNN_model_filename)  # load saved model

    else:

        # Build our CNN mode layer-by-layer
        cnn = Sequential()
        cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        cnn.add(Conv2D(64, (3, 3), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())
        cnn.add(Dense(128, activation='relu'))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(n_categories, activation='softmax'))

        # Set our optimizer and loss function (similar settings to our CAE approach)
        cnn.compile(loss = keras.losses.categorical_crossentropy,
                    optimizer = keras.optimizers.Adadelta(),
                    metrics = ['accuracy'])

        # Train our CNN
        cnn.fit(x_train, y_train,
                batch_size = batch_size_CNN,
                epochs = n_epochs_CNN,
                verbose = 1,
                validation_data = (x_test, y_test))

        # Save trained CNN model
        cnn.save(CNN_model_filename)  # creates a HDF5 file

    # Evaluate our model test loss/accuracy
    score = cnn.evaluate(x_test, y_test, verbose=0)
    print('CNN classification test loss: {0}'.format(score[0]))
    print('CNN classification test accuracy: {0}%'.format(score[1]))

    #
    # Visualization: print 10 randomly selected query images and their classifications
    # Note that we kept original test data sets for the purpose of printing here
    #
    if 1:
        x_query_plot = x_query.copy()
        x_query_plot = np.array(x_query_plot).reshape((len(x_query_plot), ypixels, xpixels, 1))  # reshape
        y_query_plot_pred = cnn.predict_classes(x_query_plot)  # predict the class index (integer)

        print("Plotting query predictions")

        # original noisy query images
        plot_unlabeled_images_random(x_query_original, 10,
                                     "", ypixels, xpixels, seed,
                                     "answer/query.png")

        # use CAE: noisy image -> clean image
        plot_unlabeled_images_random(denoised_query_imgs, 10,
                                     "", ypixels, xpixels, seed,
                                     "answer/CAE_result.png")

        # use CNN: denoised image -> predict label
        plot_labeled_images_random(x_query, y_query_plot_pred, categories, 10,
                                   "Classifying extracted query images", ypixels, xpixels, seed,
                                   "answer/CNN_result.png")

        # main result: noisy query images -> predict label
        plot_labeled_images_random(x_query_original, y_query_plot_pred, categories, 10,
                                   "Classifying extracted query images", ypixels, xpixels, seed,
                                   "answer/MAIN_result.png")



# ===============================================
#
# Side functions
#
# ===============================================


# converts clean image to a noisy one (keeps image size the same)
def add_noise(x_clean, noise_factor):
    x = x_clean.copy()
    x_shape = x.shape
    x = x + noise_factor * 255 * (np.random.normal(loc=0.0, scale=1.0, size=x_shape) + 1) / 2
    x_noisy = np.clip(x, 0., 255.)
    return x_noisy

# converts image list to a normed image list (used as input for NN)
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
def plot_labeled_images_random(image_list, label_list, categories, n, title_str, ypixels, xpixels, seed, filename):
    random.seed(seed)
    index_sample = random.sample(range(len(image_list)), n)
    plt.figure(figsize=(2*n, 2))
    #plt.suptitle(title_str)
    for i, ind in enumerate(index_sample):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(image_list[ind].reshape(ypixels, xpixels))
        plt.gray()
        ax.set_title(categories[label_list[ind]], fontsize=20)
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    if 1:
        pylab.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

# plot_unlabeled_images_random: plots unlabeled images at random
def plot_unlabeled_images_random(image_list, n, title_str, ypixels, xpixels, seed, filename):
    random.seed(seed)
    index_sample = random.sample(range(len(image_list)), n)
    plt.figure(figsize=(2*n, 2))
    plt.suptitle(title_str)
    for i, ind in enumerate(index_sample):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(image_list[ind].reshape(ypixels, xpixels))
        plt.gray()
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    if 1:
        pylab.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

# plot_compare: given test images and their reconstruction, we plot them for visual comparison
def plot_compare(x_test, decoded_imgs, filename):
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

    if 1:
        pylab.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

# plot_img: plots greyscale image
def plot_img(img, title_str, fignum):
    plt.plot(fignum), plt.imshow(img, cmap='gray')
    plt.title(title_str), plt.xticks([]), plt.yticks([])
    fignum += 1  # move onto next figure number
    plt.show()
    return fignum

# read image
def read_img(img_filename, gray_scale=False):
    img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
    return img

#
# Driver file
#
if __name__ == '__main__':
    main()
