# Google-QuickDraw

Given a set of task images of a noisy Google QuickDraw object embedded in a 112 x 112 pixel black background, we classify the object drawn in the noisy sketch. An example classification of 5 of the described task images being classified by our trained model 

![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/results/MAIN_RESULT.png)

The steps taken explicitly by our model are:

1) Use OpenCV for edge/object detection of the 112 x 112 image to detect the noise sketch box

2) Denoise the noisy image using a convolutional autoencoder (CAE)
![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/results/CAE_training.png)

3) Classify the denoised (clean) image using a convolutional neural networks (CNN)
![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/results/CNN_training.png)

Note that we only incorporate 5 classes in the training (specifically sketch labels: car, fish, cat, coffee cup, cookie). There are couple hundred more class labels one can expand to, but for the sake of simplicity, we only classify for these 5 objects.

Usage:

...


Author: Anson Wong (ankonzoid)
