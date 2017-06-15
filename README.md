# GoogleSketch 

Given a set of task images of a noisy Google SketchRNN object embedded in a 112 x 112 pixel black background, we classify the object drawn in the noisy sketch. An example of 5 task images being classified by our trained model 

![My image](ankonzoid.github.com/GoogleSketch/results/MAIN_RESULT.png)

The steps taken by our model are:

1) Use OpenCV for edge/object detection of the 112 x 112 image to detect the noise sketch box
2) Denoise the noisy image using a convolutional autoencoder
3) Classify the denoised image using a convolutional neural networks

Note that we only incorporate 5 classes in the training (specifically sketch labels: car, fish, cat, coffee cup, cookie). There are couple hundred more class labels one can expand to, but for the sake of simplicity, we only classify for these 5 objects.

Author: Anson Wong (ankonzoid)
