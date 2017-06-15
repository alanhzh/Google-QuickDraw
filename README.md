# GoogleSketch 

Given a set of task images of a noisy Google SketchRNN object embedded in a 112 x 112 pixel black background, we classify the object drawn in the noisy sketch.

The procedure of the algorithm is to:

1) Use OpenCV for edge/object detection of the 112 x 112 image to detect the noise sketch box
2) Denoise the noisy image using a convolutional autoencoder
3) Classify the denoised image using a convolutional neural networks

Note that we only incorporate 5 classes in the training (specifically sketch labels: car, fish, cat, coffee cup, cookie). There are couple hundred more class labels one can expand to, but for the sake of simplicity, we only classify for these 5 objects.

Author: Anson Wong (ankonzoid)
