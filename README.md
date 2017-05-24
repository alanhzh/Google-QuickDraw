# GoogleSketch

Given a set of task images that are large 112 x 112 images with a noisy object image in it, we use Google SketchRNN data set of labeled clean animals/objects as training data to classify what the noisy object is in the task images. The tools use (in chronological order) is opencv for edge/object detection, convolutional autoencoders for denoising, and convolutional neural networks for classification.
