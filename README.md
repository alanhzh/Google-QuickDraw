# Google-QuickDraw

Given a set of query images of a noisy Google QuickDraw object, we classify the object drawn in the noisy sketch. An example classification of 10 of the described task images being classified by our trained model 

![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png)

<img src="https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/MAIN_result.png" width="40%">

The steps taken explicitly by our model are:

1) Create/extract query images
![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/query.png)

2) Denoise the noisy query images using a convolutional autoencoder (convAE)
![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CAE_result.png)

3) Classify the denoised query images using a convolutional neural networks (convNN)
![My image](https://github.com/ankonzoid/Google-QuickDraw/blob/master/answer/CNN_result.png)

Note that we only incorporate 5 classes in the training (specifically sketch labels: car, fish, cat, coffee cup, cookie). There are couple hundred more class labels one can expand to, but for the sake of simplicity, we only classify for these 5 objects.


Usage:
> python QuickDraw_noisy_classifier.py