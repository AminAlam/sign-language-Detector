# sign-language-Detector

### What is this?
This is a ML model which can detect sing language from hand gestures(real time detection). This model has been made by tensoeflow and is trained by 30000 images. It was one of the home works of SUT Deep Learning course.

### Dependencies
* Tensorflow(v1)
* sklearn.metrics.confusion_matrix
* sklearn.preprocessing.OneHotEncoder
* NumPy
* Matplotlib
* cv2
* csv
* random
* scipy.ndimage.interpolation.rotate


### How to use it?
Download the `RealTimeDetection.py` and the model foder(`89P_ExtendedDataset`). Put both of them in same directory. Run `RealTimeDetection.py`.


### Data Augmentation
I have augmented the training set using rotation and a random black box on the images.

| Image Type  | Representaion|
| ------------- |-------------|
| Original Image  | ![image 1](http://ee.sharif.edu/~amin/static/Deep/sample.png)|
| Rotated Image  | ![image 2](http://ee.sharif.edu/~amin/static/Deep/rotated_sample.png)|
| Rotated and Cropped image  | ![image 3](http://ee.sharif.edu/~amin/static/Deep/croped_sample.png)|



### Some Test Images
| ![image 1](http://ee.sharif.edu/~amin/static/Deep/Deep_01.png)  | ![image 2](http://ee.sharif.edu/~amin/static/Deep/Deep_02.png)|
| ------------- |-------------|
| ![image 3](http://ee.sharif.edu/~amin/static/Deep/Deep_03.png)  | ![image 4](http://ee.sharif.edu/~amin/static/Deep/Deep_04.png)|
