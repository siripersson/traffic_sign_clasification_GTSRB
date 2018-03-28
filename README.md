# traffic_sign_clasification_GTSRB
Application of the German Traffic Sign Recognition Benchmark (GTSRB) on two Convolutional Neural Networks(VGG16 and InceptionV3) using a Python deep learning library called Keras

## Configuration
All the files have to be in the same folder. To start the training process write `python fine_tune_Inception.py` to train the InceptionV3 model and `python fine_tune_VGG16.py`. Each script will also run the corresponding preprocessing script: load_GTSRB_data_for_Inception.py and load_GTSRB_data_for_VGG16.py which preprocesses the images from the dataset.

### Prerequisites 
- Python 3.5
- SciPy with NumPy
- Matplotlib
- Keras
- Theano/Tensorflow

- Also, NVIDIA CUDA needs to be installe and make sure to use GPU since makes the training process faster.

