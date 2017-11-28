import cv2
import numpy as np
import hickle as hkl
import pickle as pkl
import os
import keras
from keras import backend as K
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
import os
import glob
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot as plt

NUM_CLASSES = 43
IMG_SIZE = 224 #48

nb_train_samples = 39209 # 3000 training samples
nb_valid_samples = 12630 # 100 validation samples
num_classes = 43
sample_frac=0.3

plt.style.use('ggplot')

# Plot the traffic sign distribution
def show_class_distribution(classIDs, title):
    plt.figure(figsize=(15,5))
    #plt.title('Class ID distribution for ', title)
    plt.hist(classIDs, bins=num_classes, edgecolor='black')
    plt.show()
    
############################################################
def preprocess_img(img):
    #cv2.imshow('image', img) # display an image
   # cv2.imwrite('img_before.png', img)
   
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # rescale to standard size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    #cv2.imshow('image2', img) # display an image
    #cv2.imwrite('img_after.png', img*255) # save image OBS! converting image to uint8 you need to multiply by 255 to get the correct range
    #
    cv2.waitKey(0)

    return img


def get_class(img_path):
    return int(img_path.split('\\')[-2])
############################################################


def load_GTSRB_data_1(img_rows, img_cols):

    # For training
    print("For training")
    root_dir = r'D:\GTSRB\Final_Training\Images'
    imgs = []
    labels = []

    all_img_paths_before = glob.glob(os.path.join(root_dir, '*/*.ppm'))
   # print("all image paths: ", all_img_paths_before)
    np.random.shuffle(all_img_paths_before)
    all_img_paths=[]
    for i in range(len(all_img_paths_before)):
        if i%int(1/sample_frac)==0:
            all_img_paths.append(all_img_paths_before[i])
            
   # print("all image paths: ", all_img_paths)        
    print("Number of files: ", len(all_img_paths))
    
    for img_path in all_img_paths:
        try:
            img = preprocess_img(cv2.imread(img_path,1)) # load image with color
            
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass
    print("imgs: ", len(imgs))
    print("labels: ", len(labels))
    
    X_train= np.array(imgs, dtype='float32')
    print("klar med X_train")
    Y_train = np.array(labels)
    print("Klar med Y_train")
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    print("Klar med reshaping av Y_train")    

    
    # For validation
    print("For validation")
    root_dir2 = r'D:\GTSRB\Final_Test\Images'
    imgs2 = []
    labels2 = []

    test_before = pd.read_csv(r'D:\GTSRB\GT-final_test.csv',sep=';') # test answers
    #show_class_distribution(test_before['ClassId'], ' Train data') # plot distribution
   # print("Test_before: ", test_before)
    test=test_before.sample(frac=sample_frac)
   # print("Test: ", test)
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        
        img_path = os.path.join(r'D:\GTSRB\Final_Test\Images',file_name)
        imgs2.append(preprocess_img(cv2.imread(img_path)))
        labels2.append(class_id)
    print("imgs2: ", len(imgs2))
    print("labels2: ", len(labels2))
    
    print("creating X_valid")
    X_valid = np.array(imgs2)
    print("creating Y_valid")
    Y_valid = np.array(labels2)
    print("reshaping y_valid")
    Y_valid = Y_valid.reshape(Y_valid.shape[0],1) # la till detta för att annars hamnade den på fel format

    print("After")
    print("Length of X_valid: ", len(X_valid))
    print("Length of Y_valid: ", len(Y_valid))
    print("Length of X_train: ", len(X_train))
    print("Length of Y_train: ", len(Y_train))
    
    print("Transposing")
    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([img.transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([img.transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    Y_test=Y_valid # to confusion matrix

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    print("Length of X_valid: ", len(X_valid))
    print("Length of Y_valid: ", len(Y_valid))
    print("Length of X_train: ", len(X_train))
    print("Length of Y_train: ", len(Y_train))

    print( "Type X_train: " , type(X_train))
    print( "Type X_valid: " , type(X_valid))
    print( "Type Y_train: " , type(Y_train))
    print( "Type Y_valid: " , type(Y_valid))

    print( "Shape X_train: " , X_train.shape)
    print( "Shape X_valid: " , X_valid.shape)
    print( "Shape Y_train: " , Y_train.shape)
    print( "Shape Y_valid: " , Y_valid.shape)

    # Switch RGB to BGR order
    X_train=X_train[:, ::-1, :, :]
    X_valid=X_valid[:, ::-1, :, :]

    print("Subracting mean pixel")
    m0, m1, m2 = np.mean(X_train[:, 0, :, :]), np.mean(X_train[:, 1, :, :]), np.mean(X_train[:, 2, :, :])
    
    # Subract ImageNet mean pixel
    print("Subract X_train m0")
    X_train[:, 0, :, :] -= m0
    print("Subract X_train m1")
    X_train[:, 1, :, :] -= m1
    print("Subract X_train m2")
    X_train[:, 2, :, :] -= m2
   
    print("Subract X_valid m0")
    X_valid[:, 0, :, :] -= m0
    print("Subract X_valid m1")
    X_valid[:, 1, :, :] -= m1
    print("Subract X_valid m2")
    X_valid[:, 2, :, :] -= m2
    
    print("After reshaping")
    print( "Shape X_train: " , X_train.shape)
    print( "Shape X_valid: " , X_valid.shape)
    print( "Shape Y_train: " , Y_train.shape)
    print( "Shape Y_valid: " , Y_valid.shape)

    print("Done!")
        

    return X_train, Y_train, X_valid, Y_valid, Y_test
   
