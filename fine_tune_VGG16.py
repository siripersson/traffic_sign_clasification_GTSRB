# Taken from: https://github.com/flyyufelix/cnn_finetune/blob/master/vgg16.py 
# based on: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 

# -*- coding: utf-8 -*-
import keras
import itertools
import sys
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from new_load_GTSRB_VGG16 import load_GTSRB_data_1

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

# Variables to run the script with a bat-script
dropout_rate= float(sys.argv[1])#0.5
lr= float(sys.argv[2] )#1e-3
batch_size= int(sys.argv[3])#10
weights_filename= 'vgg16_weights_frac_0_3_lr_' + str(lr) +'_batch'+str(batch_size)+'_drop_'+str(dropout_rate)+'_epochs_30.h5'
matrix_filename= 'conf_matrix_frac_0_3_lr_' + str(lr) +'_batch'+str(batch_size)+'_drop_'+str(dropout_rate)+'_epochs_30.png'
log_filename='log_frac_0_3_lr_' + str(lr) +'_batch'+str(batch_size)+'_drop_'+str(dropout_rate)+'_epochs_30'
result_file='result_frac_0_3_lr_' + str(lr) +'_batch'+str(batch_size)+'_drop_'+str(dropout_rate)+'_epochs_30.txt'
   
def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    #VGG 16 Model for Keras
   # Model Schema is based on 
   # https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
   # ImageNet Pretrained Weights 
   # https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
   # Parameters:
    #  img_rows, img_cols - resolution of inputs
    #  channel - 1 for grayscale, 3 for color 
     # num_classes - number of categories for our classification task
        
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('imagenet_models/vgg16_weights_th_dim_ordering_th_kernels.h5')
   
    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:10]:
        layer.trainable = False

    # Learning rate is changed 
    sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def lr_schedule(epoch): #  function that takes an epoch index as input and returns a new learning rate as output
    return lr*(0.1**int(epoch/10))

if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 43  
    #batch_size = 15#10#40# Tidigare har all körts med 20 
    nb_epoch = 30 #10

    # Load data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid, Y_test = load_GTSRB_data_1(img_rows, img_cols)

    # Load our model
    print("loading model")
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    csv_logger=CSVLogger(log_filename) # callback that streams epoch results to a csv file

    print("Start fine tuning")
    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split=0.2, # fraction of the data held-out for validation
              callbacks=[LearningRateScheduler(lr_schedule),history, csv_logger,
                ModelCheckpoint(weights_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
    
              )
    #ModelCheckpoint('vgg16_weights.{epoch:02d}-{val_loss:.2f}.h5',
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    
    #Get history of accuracy and plot it
   # print("history acc: ",history.acc)
   # print(" history acc type: ", type(history.acc))
   # np.save('history_acc_vgg16', history.acc)
    #plt.plot(range(1,nb_epoch+1), history.acc)
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.title("VGG16")
    #plt.show()

    y_pred= model.predict_classes(X_valid)
    print("Predictions: ", y_pred)
    model.metrics_names
    
    y_eval=model.evaluate(X_valid,Y_valid)
    print("Evaluation: ", y_eval)

    f=open(result_file, 'w')
    f.write('Y_pred: ' + str(y_pred) )
    f.write('Y_eval: ' + str(y_eval))
    f.close()
    
    #print("Y_pred: ", y_pred)
    #print("Y_valid: ", Y_test)
    #print("Type Y_pred: ", type(y_pred))
    #print("Type Y_valid: ", type(Y_test))
    cm=confusion_matrix(Y_test, y_pred) # confusion matrix
    print(cm)

    plt.matshow(cm)
    plt.title('Confusion matrix VGG16')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig(matrix_filename)
    plt.close()
   
    print("Done!")
