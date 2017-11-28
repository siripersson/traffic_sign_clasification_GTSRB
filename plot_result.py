
# Plot the results from the csv file training

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import matplotlib.patches as mpatches

def plotGraph(nb_epochs):

    result = pd.read_csv(r'D:\Siri_test\weights\Fine_tune_with_GTSRB\log_frac_0_3_lr_0.001_batch10_drop_0.6_epochs_30',sep=',')
    #print(result)
   # print(result['acc'])
    x=range(1,nb_epochs+1)
    plt.plot(x, result['acc'],'g--')
    plt.plot(x, result['loss'], 'g')
    plt.plot(x, result['val_acc'], 'r--')
    plt.plot(x, result['val_loss'], 'r')
    green=mpatches.Patch(color='green', label='Training data')
    red=mpatches.Patch(color='red', label='Validation data')
    plt.legend(handles=[green, red])
    plt.ylim([0,1])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("VGG16 - Accuracy and loss curves")
    plt.show()

    return

plotGraph(30)
