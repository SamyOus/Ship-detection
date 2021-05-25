import json
import numpy as np
import pandas as pd
import math

from IPython.display import clear_output
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.metrics import Metric
import tensorflow as tf
from tensorflow import keras


def into_nparray(df_x, column_name):
    """Convert a column containing lists into a column containing np.arrays of uint8 values
    Args:
        df_x (pd.DataFrame): DataFrame to modify.
        column_name (str) : name of the column to cast
    Returns:
        pd.DataFrame: the DataFrame modified
    """
    def list_into_np_array(pixel_vals):
        return np.array(pixel_vals).astype('uint8')
    df_x[column_name] = df_x[column_name].apply(list_into_np_array)
    return df_x


def re_arange_image(df_x,column_name):
    """Reshape a column containing np.arrays of size 3*80*80 each one to a arrays of shape 80,80,3
    Args:
        df_x (pd.DataFrame): DataFrame to modify.
        column_name (str) : name of the column to reshape
    Returns:
        pd.DataFrame: the DataFrame modified
    """

    def arr_reshape(arr):
        return arr.reshape(3, int(19200/3)).T.reshape((80,80,3))
        #return arr.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
    df_x[column_name] = df_x[column_name].apply(arr_reshape)
    return df_x



def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)
        
        
def findMiddle(input_list):
    middle = float(len(input_list))/2
    if len(input_list) % 2 != 0:
        return (input_list[int(middle - .5)],input_list[int(middle - .5)])
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])
    
    
    
    
    
def show_examples(classes, images, labels, nbr_images=25, set_name=''):
    """Plots a given number of images in a subplot,
    calculates automatically the dimension of the grid
    Args:
        classes (dict): A mapping of the labels class names.
        images (pd.Series) : Series of matrices of images to plot
        labels (pd.Series) : labels of the images (0,1,...)
        nbr_images (int) : number of images to plot
        set_name (str) : Default=' ', title of the plot
    
    """

    #Find the best axes dimension for the given number of images
    dim1,dim2 = findMiddle(list(divisorGenerator(nbr_images)))
    
    size = 20
        
    #Set the nbr of images per axes, the figsize, and get the axes
    fig, axes = plt.subplots(dim1,dim2, figsize=(size,size))
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(images.iloc[i])
        ax.set_title(label = classes[labels.iloc[i]]+"_"+str(images.index[i]),
                    fontsize = size/1.3)
    
    plt.suptitle(t = f"{str(nbr_images)} exemples of : {set_name}",
                 fontsize = size,
                 va='top', y =1.0)
    
    plt.tight_layout()
    plt.show()
    
      
    
    
    
    
def plot_images_by_batches(df,batches,batch_size,classes=None,clear=None):
    """Plots a set of images by batches
    Args:
        df(pd.DataFrame): the DataFrame containing the images and the labels.
        batches (int) : number of batches (iterations)
        batch_size (int) : number of images to plot at each iteration
        clear (Bool) : Default=None, Choose if each iteration erases the previous plot    
    """
    len_df = df.shape[0]
    step = len_df // batches
    if not classes:
        classes = {0:'no_ship',1:'ship'}
    for i in range(0,len_df-1, step):
        if clear :
            print('Im in the clear section')
            clear_output(wait=True)
        df_temp = df.iloc[i:i+step].sample(batch_size)
        show_examples(classes=classes,
                      images= df_temp['data'], 
                      labels= df_temp['labels'],
                      nbr_images=batch_size,
                      set_name= f'{str(batch_size)} images from {str(i)} to {str(i+step)}')
        interaction = input("Press ENTER to continue and SPACE to stop ...")
        if interaction == ' ' or i == len_df-1 :
            break        
    print("END OF THE LOOP")
    
    
    
def plot_metrics(history):
    """Plots of a set of metrics measuring the training stage 
    The metrics are : ['loss', 'prc', 'precision', 'recall']
    Args:
        history : The result of the training stage    
    """

    metrics = ['loss', 'prc', 'precision', 'recall']
    plt.figure(figsize=(10,10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

        
def plot_cm(labels, predictions, p=0.5):
    """Calculates and plots of the confusion matrix 
    Args:
        labels : the labels of the set
        predictions : the predictions of the model
    """

    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    
    
def plot_roc(name, labels, predictions, **kwargs):
    """Calculates and plots the roc curve 
    Args:
        name (str) : name of the scatter
        labels : the labels of the set
        predictions : the predictions made by the model on the set
        
    """

    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')   
    
    
def plot_prc(name, labels, predictions, **kwargs):
    """Calculates and plots Precision/Recall Curve 
    Args:
        name (str) : name of the scatter
        labels : the labels of the set
        predictions : the predictions made by the model on the set
        
    """

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

