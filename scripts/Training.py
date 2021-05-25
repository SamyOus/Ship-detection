import json
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import time
import tensorflow as tf
from tensorflow import keras

#if error launch => launch a second time
try:
    from keras.utils import to_categorical
except AttributeError as err:

from keras.utils import to_categorical    

from tensorflow.python.keras.metrics import Metric
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D,Flatten,
                                    Dense, Dropout, Activation)


def main():
    #read dataset
    with open(r'../data/shipsnet.json') as f:
        dataset = json.load(f)
    X = np.array(dataset['data']).astype('uint8')
    y = np.array(dataset['labels']).astype('uint8')
    X = X.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
    X = X / 255
    
    #split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.20,random_state=42)
    
    #Metrics selection    
    METRICS = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    #Large number of epochs due to early stopping
    EPOCHS = 100

    #early stoppnig to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)


    #Select the initial bias to accelerate the speed of laerning 
    neg, pos = 3000, 1000
    initial_bias = np.log([pos/neg])
    output_bias = tf.keras.initializers.Constant(initial_bias)


    #Build the model
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(80, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (3, 3), input_shape=(80, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid',
                   bias_initializer=output_bias)) 
    

    #Compile the model
    model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=METRICS)

    #Train the model
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        callbacks=[early_stopping],
                        validation_data=(X_val, y_val))


    #Save the model
    ts = time.time()
    ts = str(int(ts))
    model.save('../models/CNN_'+ts+'.h5')

    
if __name__ == '__main__':
    main()
